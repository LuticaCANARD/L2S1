"""Own one resident Rust process; reuse the TypeScript runtime bundle format."""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import platform
import re
import shutil
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from .backend import L2S1Error, positive_timeout
from .http import L2S1Client
from .stdio import StdioClient
from .models import Capabilities, DecisionPolicy, DecisionRequest, DecisionResponse

VERSION = "0.1.1"
Device = Literal["cpu", "cuda", "metal"]
ExecutionMode = Literal["fresh", "prefix-reuse", "state-restore", "parallel"]


@dataclass(frozen=True)
class LoadOptions:
    model: str
    transport: Literal["stdio", "http"] = "stdio"
    binary_path: str | None = None
    runtime_dir: str | Path | None = None
    device: Device = "cpu"
    mmproj: str | None = None
    lora: str | None = None
    context: int | None = None
    batch: int | None = None
    ubatch: int | None = None
    threads: int | None = None
    gpu_layers: int | None = None
    execution_mode: ExecutionMode | None = None
    parallel_width: int | None = None
    prompt_layout: Literal["legacy", "state-first"] | None = None
    prompt_detail: Literal["minimal", "typed", "typed-examples"] | None = None
    policy: DecisionPolicy | None = None
    startup_timeout_ms: int = 120_000
    timeout_ms: int = 180_000
    extra_args: tuple[str, ...] = field(default_factory=tuple)
    on_stderr: Callable[[str], None] | None = None


def _arguments(options: LoadOptions) -> list[str]:
    if not options.model.strip():
        raise ValueError("model is required")
    if options.device not in {"cpu", "cuda", "metal"}:
        raise ValueError("unsupported device")
    if any(arg == "--stdio" or arg.startswith("--stdio=") or arg == "--listen" or arg.startswith("--listen=") for arg in options.extra_args):
        raise ValueError("--stdio/--listen are managed by L2S1.load")
    if options.transport not in {"stdio", "http"}:
        raise ValueError("unsupported transport")
    positive_timeout(options.startup_timeout_ms)
    positive_timeout(options.timeout_ms)
    args = ["--model", options.model]
    if options.device != "cpu":
        args.extend(["--device", options.device])
    for name in ("mmproj", "lora", "context", "batch", "ubatch", "threads", "gpu_layers", "execution_mode",
                 "parallel_width", "prompt_layout", "prompt_detail"):
        value = getattr(options, name)
        if value is not None:
            args.extend(["--" + name.replace("_", "-"), str(value)])
    if options.policy is not None:
        policy = DecisionPolicy.model_validate(options.policy)
        args.extend(["--min-top-probability", str(policy.min_top_probability),
                     "--min-candidate-mass", str(policy.min_candidate_mass)])
    return [*args, *options.extra_args, *( ["--listen", "127.0.0.1:0"] if options.transport == "http" else ["--stdio"] )]


def _resolve_runtime(options: LoadOptions) -> tuple[str, dict[str, str]]:
    env = dict(os.environ)
    if options.binary_path is not None:
        if options.runtime_dir is not None:
            raise ValueError("choose binary_path or runtime_dir")
        if not options.binary_path.strip():
            raise ValueError("binary_path must not be empty")
        return options.binary_path, env
    if options.runtime_dir is None:
        executable = shutil.which("l2s1")
        if executable is None:
            raise L2S1Error("Supply binary_path or a TypeScript runtime bundle runtime_dir, or install l2s1 on PATH",
                            "runtime_not_installed")
        return executable, env
    root = Path(options.runtime_dir).resolve()
    manifest = json.loads((root / "runtime-manifest.json").read_text(encoding="utf-8"))
    machine = platform.machine().lower()
    arch = {"x86_64": "x64", "amd64": "x64", "aarch64": "arm64", "arm64": "arm64"}.get(machine, machine)
    host = {"linux": "linux", "darwin": "darwin", "win32": "win32"}.get(sys.platform, sys.platform)
    if manifest.get("platform") != f"{host}-{arch}":
        raise L2S1Error("Runtime bundle does not match this OS/architecture", "unsupported_platform")
    if manifest.get("version") != VERSION:
        raise L2S1Error(f"Runtime version mismatch: expected {VERSION}", "runtime_version_mismatch")
    if options.device not in manifest.get("devices", []):
        raise L2S1Error("Runtime bundle does not support the requested device", "unsupported_device")
    executable_name = "l2s1.exe" if host == "win32" else "l2s1"
    files = manifest.get("files", [])
    if not isinstance(files, list) or not any(item.get("path") == f"bin/{executable_name}" for item in files):
        raise L2S1Error("Runtime manifest is missing its executable", "invalid_runtime")
    for item in files:
        path = (root / item["path"]).resolve()
        if not path.is_relative_to(root / "bin") or not path.is_file():
            raise L2S1Error("Runtime manifest contains a missing or invalid path", "invalid_runtime")
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if path.stat().st_size != item["bytes"] or digest != item["sha256"]:
            raise L2S1Error("Runtime bundle checksum mismatch", "invalid_runtime")
    variable = "PATH" if host == "win32" else "DYLD_LIBRARY_PATH" if host == "darwin" else "LD_LIBRARY_PATH"
    if host == "win32":
        variable = next((key for key in env if key.lower() == "path"), variable)
    env[variable] = os.pathsep.join(value for value in (str(root / "bin"), env.get(variable, "")) if value)
    return str(root / "bin" / executable_name), env


class RustProcessBackend:
    def __init__(self, child: asyncio.subprocess.Process) -> None:
        self._child = child
        self._client: L2S1Client | StdioClient | None = None
        self._drain: asyncio.Task[None] | None = None
        self._closing: asyncio.Task[None] | None = None

    @classmethod
    async def load(cls, options: LoadOptions) -> RustProcessBackend:
        args = _arguments(options)
        executable, env = _resolve_runtime(options)
        try:
            child = await asyncio.create_subprocess_exec(executable, *args, env=env,
                                                        stdin=asyncio.subprocess.PIPE if options.transport == "stdio" else asyncio.subprocess.DEVNULL,
                                                        stdout=asyncio.subprocess.PIPE if options.transport == "stdio" else asyncio.subprocess.DEVNULL,
                                                        stderr=asyncio.subprocess.PIPE,
                                                        limit=64 * 1024 * 1024)
        except OSError as error:
            raise L2S1Error(f"Could not start Rust executable: {error}", "spawn_failed") from error
        backend = cls(child)
        ready: asyncio.Future[str] = asyncio.get_running_loop().create_future()

        async def drain() -> None:
            assert child.stderr is not None
            tail = ""
            pending = ""
            try:
                while chunk := await child.stderr.read(4096):
                    text = chunk.decode("utf-8", errors="replace")
                    tail = (tail + text)[-8192:]
                    if options.on_stderr is not None:
                        options.on_stderr(text)
                    if ready.done():
                        continue
                    pending += text
                    lines = pending.split("\n")
                    pending = lines.pop()[-8192:]
                    for line in lines:
                        if options.transport == "stdio" and line.strip() == "l2s1 stdio ready" and not ready.done():
                            ready.set_result("stdio")
                        match = re.fullmatch(r"l2s1 HTTP listening on 127\.0\.0\.1:(\d+)\s*", line)
                        if match and 0 < int(match[1]) <= 65535 and not ready.done():
                            ready.set_result(f"http://127.0.0.1:{match[1]}")
                if not ready.done():
                    ready.set_exception(L2S1Error(f"Rust process exited during startup: {tail}", "startup_failed"))
            except Exception as error:
                if not ready.done():
                    ready.set_exception(error)
                # A diagnostic callback must never stop draining the child pipe.
                while await child.stderr.read(4096):
                    pass

        backend._drain = asyncio.create_task(drain())
        try:
            async with asyncio.timeout(options.startup_timeout_ms / 1000):
                url = await ready
                backend._client = StdioClient(child, options.timeout_ms) if options.transport == "stdio" else L2S1Client(url, timeout_ms=options.timeout_ms)
                await backend._client.health()
            return backend
        except BaseException:
            await backend.close()
            raise

    def _open_client(self) -> L2S1Client | StdioClient:
        if self._closing is not None or self._child.returncode is not None or self._client is None:
            raise L2S1Error("Rust process is closed", "process_closed")
        return self._client

    async def decide(self, request: DecisionRequest, *, timeout_ms: int | None = None) -> DecisionResponse:
        return await self._open_client().decide(request, timeout_ms=timeout_ms)

    async def decide_batch(self, requests: Sequence[DecisionRequest], *, timeout_ms: int | None = None) -> list[DecisionResponse]:
        return await self._open_client().decide_batch(requests, timeout_ms=timeout_ms)

    async def capabilities(self, *, timeout_ms: int | None = None) -> Capabilities:
        return await self._open_client().capabilities(timeout_ms=timeout_ms)

    async def _close(self) -> None:
        if self._client is not None:
            await self._client.close()
        if self._child.returncode is None:
            try:
                self._child.terminate()
            except ProcessLookupError:
                pass
            try:
                async with asyncio.timeout(2):
                    await self._child.wait()
            except TimeoutError:
                try:
                    self._child.kill()
                except ProcessLookupError:
                    pass
                await self._child.wait()
        if self._drain is not None:
            await self._drain

    async def close(self) -> None:
        if self._closing is None:
            self._closing = asyncio.create_task(self._close())
        await asyncio.shield(self._closing)
