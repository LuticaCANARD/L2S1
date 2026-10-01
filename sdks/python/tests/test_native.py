from __future__ import annotations

import asyncio
import sys
import unittest
from unittest.mock import patch

from l2s1 import L2S1, L2S1Error, LoadOptions


class Startup(unittest.IsolatedAsyncioTestCase):
    async def test_invalid_options_and_missing_executable(self) -> None:
        for options in (LoadOptions(model=""), LoadOptions(model="fixture", timeout_ms=0),
                        LoadOptions(model="fixture", extra_args=("--listen=0.0.0.0:8080",))):
            with self.assertRaises(ValueError):
                await L2S1.load(options)
        with self.assertRaises(L2S1Error) as caught:
            await L2S1.load(LoadOptions(model="fixture", binary_path="/nonexistent/l2s1-fixture"))
        self.assertEqual(caught.exception.code, "spawn_failed")

    async def test_startup_deadline_and_cancellation_reap_owned_children(self) -> None:
        spawn = asyncio.create_subprocess_exec
        children: list[asyncio.subprocess.Process] = []
        started = asyncio.Event()

        async def replacement(*args: object, **kwargs: object) -> asyncio.subprocess.Process:
            # Real child process with a deliberately stalled startup; no server.
            child = await spawn(sys.executable, "-c", "import time; time.sleep(30)",
                                stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE)
            children.append(child)
            started.set()
            return child

        with patch("l2s1.native.asyncio.create_subprocess_exec", side_effect=replacement):
            with self.assertRaises(TimeoutError):
                await L2S1.load(LoadOptions(model="fixture", binary_path=sys.executable, startup_timeout_ms=30))
            self.assertIsNotNone(children[-1].returncode)
            started.clear()
            pending = asyncio.create_task(L2S1.load(LoadOptions(model="fixture", binary_path=sys.executable)))
            await started.wait()
            pending.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await pending
            self.assertIsNotNone(children[-1].returncode)


if __name__ == "__main__":
    unittest.main()
