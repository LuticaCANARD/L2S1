"""Exercise actual MCP -> resident HTTP -> model inference with the official SDK."""

import argparse
import asyncio
from datetime import timedelta
import json
from pathlib import Path
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT = Path(__file__).resolve().parents[1]


async def smoke(backend_url: str, timeout: float) -> None:
    request = json.loads((ROOT / "examples/warehouse.json").read_text(encoding="utf-8"))
    params = StdioServerParameters(command=sys.executable, args=[
        str(ROOT / "mcp/server.py"), "--backend-url", backend_url, "--timeout", str(timeout)])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write, read_timeout_seconds=timedelta(seconds=timeout + 10)) as session:
            await session.initialize()
            await session.read_resource("l2s1://schema/request")
            valid = await session.call_tool("l2s1_validate", {"request": request})
            capabilities = await session.call_tool("l2s1_capabilities")
            response = await session.call_tool("l2s1_decide", {"request": request})
            for result in [valid, capabilities, response]:
                if result.isError:
                    raise RuntimeError(result.content[0].text)
            envelope = response.structuredContent
            if not envelope or envelope.get("api_version") != 1:
                raise RuntimeError("missing HTTP v1 envelope")
            if [item["id"] for item in envelope["results"]] != [item["id"] for item in request["decisions"]]:
                raise RuntimeError("decision IDs do not match the request")
            for item in envelope["results"]:
                if item["status"] not in {"selected", "abstained"}:
                    raise RuntimeError("unknown decision status")
                if item["status"] == "abstained":
                    value = item["value"]
                    selected = value.get("value") if value["type"] == "binary" else value.get("selected")
                    if selected is not None or not item["abstention_reasons"]:
                        raise RuntimeError("abstention was not preserved")
            print(json.dumps({"validation": valid.structuredContent, "capabilities": capabilities.structuredContent,
                              "response": envelope}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend-url", default="http://127.0.0.1:8080")
    parser.add_argument("--timeout", type=float, default=180)
    args = parser.parse_args()
    asyncio.run(smoke(args.backend_url, args.timeout))
