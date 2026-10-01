from __future__ import annotations

import asyncio
import json
import sys
import unittest

from l2s1 import L2S1Error
from l2s1.stdio import StdioClient


class CompiledRpcLifetime(unittest.IsolatedAsyncioTestCase):
    async def test_timeout_discards_late_reply_without_replay_and_keeps_next_call_correlated(self) -> None:
        # Controlled RPC peer: transport lifetime evidence, not native inference.
        code = '''import json, sys, time
for line in sys.stdin:
    call = json.loads(line)
    time.sleep(0.05)
    print(json.dumps({"id": call["id"], "result": call["body"]}), flush=True)
'''
        child = await asyncio.create_subprocess_exec(sys.executable, "-u", "-c", code,
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE)
        client = StdioClient(child, 5000)
        try:
            with self.assertRaises(TimeoutError):
                await client._request("/v1/decisions", json.dumps({"state": 1}), 5)
            self.assertEqual(await client._request("/v1/decisions", json.dumps({"state": 2}), 5000), {"state": 2})
            self.assertEqual(client._counter, 2)  # No retry after the timeout.
            task = asyncio.create_task(client._request("/v1/decisions", json.dumps({"state": 3}), 5000))
            await asyncio.sleep(0)
            await client.close()
            with self.assertRaises(L2S1Error) as caught:
                await task
            self.assertEqual(caught.exception.code, "backend_closed")
        finally:
            await client.close()
            if child.returncode is None:
                child.terminate()
            await child.wait()

    async def test_close_before_reader_starts_is_idempotent(self) -> None:
        child = await asyncio.create_subprocess_exec(sys.executable, "-c", "import time; time.sleep(10)",
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE)
        client = StdioClient(child, 5000)
        try:
            await client.close()
            await client.close()
        finally:
            child.terminate()
            await child.wait()
