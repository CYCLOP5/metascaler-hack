from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List, Optional


def _pretty(o: Any, compact: bool = False) -> str:
    if compact:
        return json.dumps(o, separators=(",", ":"), default=str)
    return json.dumps(o, indent=2, default=str)


def _try_mcp_client(url: str):
    try:
        from openenv.core.mcp_client import MCPToolClient

        return MCPToolClient(base_url=url).sync()
    except Exception as e:
        print(f"err mcp init: {e}", file=sys.stderr)
        sys.exit(2)


def cmd_health(url: str) -> None:
    import requests

    r = requests.get(f"{url.rstrip('/')}/health", timeout=10)
    print(_pretty(r.json()))


def cmd_tools(url: str) -> None:
    with _try_mcp_client(url) as env:
        env.reset()
        tools = env.list_tools()
        print(_pretty([{"name": t.name, "schema": getattr(t, "input_schema", None) or getattr(t, "inputSchema", None)} for t in tools]))


def cmd_loop(url: str, tier: int, seed: int) -> None:
    with _try_mcp_client(url) as env:
        env.reset(difficulty=tier, seed=seed)
        for t in range(30):
            res = env.call_tool("advance_cycle")
            print(f"step {t} result:", _pretty(res, compact=True))
            meta = res if isinstance(res, dict) else {}
            if meta.get("done"):
                break


def cmd_query(url: str, src: str, dst: str) -> None:
    with _try_mcp_client(url) as env:
        env.reset()
        print(_pretty(env.call_tool("query_network", source_id=src, dest_id=dst)))


def cmd_dispatch(url: str, src: str, dst: str, qty: int, tier: int, seed: int) -> None:
    with _try_mcp_client(url) as env:
        env.reset(difficulty=tier, seed=seed)
        print(_pretty(env.call_tool("dispatch_inventory", routes=[{"src": src, "dst": dst, "qty": qty}])))


def cmd_probe(url: str) -> None:
    import requests

    base = url.rstrip("/")
    r = requests.post(f"{base}/reset", json={"seed": 7, "difficulty": 1}, timeout=15)
    print("probe /reset:", r.status_code, _pretty(r.json(), compact=True)[:200])
    r = requests.post(f"{base}/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"}, timeout=15)
    print("probe /mcp list:", r.status_code, _pretty(r.json(), compact=True)[:300])


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8000")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("health")
    sub.add_parser("tools")
    sub.add_parser("probe")
    pl = sub.add_parser("loop")
    pl.add_argument("--tier", type=int, default=1)
    pl.add_argument("--seed", type=int, default=0)
    pq = sub.add_parser("query")
    pq.add_argument("src")
    pq.add_argument("dst")
    pd = sub.add_parser("dispatch")
    pd.add_argument("src")
    pd.add_argument("dst")
    pd.add_argument("qty", type=int)
    pd.add_argument("--tier", type=int, default=1)
    pd.add_argument("--seed", type=int, default=0)

    a = p.parse_args()
    if a.cmd == "health":
        cmd_health(a.url)
    elif a.cmd == "tools":
        cmd_tools(a.url)
    elif a.cmd == "probe":
        cmd_probe(a.url)
    elif a.cmd == "loop":
        cmd_loop(a.url, a.tier, a.seed)
    elif a.cmd == "query":
        cmd_query(a.url, a.src, a.dst)
    elif a.cmd == "dispatch":
        cmd_dispatch(a.url, a.src, a.dst, a.qty, a.tier, a.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
