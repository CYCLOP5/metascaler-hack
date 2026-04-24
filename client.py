from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List, Optional

import requests


class DSCClient:
    def __init__(self, base_url: str = "http://127.0.0.1:7860", timeout: int = 30):
        self.base = base_url.rstrip("/")
        self.timeout = timeout

    def health(self) -> Dict[str, Any]:
        r = requests.get(f"{self.base}/health", timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def reset(self, seed: Optional[int] = None, difficulty: int = 1) -> Dict[str, Any]:
        payload = {"seed": seed, "difficulty": difficulty}
        r = requests.post(f"{self.base}/reset", json=payload, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        r = requests.post(f"{self.base}/step", json={"action": action}, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def query(self, src: str, dst: str) -> Dict[str, Any]:
        return self.step({"kind": "query_network", "source_id": src, "dest_id": dst})

    def dispatch(self, routes: List[Dict[str, Any]]) -> Dict[str, Any]:
        return self.step({"kind": "dispatch_inventory", "routes": routes})

    def advance(self) -> Dict[str, Any]:
        return self.step({"kind": "advance_cycle"})

    def mcp_tools(self) -> Dict[str, Any]:
        payload = {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
        r = requests.post(f"{self.base}/mcp", json=payload, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def mcp_call(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        payload = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": name, "arguments": args},
        }
        r = requests.post(f"{self.base}/mcp", json=payload, timeout=self.timeout)
        r.raise_for_status()
        return r.json()


def _pretty(o: Dict[str, Any], compact: bool = False) -> str:
    if compact:
        return json.dumps(o, separators=(",", ":"))
    return json.dumps(o, indent=2)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:7860")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("health")
    pr = sub.add_parser("reset")
    pr.add_argument("--seed", type=int, default=0)
    pr.add_argument("--tier", type=int, default=1)

    pq = sub.add_parser("query")
    pq.add_argument("src")
    pq.add_argument("dst")

    pd = sub.add_parser("dispatch")
    pd.add_argument("src")
    pd.add_argument("dst")
    pd.add_argument("qty", type=int)

    sub.add_parser("advance")
    sub.add_parser("tools")

    pmc = sub.add_parser("call")
    pmc.add_argument("name")
    pmc.add_argument("--args", default="{}")

    pl = sub.add_parser("loop")
    pl.add_argument("--seed", type=int, default=0)
    pl.add_argument("--tier", type=int, default=1)

    a = p.parse_args()
    c = DSCClient(a.url)

    if a.cmd == "health":
        print(_pretty(c.health()))
    elif a.cmd == "reset":
        print(_pretty(c.reset(seed=a.seed, difficulty=a.tier)))
    elif a.cmd == "query":
        print(_pretty(c.query(a.src, a.dst)))
    elif a.cmd == "dispatch":
        print(_pretty(c.dispatch([{"src": a.src, "dst": a.dst, "qty": a.qty}])))
    elif a.cmd == "advance":
        print(_pretty(c.advance()))
    elif a.cmd == "tools":
        print(_pretty(c.mcp_tools()))
    elif a.cmd == "call":
        print(_pretty(c.mcp_call(a.name, json.loads(a.args))))
    elif a.cmd == "loop":
        obs = c.reset(seed=a.seed, difficulty=a.tier)
        print("reset", _pretty({"step": obs["step"], "nodes": len(obs["nodes"])}))
        for t in range(30):
            out = c.advance()
            if out.get("done"):
                print(f"done step={out['step']} reward={out['reward']} term={out['metadata'].get('terminal')}")
                break
    return 0


if __name__ == "__main__":
    sys.exit(main())
