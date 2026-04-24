from __future__ import annotations

import os
from typing import Any, Dict, Optional

from pydantic import BaseModel

try:
    from .dsc_environment import DSCEnv
    from .models import DSCActionEnvelope, DSCObservation
except ImportError:
    from server.dsc_environment import DSCEnv
    from server.models import DSCActionEnvelope, DSCObservation


class ResetRequest(BaseModel):
    seed: Optional[int] = None
    difficulty: int = 1


def _build_app():
    max_concurrent = int(os.getenv("MAX_CONCURRENT_ENVS", "8"))
    try:
        try:
            from openenv.core.env_server.http_server import create_app as _oe_create_app
        except Exception:
            from openenv.core.env_server import create_app as _oe_create_app
        from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation

        print("init oe app")
        return _oe_create_app(
            DSCEnv,
            CallToolAction,
            CallToolObservation,
            env_name="dsc_co",
            max_concurrent_envs=max_concurrent,
        )
    except Exception:
        print("init fa app")
        from fastapi import FastAPI, HTTPException

        fa = FastAPI(title="openenv-dsc-co", version="0.1.0")
        env = DSCEnv()

        @fa.get("/health")
        def health() -> Dict[str, Any]:
            return {"ok": True}

        @fa.post("/reset")
        def reset(req: ResetRequest) -> Dict[str, Any]:
            obs = env.reset(seed=req.seed, difficulty=req.difficulty)
            return obs.model_dump()

        @fa.post("/step")
        def step(payload: Dict[str, Any]) -> Dict[str, Any]:
            try:
                obs = env.step(payload)
                return obs.model_dump()
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"err step: {e}")

        @fa.get("/state")
        def state() -> Dict[str, Any]:
            st = env.state
            return {
                "episode_id": getattr(st, "episode_id", ""),
                "step_count": getattr(st, "step_count", 0),
            }

        @fa.post("/mcp")
        def mcp(payload: Dict[str, Any]) -> Dict[str, Any]:
            method = payload.get("method", "")
            rpc_id = payload.get("id", 0)
            if method == "tools/list":
                return {
                    "jsonrpc": "2.0",
                    "id": rpc_id,
                    "result": {
                        "tools": [
                            {"name": "query_network", "description": "edge info"},
                            {"name": "dispatch_inventory", "description": "ship routes"},
                            {"name": "advance_cycle", "description": "tick time"},
                        ]
                    },
                }
            if method == "tools/call":
                params = payload.get("params", {})
                name = params.get("name", "")
                args = params.get("arguments", {})
                if name == "query_network":
                    res = env._handle_query(args.get("source_id", ""), args.get("dest_id", ""))
                elif name == "dispatch_inventory":
                    res = env._handle_dispatch(args.get("routes", []))
                elif name == "advance_cycle":
                    res = env._handle_advance()
                else:
                    return {"jsonrpc": "2.0", "id": rpc_id, "error": {"code": -32601, "message": "no tool"}}
                return {"jsonrpc": "2.0", "id": rpc_id, "result": res}
            return {"jsonrpc": "2.0", "id": rpc_id, "error": {"code": -32600, "message": "bad req"}}

        return fa


app = _build_app()


def main():
    import uvicorn

    port = int(os.getenv("PORT", os.getenv("APP_PORT", "8000")))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
