# architecture

## component graph

```
+-------------------+           +------------------+
|  trl grpo trainer |<--------->|  dsctoolenv x N  |
|  + unsloth vllm   |           |  (in-process)    |
+---------+---------+           +---------+--------+
          |                               |
          | rollout prompt                | self._env.step(action)
          v                               v
+-------------------+           +--------------------+
| llama-3.2-3b-instruct |<--mcp---->|   dsc env (state)  |
+-------------------+           +--------------------+
                                          |
                                          | finalize
                                          v
                                +--------------------+
                                | pulp cbc milp oracle|
                                +--------------------+
```

## runtime layers

1. `server/models.py` pydantic v2 schemas; `DSCAction` is a `RootModel` discriminated on `kind`
2. `server/dsc_environment.py` authoritative state machine, `DSCEnv(MCPEnvironment)` with `SUPPORTS_CONCURRENT_SESSIONS=True`; registers `FastMCP` tools inline
3. `server/solver.py` pulp min-cost flow; cbc solver pinned at `timeLimit=30`
4. `server/app.py` fastapi app via `openenv.core.env_server.http_server.create_app` using `CallToolAction` / `CallToolObservation` with a manual fallback for `/reset`, `/step`, `/state`, `/mcp`
5. `models.py` (root) re-export shim of `server.models.*` to satisfy `openenv push` structural check
6. `train.py` trl + unsloth + llama-3.2-3b-instruct; `environment_factory=DSCToolEnv`

## data flow

- reset: trainer or http client posts `{seed, difficulty}` -> env builds scenario -> returns observation
- step: action validated -> handler mutates inventory + pipeline + cost -> shaping reward computed -> observation returned
- advance_cycle: time ticks, arrivals land, retail demand deducted, holding cost accrues, penalty if unmet
- finalize at step 30: pulp milp solves on the exact same scenario -> terminal = `optimal / agent` clamped `[0, 1]`

## isolation

- one `DSCEnv` instance per rollout (trl `environment_factory` contract)
- state lives in the instance, not in module globals
- immutable adjacency set + immutable scenario prevent mid-episode topology mutation
