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
| qwen2.5-coder-7b  |<--mcp---->|   dsc env (state)  |
+-------------------+           +--------------------+
                                          |
                                          | finalize
                                          v
                                +--------------------+
                                | pulp cbc milp oracle|
                                +--------------------+
```

## runtime layers

1. `src/models.py` pydantic v2 schemas; tagged-union discriminator on `kind`
2. `src/env.py` authoritative state machine, `DSCEnv` class; registers `FastMCP` tools inline
3. `src/solver.py` pulp min-cost flow; cbc solver pinned at `timeLimit=30`
4. `src/server.py` fastapi app via `openenv.core.env_server.create_app` with a manual fallback for `/reset`, `/step`, `/state`, `/mcp`
5. `train.py` trl + unsloth + qwen2.5-coder-7b; `environment_factory=DSCToolEnv`

## data flow

- reset: trainer or http client posts `{seed, difficulty}` -> env builds scenario -> returns observation
- step: action validated -> handler mutates inventory + pipeline + cost -> shaping reward computed -> observation returned
- advance_cycle: time ticks, arrivals land, retail demand deducted, holding cost accrues, penalty if unmet
- finalize at step 30: pulp milp solves on the exact same scenario -> terminal = `optimal / agent` clamped `[0, 1]`

## isolation

- one `DSCEnv` instance per rollout (trl `environment_factory` contract)
- state lives in the instance, not in module globals
- immutable adjacency set + immutable scenario prevent mid-episode topology mutation
