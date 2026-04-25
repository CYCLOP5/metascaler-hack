# reward specification

## rubric


| id             | type             | value                       | trigger                                                      | caps                         |
| -------------- | ---------------- | --------------------------- | ------------------------------------------------------------ | ---------------------------- |
| r_schema       | dense            | +0.05                       | tool call parses + pydantic valid                            | per-episode dense sum <= 0.4 |
| r_valid        | dense            | +0.10                       | dispatch: edge exists + inv sufficient + qty strict int >= 1 | per-episode dense sum <= 0.4 |
| r_terminal     | sparse           | `clip(optimal/agent, 0, 1)` | step == 30                                                   |                              |
| r_neg_exploit  | terminal penalty | -1.0                        | qty <= 0 or float/bool qty                                   | ends episode                 |
| r_phantom_edge | terminal         | 0.0                         | dispatch over edge not in adjacency                          | ends episode                 |


## cumulative episode reward

```
R_ep = sum(r_schema) + sum(r_valid) + r_terminal + r_neg_exploit + r_phantom_edge
```

dense sum is hard-capped at 0.4 which is strictly less than the minimum possible terminal reward delta for correct planning (r_terminal ~ 0.8 at gap = 0.25). this blocks cyclic shaping-farm strategies.

## credit assignment

- dense rewards emit immediately per step; trl grpo advantage within a rollout group naturally rescales them
- terminal reward emits only at `done == true`; trainer must aggregate `cumulative` + `terminal` (the `DSCToolEnv` class handles this in `train.py`)
- three reward functions are passed to `GRPOTrainer`:
  - `reward_func` returns `env.cumulative` (episode total)
  - `schema_reward_func` returns `env.reward` (last-step shaping reward) for lower-variance gradient
  - `terminal_reward_func` returns `env.terminal` (final gap vs milp)
- if a TRL build calls reward functions without `environments`, `train.py` parses JSON tool-action lines from the completion, replays them through a fresh `DSCToolEnv`, and scores the same cumulative/step/terminal channels

## numerical safety

- terminal clamp `[0, 1]` avoids unbounded reward under a near-zero agent_cost
- `EPS = 1e-6` in `agent_cost` divisor
- infinite milp (timeout / infeasible) returns `float('inf')` and terminal = 0.0

