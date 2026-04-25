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

## trainer metrics

During the final HF Space run, the most important metrics are:


| metric                              | meaning                                                                                                                                             |
| ----------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| `rewards/reward_func/mean`          | cumulative environment reward from replayed tool actions                                                                                            |
| `rewards/schema_reward_func/mean`   | dense last-step validity/action reward                                                                                                              |
| `rewards/terminal_reward_func/mean` | MILP terminal reward; non-zero means episodes reached step 30 and called the verifier                                                               |
| `reward`                            | combined reward seen by GRPO                                                                                                                        |
| `reward_std`                        | reward spread across generated completions; non-zero gives GRPO contrast                                                                            |
| `frac_reward_zero_std`              | fraction of groups with no reward variation; `0` is healthy                                                                                         |
| `grad_norm`                         | non-zero gradient updates indicate training is not dead                                                                                             |
| `completions/clipped_ratio`         | fraction of generations hitting `DSC_MAX_COMPLETION`; high values mean inefficient stopping, not invalid reward if terminal reward remains non-zero |


The final evidence run uses `DSC_MAX_STEPS=400`, `DSC_NUM_GEN=8`, and `DSC_MAX_COMPLETION=768`. Training artifacts are written into the uploaded LoRA repo as `training_metrics.json`, `training_metrics.csv`, `training_summary.json`, and `training_curve.png`.

Final run summary:


| metric                       | result             |
| ---------------------------- | ------------------ |
| runtime                      | 17,469.9s / 4h 51m |
| final combined reward        | 1.304              |
| max combined reward          | 1.365              |
| final cumulative env reward  | 0.852              |
| final terminal MILP reward   | 0.226              |
| max terminal MILP reward     | 0.241              |
| final `reward_std`           | 0.079              |
| final `frac_reward_zero_std` | 0.0                |
| final `grad_norm`            | 0.045              |
| final KL                     | 0.0077             |


The completion clipping ratio stayed high, which means the model often used the full generation budget. This does not invalidate the reward: only parsed JSON tool actions are replayed, invalid tails are ignored, and terminal reward is only emitted when the environment reaches the 30-step MILP verifier.

## numerical safety

- terminal clamp `[0, 1]` avoids unbounded reward under a near-zero agent_cost
- `EPS = 1e-6` in `agent_cost` divisor
- infinite milp (timeout / infeasible) returns `float('inf')` and terminal = 0.0

