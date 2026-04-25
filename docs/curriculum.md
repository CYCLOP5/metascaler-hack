# curriculum

## tier table


| tier | suppliers | warehouses | retail | lead times  | demand model                  | disruptions           | supplier step cap |
| ---- | --------- | ---------- | ------ | ----------- | ----------------------------- | --------------------- | ----------------- |
| 1    | 1         | 1          | 1      | L = 1       | static int in [3, 8]          | none                  | 200               |
| 2    | 3         | 5          | 10     | L = 1       | gaussian, mean in [6, 14]     | none                  | 400               |
| 3    | 5         | 10         | 20     | L in [1, 5] | gaussian, mean in [7, 17]     | minor capacity jitter | 600               |
| 4    | 7         | 14         | 28     | L in [1, 7] | seasonal sin + gaussian noise | severe strikes        | 800               |


## node specs

- suppliers: initial inventory `[200, 500]`, max capacity 1000, holding 0.05
- warehouses: initial inventory `[20, 80]`, max capacity 400, holding 0.20
- retail: initial inventory `[10, 40]`, max capacity 120, holding 0.50

## edge specs

- supplier -> warehouse: unit cost `[0.5, 2.0]`, capacity `[40, 120]`
- warehouse -> retail: unit cost `[0.3, 1.5]`, capacity `[20, 80]`, lead time `max(1, L - 1)`

## shortage penalty

`P = 25.0` per unmet unit. strictly larger than the sum of all per-step edge costs over any feasible route, enforcing demand fulfillment as the dominant objective.

## gating

- `reset(difficulty=1..4)` is exposed
- tier 4 should only be used once the ema of the terminal reward over the last 50 rollouts exceeds 0.85
- the env does not auto-escalate; the trainer (or an external controller) chooses tier

## reproducibility

- `reset(seed, difficulty)` is deterministic given `(seed, difficulty)`
- all randomness flows through `random.Random(seed)` instantiated per reset
- scenarios are frozen dataclasses; no in-episode mutation of the topology

