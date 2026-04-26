# behavior trace

this is the plain-language before/after story behind the reward curves.

## business story

the supply chain has one simple failure mode: if the planner panic-ships too much inventory early, the retailer may look safe for a few turns, but the warehouse and supplier get depleted at the wrong time. later demand then causes expensive shortage penalties.

the better behavior is not "ship more." it is "ship the right amount early enough for lead time, and keep replenishment moving without flooding the network."

## trace caveat

the final evidence run used `DSC_LOG_COMPLETIONS=0`, so raw trained-model JSON completions were not preserved in the repo. the quantitative trained result is therefore shown through the final metrics CSV and reward curves.

this page shows a concrete environment trace for the missing qualitative piece:

- **before**: actual reactive greedy baseline rollout
- **after target**: actual MILP replay rollout on the same seed, showing the planned behavior GRPO is rewarded toward

both traces are tier 1, seed 7, from the same environment.

before and after behavior trace

## before: reactive / greedy

the reactive baseline keeps shipping immediately because it sees demand nearby.

```json
{"kind":"dispatch_inventory","routes":[{"src":"W0","dst":"R0","qty":14}]}
{"kind":"dispatch_inventory","routes":[{"src":"W0","dst":"R0","qty":14},{"src":"S0","dst":"W0","qty":49}]}
{"kind":"advance_cycle"}
{"kind":"dispatch_inventory","routes":[{"src":"W0","dst":"R0","qty":10}]}
{"kind":"dispatch_inventory","routes":[{"src":"W0","dst":"R0","qty":10}]}
```

final result on this replay:


| metric          | value    |
| --------------- | -------- |
| agent cost      | 2157.645 |
| terminal reward | 0.423    |


## after target: planned / verifier replay

the planned replay waits when stock is sufficient, then sends smaller batches aligned with demand and lead time.

```json
{"kind":"advance_cycle"}
{"kind":"advance_cycle"}
{"kind":"dispatch_inventory","routes":[{"src":"W0","dst":"R0","qty":3}]}
{"kind":"advance_cycle"}
{"kind":"dispatch_inventory","routes":[{"src":"W0","dst":"R0","qty":6}]}
{"kind":"advance_cycle"}
{"kind":"dispatch_inventory","routes":[{"src":"W0","dst":"R0","qty":6}]}
```

final result on the same tier and seed:


| metric          | value   |
| --------------- | ------- |
| agent cost      | 952.025 |
| terminal reward | 0.959   |


## trained run evidence

the final trained LoRA run moved in the right direction quantitatively:


| metric                | first logged step | final step |
| --------------------- | ----------------- | ---------- |
| combined reward       | 0.622             | 1.304      |
| cumulative env reward | 0.505             | 0.852      |
| terminal MILP reward  | 0.052             | 0.226      |


that does not mean the trained policy reached MILP replay quality; it means GRPO found non-zero terminal-verifier signal and improved toward the planned behavior. for future exact qualitative trained samples, run with:

```bash
DSC_LOG_COMPLETIONS=1
```

and keep the best parsed completion JSON alongside `training_metrics.csv`.