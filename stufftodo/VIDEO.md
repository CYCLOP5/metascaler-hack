# 2-minute demo video script

target: < 120 seconds. 16:9. screen recording + voiceover. no custom UI required.

## recording setup

Use browser tabs and local files we already have:

- local animated visual: `stufftodo/demo_animation.html`
- `README.md` rendered on GitHub or Hugging Face
- public env Space: https://huggingface.co/spaces/AceofStades/dsc_co
- Trackio dashboard: https://huggingface.co/spaces/AceofStades/dsc-co-trackio
- LoRA repo: https://huggingface.co/AceofStades/dsc-co-grpo-lora
- local plots: `assets/terminal_bars.png`, `assets/gap_hist.png`, `assets/training_curve.png`
- code snippets: `server/dsc_environment.py`, `server/solver.py`, `train.py`

## shot list

### scene 1  (0:00 - 0:12)  the hook

**screen**: open `stufftodo/demo_animation.html`. Start on the animated supplier -> warehouse -> retail flow, then cut to the README baseline table showing zero-op/greedy/milp replay gaps.

**voiceover**: "This environment trains an LLM to do something it is bad at: plan through delayed consequences. In a 30-step supply chain, greedy behavior leaves a 159 percent gap on tier one; doing nothing is worse."

**caption**: `30-step planning. MILP-verified reward.`

---

### scene 2  (0:12 - 0:30)  run the public env

**screen**: open `AceofStades/dsc_co`. Click `Reset`. Fill the OpenEnv form:

```text
Type: call_tool
Tool Name: query_network
Arguments: {"source_id": "S0", "dest_id": "W0"}
```

Then show `dispatch_inventory` and `advance_cycle` examples from README.

**voiceover**: "The agent sees a typed JSON environment and acts through three tools: query the network, dispatch inventory, and advance the cycle. The graph is partially observable, so it has to ask before it ships."

**caption**: `query_network -> dispatch_inventory -> advance_cycle`

---

### scene 3  (0:30 - 0:48)  the verifier

**screen**: `docs/milp-formulation.md` or `server/solver.py`, highlighting the min-cost-flow objective and `pulp.PULP_CBC_CMD`. Briefly cut back to the animated page's metric cards.

**voiceover**: "The reward is not a language-model judge. At step thirty, the exact same scenario is solved as a min-cost-flow MILP with CBC. Terminal reward is optimal cost over agent cost, clamped between zero and one."

**caption**: `terminal = clip(optimal_cost / agent_cost, 0, 1)`

---

### scene 4  (0:48 - 1:10)  anti-hacking

**screen**: `docs/anti-hacking.md` or README anti-hacking table. Highlight negative qty, phantom edges, and dense reward cap.

**voiceover**: "The environment is hard to game. Negative or float quantities terminate with a penalty. Phantom edges end the episode. Dense reward is capped, so cyclic reward farming cannot beat the terminal verifier."

**caption**: `strict tool schema + dense cap + phantom-edge gate`

---

### scene 5  (1:10 - 1:33)  training evidence

**screen**: Trackio dashboard or `assets/training_curve.png`, then LoRA repo file list showing `training_metrics.csv`, `training_metrics.json`, and `training_curve.png`.

**voiceover**: "We trained Llama 3.2 3B with TRL GRPO and Unsloth for 400 steps on Hugging Face Spaces. Combined reward rose from 0.622 to 1.304. Verified terminal reward rose from 0.052 to 0.226, and reward variance stayed non-zero."

**caption**: `400 GRPO steps: reward 0.622 -> 1.304`

---

### scene 6  (1:33 - 1:50)  baseline headroom

**screen**: show `assets/terminal_bars.png` and `assets/gap_hist.png`.

**voiceover**: "The baselines explain the learning target. Greedy reactive planning is far below the MILP replay ceiling. That gap is the signal this environment exposes."

**caption**: `greedy terminal 0.39 vs MILP replay 0.94 on tier 1`

---

### scene 7  (1:50 - 2:00)  close

**screen**: links table in README, ending on the env Space URL and LoRA repo URL.

**voiceover**: "The environment is live on Hugging Face, the adapter and metrics are public, and every reward number is computed by a deterministic verifier."

**caption**: `hf.co/spaces/AceofStades/dsc_co`

---

## assets checklist

- [x] `assets/terminal_bars.png` - baseline terminal rewards
- [x] `assets/gap_hist.png` - baseline optimality gaps
- [x] `assets/training_curve.png` - final GRPO reward/loss curve
- [x] `stufftodo/demo_animation.html` - local animated visual for screen recording
- [x] public Trackio run - `AceofStades-1777132468`
- [x] LoRA repo metrics - `training_metrics.csv`, `training_metrics.json`, `training_summary.json`
- [ ] YouTube URL - add to README and `BLOG.md` after upload

## voiceover tone

- calm, concrete, and judge-friendly
- no hype words
- emphasize: `30 steps`, `MILP verifier`, `reward 0.622 -> 1.304`, `terminal 0.052 -> 0.226`

