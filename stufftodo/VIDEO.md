# 2-minute demo video script

Target: under 120 seconds. Record in 16:9. Use `stufftodo/demo_animation.html` as the main visual, with short cuts to the live Space, README, Trackio, and the LoRA repo.

## recording setup

- Open `stufftodo/demo_animation.html` locally in a browser.
- Open the environment Space: https://huggingface.co/spaces/AceofStades/dsc_co
- Open the Trackio dashboard: https://huggingface.co/spaces/AceofStades/dsc-co-trackio
- Open the LoRA repo: https://huggingface.co/AceofStades/dsc-co-grpo-lora
- Open `README.md`, `docs/curriculum.md`, `docs/milp-formulation.md`, and `docs/anti-hacking.md` for quick evidence cuts.

## shot list

### scene 1 (0:00-0:13) hook

**screen**: Start at the top of `stufftodo/demo_animation.html`. Show the animated supplier -> warehouse -> retail flow.

**voiceover**: "This is openenv-dsc-co: a 30-step supply-chain planning environment for language agents. The challenge is delayed consequences. Ship too greedily early, and later demand becomes expensive or impossible to satisfy."

**caption**: `30-step planning with tool actions`

### scene 2 (0:13-0:28) curriculum and tier diagrams

**screen**: Scroll to `Curriculum tiers`. Pause on tier 1, then pan across tier 2, tier 3, and tier 4.

**voiceover**: "The curriculum scales from a single route to larger networks: tier two has 3 suppliers, 5 warehouses, and 10 retailers; tier three adds longer lead times and jitter; tier four adds seasonality and severe disruptions."

**caption**: `Tier 1 -> Tier 4: same API, harder planning`

### scene 3 (0:28-0:45) live environment

**screen**: Switch to the public Space. Click `Reset`, then show a valid tool call:

```text
Type: call_tool
Tool Name: query_network
Arguments: {"source_id": "S0", "dest_id": "W0"}
```

Then briefly show examples for `dispatch_inventory` and `advance_cycle` from the README.

**voiceover**: "The agent does not directly edit state. It acts through three typed tools: query the graph, dispatch inventory, and advance the cycle. That makes the policy inspectable and keeps invalid actions easy to catch."

**caption**: `query_network -> dispatch_inventory -> advance_cycle`

### scene 4 (0:45-1:05) verifier reward

**screen**: Return to `demo_animation.html` and scroll to `The reward is a verifier`. Then cut to `docs/milp-formulation.md` or `server/solver.py`.

**voiceover**: "The terminal reward is not judged by another language model. At step thirty, the same scenario is solved as a min-cost-flow MILP with CBC. The score is optimal cost over agent cost, clamped from zero to one."

**caption**: `terminal = clip(optimal_cost / agent_cost, 0, 1)`

### scene 5 (1:05-1:22) anti-hacking

**screen**: Show `docs/anti-hacking.md` or the README anti-hacking section.

**voiceover**: "The environment also blocks obvious reward hacks. Negative quantities, float quantities, phantom edges, and malformed tool calls are rejected or terminate with penalties. Dense reward is capped, so loops cannot farm their way past the verifier."

**caption**: `strict schema + invalid-action gates + capped dense reward`

### scene 6 (1:22-1:43) training evidence

**screen**: Show the `Final GRPO run` section in the animation, then switch to Trackio or `assets/training_curve.png`.

**voiceover**: "The final training run used Llama 3.2 3B with TRL GRPO and Unsloth on Hugging Face Spaces. Over 400 steps, combined reward rose from 0.622 to 1.304, and terminal MILP reward rose from 0.052 to 0.226."

**caption**: `400 GRPO steps: reward 0.622 -> 1.304`

### scene 7 (1:43-1:56) artifacts

**screen**: Open the LoRA repo and show `training_metrics.csv`, `training_metrics.json`, `training_summary.json`, and `training_curve.png`.

**voiceover**: "The adapter, plots, CSV metrics, JSON metrics, and model card are all public, so the run is auditable after the Space finishes."

**caption**: `public LoRA + raw metrics + training curve`

### scene 8 (1:56-2:00) close

**screen**: End on the README links or the footer of `demo_animation.html`.

**voiceover**: "The Space is live, the training artifacts are public, and the reward path is deterministic."

**caption**: `hf.co/spaces/AceofStades/dsc_co`

## exact tool examples for the Space

Use these if the browser form is visible in the recording:

```text
Type: call_tool
Tool Name: query_network
Arguments: {"source_id": "S0", "dest_id": "W0"}
```

```text
Type: call_tool
Tool Name: dispatch_inventory
Arguments: {"source_id": "S0", "dest_id": "W0", "quantity": 50}
```

```text
Type: call_tool
Tool Name: advance_cycle
Arguments: {}
```

## assets checklist

- [x] `stufftodo/demo_animation.html` - main animated visual with tier diagrams
- [x] `assets/terminal_bars.png` - baseline terminal rewards
- [x] `assets/gap_hist.png` - baseline optimality gaps
- [x] `assets/training_curve.png` - final GRPO reward/loss curve
- [x] public Trackio run - `AceofStades-1777132468`
- [x] LoRA repo metrics - `training_metrics.csv`, `training_metrics.json`, `training_summary.json`
- [ ] YouTube URL - add to README and `BLOG.md` after upload

## narration rules

- Do not say "language-model judge" except to clarify that there is no LLM judge in the reward path.
- Emphasize `tier 2`, `tier 3`, and `tier 4` visually so the video does not look like a one-node toy.
- Keep the reward explanation concrete: deterministic MILP verifier, bounded terminal score, public metrics.
