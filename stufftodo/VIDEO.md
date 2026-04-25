# 2-minute demo video script

target: < 120 seconds. 16:9. 30fps. voiceover + screen capture.

## shot list

### scene 1  (0:00 - 0:12)  the hook

**screen**: a terminal running a 7b baseline agent against the dsc env. agent_cost climbs into the red. retail node flashes empty.

**voiceover**: "language models can write code. they can't yet plan. give a 7-billion parameter model a 30-step supply chain, and it empties the closest warehouse on turn one, then watches the retailer starve for the next twenty-nine turns."

**caption**: `baseline optimality gap: 440%`

---

### scene 2  (0:12 - 0:30)  the environment

**screen**: a simple animated network diagram. 1 supplier -> 1 warehouse -> 1 retailer. labels: lead_time, capacity, unit_cost. arrows move.

**voiceover**: "openenv-dsc-co is a verifiable 30-step supply-chain environment. the agent discovers the graph via mcp tool calls - `query_network`, `dispatch_inventory`, `advance_cycle`. every request is strict-json validated. negative quantities, phantom edges, cyclic reward farming all hit hard programmatic walls."

**caption**: `3 mcp tools. 30 steps. 100% api-driven.`

---

### scene 3  (0:30 - 0:50)  the verifier

**screen**: swimlane. left: agent trajectory + agent_cost. right: pulp milp formulation (objective and constraints typed out line by line). a cbc solver log fills briefly. right side shows optimal_cost.

**voiceover**: "the reward is not opinion. at step thirty, pulp compiles the scenario into a mixed-integer linear program, coin-or cbc computes the mathematical minimum cost, and the terminal reward is the ratio - optimal over agent - clamped to zero-one. no lm-as-judge. zero-variance signal."

**caption**: `reward = clip(optimal / agent, 0, 1)`

---

### scene 4  (0:50 - 1:15)  the rl loop

**screen**: overlay of `train.py` -> `GRPOTrainer` ->  `DSCToolEnv` rollouts -> local JSON action replay fallback -> group-relative advantage formula -> unsloth.

**voiceover**: "we train llama-3.2-3b-instruct with trl grpo and unsloth 4-bit quantization on huggingface spaces. the model emits json tool actions. if trl does not hand an environment object into the reward function, train.py replays those actions locally through the same dsc environment, so dense shaping and sparse terminal reward still become a real policy-gradient signal."

**caption**: `grpo + unsloth + verified local replay on huggingface spaces`

---

### scene 5  (1:15 - 1:40)  the before / after

**screen**: split-screen. left: pre-training rollout animation. right: post-training rollout animation on the exact same seed. both overlay a running agent_cost plot. right side converges near optimal; left keeps climbing.

**voiceover**: "same seed, same scenario. before training, the agent panics, ships late, pays shortage. after training, the agent queries the network, dispatches bulk from supplier to warehouse on turn one - accepting the five-step delay - then meters small shipments to retail exactly when demand spikes."

**caption**: `baseline 0.39 terminal --> trained 0.85+ terminal`

---

### scene 6  (1:40 - 1:55)  the proof

**screen**: grafana-ish trackio dashboard. mean reward curve climbing. 43 green pytest dots. docker build success. hf space url.

**voiceover**: "forty-three tests green. docker space deploys in one command. the trackio dashboard is public - anyone can watch the reward curve climb in real time."

**caption**: `hf space live. milp-verified. 100% reproducible.`

---

### scene 7  (1:55 - 2:00)  the cta

**screen**: repo url + hf space url + command: `openenv push`.

**voiceover**: "openenv-dsc-co. go clone it."

**caption**: `github.com/CYCLOP5/metascaler-hack  /  hf.co/spaces/AceofStades/dsc_co`

---

## assets checklist

- [ ] `assets/terminal_bars.png` - rendered (eval + viz)
- [ ] `assets/gap_hist.png` - rendered (eval + viz)
- [ ] `assets/train_curve.png` - generated post-training from `viz.py train --csv ...`
- [ ] `assets/pre_vs_post.mp4` - side-by-side animation (produced separately)
- [ ] `assets/network_diagram.png` - simple supplier/warehouse/retail node diagram

## voiceover tone

- flat, technical, no excitement words
- short sentences
- reader pacing ~160 wpm for exactly 320 words in 120 seconds
- emphasis only on numbers: `440%`, `30`, `zero-variance`, `optimal / agent`
