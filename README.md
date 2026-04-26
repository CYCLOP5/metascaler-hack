---
title: openenv-dsc-co
emoji: "📦"
colorFrom: indigo
colorTo: red
sdk: docker
app_port: 7860
app_file: app.py
pinned: true
license: apache-2.0
tags:
  - openenv
  - rlvr
  - rlve
  - grpo
  - trl
  - unsloth
  - supply-chain
  - long-horizon
  - mcp
  - milp
short_description: 30-step supply chain rlvr env with pulp milp oracle
---

# openenv-dsc-co

dynamic supply chain combinatorial orchestration. a meta openenv-compliant rlvr/rlve environment. a 30-step multi-echelon supply chain graph verified by a deterministic pulp/cbc mixed-integer linear programming oracle. 100% api/json driven. single unprivileged docker container. hf space.

## links


| artifact                        | url                                                                                                                                                                                          |
| ------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| live hf space (env server)      | [https://huggingface.co/spaces/AceofStades/dsc_co](https://huggingface.co/spaces/AceofStades/dsc_co)                                                                                         |
| hf space (training node)        | [https://huggingface.co/spaces/AceofStades/openenv-dsc-co-training](https://huggingface.co/spaces/AceofStades/openenv-dsc-co-training)                                                       |
| github source                   | [https://github.com/CYCLOP5/metascaler-hack](https://github.com/CYCLOP5/metascaler-hack)                                                                                                     |
| trained lora adapter            | [https://huggingface.co/AceofStades/dsc-co-grpo-lora](https://huggingface.co/AceofStades/dsc-co-grpo-lora)                                                                                   |
| final training curve            | [https://huggingface.co/AceofStades/dsc-co-grpo-lora/blob/main/training_curve.png](https://huggingface.co/AceofStades/dsc-co-grpo-lora/blob/main/training_curve.png) (uploaded with adapter) |
| final metrics artifacts         | [`results/training_metrics.csv`](results/training_metrics.csv), [`results/training_metrics.json`](results/training_metrics.json), [`results/training_summary.json`](results/training_summary.json) |
| trackio live training dashboard | [https://huggingface.co/spaces/AceofStades/dsc-co-trackio](https://huggingface.co/spaces/AceofStades/dsc-co-trackio) (separate dashboard Space in `trackio_space/`)                          |
| blog post                       | [BLOG.md](BLOG.md)                                                                                                                                                                           |
| 5-minute demo video             | [https://www.youtube.com/watch?v=YvUwxGc8Mzo](https://www.youtube.com/watch?v=YvUwxGc8Mzo)                                                                                                   |


## docs index

start here, then jump into whichever md interests you most:


| doc                                                  | what it shows                                                 |
| ---------------------------------------------------- | ------------------------------------------------------------- |
| [BOOTSTRAP.md](BOOTSTRAP.md)                         | full setup, local eval, Space deploy, and A100 training path   |
| [BLOG.md](BLOG.md)                                   | submission narrative: problem, verifier, training loop, proof |
| [trackio_space/README.md](trackio_space/README.md)   | live Trackio dashboard Space for reward/loss monitoring       |
| [docs/architecture.md](docs/architecture.md)         | runtime architecture, trainer/env/server data flow            |
| [docs/reward-spec.md](docs/reward-spec.md)           | reward components, dense cap, terminal verifier signal        |
| [docs/milp-formulation.md](docs/milp-formulation.md) | exact min-cost-flow MILP solved by CBC                        |
| [docs/curriculum.md](docs/curriculum.md)             | four procedural difficulty tiers and reproducibility          |
| [docs/anti-hacking.md](docs/anti-hacking.md)         | hard gates against reward/specification hacking               |


## try the public env

Open the runnable environment Space: [https://huggingface.co/spaces/AceofStades/dsc_co](https://huggingface.co/spaces/AceofStades/dsc_co)

Click `Reset`, then use `Step` with these exact values:

```text
Type:
call_tool

Tool Name:
query_network

Arguments:
{"source_id": "S0", "dest_id": "W0"}
```

Then ship inventory on the discovered edge:

```text
Type:
call_tool

Tool Name:
dispatch_inventory

Arguments:
{"routes":[{"src":"S0","dst":"W0","qty":20}]}
```

Then advance the simulator:

```text
Type:
call_tool

Tool Name:
advance_cycle

Arguments:
{}
```

Equivalent raw action payload:

```json
{"type":"call_tool","tool_name":"query_network","arguments":{"source_id":"S0","dest_id":"W0"}}
```

## tl;dr


| aspect       | value                                                    |
| ------------ | -------------------------------------------------------- |
| action space | 3 mcp tools, strict pydantic v2 validation               |
| observation  | typed json, partial-observable via `query_network`       |
| horizon      | 30 discrete steps                                        |
| reward       | dense shaping (≤ 0.4) + terminal `clip(opt/agent, 0, 1)` |
| verifier     | coin-or cbc milp, zero-variance signal                   |
| trainer      | trl grpo + unsloth + llama-3.2-3b-instruct qlora         |
| curriculum   | 4 procedurally-generated tiers with ema gating           |


## why this environment

llms default to step-wise greedy decisions. give a 7b instruct model a 30-step supply chain and early moves permanently truncate the viable solution space. this env measures and trains through that failure mode with a zero-variance, math-optimal reward.

![reward path diagram](assets/reward_path.svg)

| measurement                        | value                        |
| ---------------------------------- | ---------------------------- |
| baseline greedy gap (tier 1, n=5)  | **159%**                     |
| baseline zero-op gap (tier 1, n=5) | **448%**                     |
| milp-replay gap (tier 1, n=5)      | **7%**                       |
| gradient headroom                  | ~0.55 terminal reward points |


![baseline terminal reward](assets/terminal_bars.png)

## final training results

Final GRPO evidence run: `400` steps, `2,000` prompts, `8` generations per prompt, `max_completion_length=768`, `Llama-3.2-3B-Instruct` 4-bit QLoRA via Unsloth on an A100 Space. Raw artifacts are committed under `results/` and mirrored in the LoRA repo.

| metric | first logged step | final step | best / aggregate |
|---|---:|---:|---:|
| combined reward | 0.622 | 1.304 | max 1.365 |
| cumulative env reward | 0.505 | 0.852 | last-25 mean 0.855 |
| terminal MILP reward | 0.052 | 0.226 | max 0.241 |
| reward std | 0.387 | 0.079 | `frac_reward_zero_std=0` at final |
| GRPO train loss | — | -0.049 | 400-step run |
| runtime | — | 4h 51m | `0.023` steps/sec |

![final GRPO training curve](assets/training_curve.png)

caption: the model moves from sparse/low terminal verifier reward to stable non-zero MILP terminal reward while maintaining non-zero reward variance and gradients. The high completion clipping rate reflects inefficient stopping, but terminal reward remains active because valid parsed actions reach the 30-step verifier.

![reward components](assets/reward_components.png)

caption: the final metrics CSV is preserved under `results/`, so this plot is reproducible without relying on transient Space logs or screenshots.

![training health](assets/training_health.png)

caption: reward variance remains non-zero and KL stays controlled through the final run.

![completion diagnostics](assets/completion_diagnostics.png)

caption: the model often uses the full completion budget, but parsed action sequences still reach terminal verifier scoring.

## quick start

For a full setup guide, including local environment checks, OpenEnv deployment, and the intended Hugging Face A100 training workflow, see [BOOTSTRAP.md](BOOTSTRAP.md).

```
make install
make test
make eval N=10 TIERS="1 2"
make viz
make serve
```

then from a second shell:

```
python client.py reset --tier 1 --seed 7
python client.py query S0 W0
python client.py dispatch S0 W0 50
python client.py advance
python client.py tools
```

## hf space deployment

live hf space: [https://huggingface.co/spaces/AceofStades/dsc_co](https://huggingface.co/spaces/AceofStades/dsc_co)

reproduce the deploy:

```
huggingface-cli login
openenv push -r AceofStades/dsc_co --exclude .openenvignore
```

`-r` (aka `--repo-id`) takes `username/env-name`. `--exclude .openenvignore` is **required** — the cli's default ignore is only `.*`, `__pycache__`, `*.pyc`, so your local `env/` venv would otherwise upload (~400 mb of compiled `.so` + cbc binaries = 500 error from hf).

optional flags: `--private`, `--base-image ghcr.io/meta-pytorch/openenv-base:latest`, `--hardware cpu-basic`, `--env-var KEY=VAL`, `--secret KEY=VAL`.

manual docker alternative:

```
docker build -t openenv-dsc-co .
docker run --rm -p 7860:7860 openenv-dsc-co
```

the hf space uses port 8000 by default (`openenv.yaml`). the root `Dockerfile` is wired for port 7860 if you deploy directly as a plain docker space.

## training on huggingface spaces

we use huggingface spaces as on-demand, high-vram gpu compute nodes for grpo training. The intended full training path is the root `Dockerfile` on a Hugging Face **A100 Docker Space**. The notebooks in `notebooks/` are provided for reference and experimentation, but the final evidence run was trained through the Docker Space workflow.

workflow:

1. create a new space on huggingface (select **A100** gpu hardware for the final/full run, **Docker** sdk).
2. configure space secrets:
  - `HF_TOKEN` — a write-scope token from [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
  - `DSC_HF_REPO` — `AceofStades/dsc-co-grpo-lora` (where the trained lora will be pushed)
3. push this codebase to the space:
  ```bash
   git remote add space https://huggingface.co/spaces/<your-user>/<your-space>
   git push space master:main
  ```
4. the space will build the `Dockerfile`, spin up a lightweight fastapi server (`app.py`) to satisfy the space's port 7860 health check, and kick off `train.py` in the background. check the space logs to watch unsloth run!

training stack:

- `unsloth/Llama-3.2-3B-Instruct-bnb-4bit` 4-bit qlora, r=32
- defaults: `num_generations=4`, `max_completion_length=512`, `beta=0.04`; override with `DSC_NUM_GEN`, `DSC_MAX_COMPLETION`, `DSC_BETA`
- final-run knobs: `DSC_MAX_STEPS`, `DSC_DATA_N`, `DSC_BATCH_SIZE`, `DSC_GRAD_ACCUM`, `DSC_LR`, `DSC_EPOCHS`, `DSC_TEMP`, `DSC_SAVE_STEPS`
- checkpoint recovery: set `DSC_RESUME=1` to resume the latest checkpoint in `DSC_OUT_DIR`
- vllm/fast inference is enabled only when the GPU supports it and `vllm` is installed; otherwise training falls back to the standard unsloth path
- when TRL does not pass environments into reward functions, `train.py` locally replays JSON tool actions through `DSCToolEnv` so rewards, loss, and gradients remain non-zero
- final model uploads include `training_metrics.json`, `training_metrics.csv`, and `training_curve.png` alongside the LoRA adapter
- `trackio.log({...})` streams metrics to the trackio dashboard.

final HF Space evidence run preset:

```bash
DSC_MAX_STEPS=400
DSC_DATA_N=2000
DSC_NUM_GEN=8
DSC_MAX_COMPLETION=768
DSC_SAVE_STEPS=50
DSC_RESUME=0
DSC_DEBUG=0
DSC_LOG_COMPLETIONS=0
DSC_TRACKIO=openenv-dsc-co
DSC_TRACKIO_SPACE=AceofStades/dsc-co-trackio
```

reload the trained adapter anywhere with:

```python
from unsloth import FastLanguageModel
model, tok = FastLanguageModel.from_pretrained(
    "AceofStades/dsc-co-grpo-lora",
    max_seq_length=8192, load_in_4bit=True, fast_inference=True,
)
FastLanguageModel.for_inference(model)
```

## repo layout

canonical openenv multi-mode deployment layout:

```
openenv-dsc-co/
├── BOOTSTRAP.md              full setup, local eval, deployment, and A100 training guide
├── pyproject.toml            package metadata, OpenEnv/core deps, pytest pythonpath
├── uv.lock                   pinned local/OpenEnv dependency resolution
├── openenv.yaml              OpenEnv manifest: app server.app:app on port 8000
├── __init__.py               package marker for OpenEnv packaging
├── models.py                 root re-export shim for OpenEnv structural checks
├── client.py                 CLI for reset/query/dispatch/advance/tools/health
├── app.py                    training Space health server that starts train.py in a thread
├── README.md                 main judge-facing overview, links, examples, results
├── BLOG.md                   short submission writeup / narrative
├── Makefile                  common install/test/eval/viz/serve/docker/train commands
├── Dockerfile                GPU training Space image; launches app.py then train.py
├── .gitignore                local cache/output ignore rules
├── .dockerignore             keeps local envs, outputs, and trackio_space out of training image
├── .openenvignore            keeps training-only/local files out of OpenEnv env pushes
├── requirements.txt          environment/runtime deps: fastapi, openenv, pulp, matplotlib
├── requirements-train.txt    cuda training deps: torch, trl, unsloth, trackio
├── server/
│   ├── __init__.py
│   ├── app.py                OpenEnv/FastAPI app; browser JSON-argument adapter; fallback routes
│   ├── Dockerfile            openenv-base multi-stage build for `openenv push`
│   ├── dsc_environment.py    DSCEnv(MCPEnvironment) + 4-tier curriculum + 3 fastmcp tools
│   ├── models.py             pydantic v2 schemas (DSCAction RootModel, strict int qty)
│   ├── solver.py             pulp time-expanded min-cost flow + greedy baseline
│   └── policies.py           zero_op, greedy, optimal_replay baseline rollouts
├── tests/
│   ├── __init__.py
│   ├── test_models.py        strict-int qty, action envelope parsing, observation schema
│   ├── test_env.py           reset shapes, anti-hack gates, valid flow, horizon termination
│   └── test_solver.py        milp correctness, tier shapes, bipartite edges
├── notebooks/
│   ├── train_hf_space.ipynb  reference notebook; final training used root Dockerfile on HF A100
│   └── train_kaggle.ipynb    alternate experiment notebook, not the canonical final run path
├── docs/
│   ├── architecture.md
│   ├── reward-spec.md
│   ├── milp-formulation.md
│   ├── curriculum.md
│   └── anti-hacking.md
├── assets/
│   ├── reward_path.svg       reward/verifier flow diagram
│   ├── completion_diagnostics.png completion length and clipping diagnostics
│   ├── gap_hist.png          baseline optimality gap histogram
│   ├── reward_components.png final run reward component curves
│   ├── training_health.png   reward variance and KL diagnostics
│   ├── terminal_bars.png     baseline terminal reward comparison
│   └── training_curve.png    final GRPO reward/loss curve
├── results/
│   ├── training_metrics.csv  final LoRA repo metrics export
│   ├── training_metrics.json final LoRA repo metrics export
│   └── training_summary.json final run summary
├── stufftodo/
│   └── demo_animation.html   local animation page used for the demo video
├── trackio_space/
│   ├── README.md             Trackio dashboard Space card/deploy notes
│   ├── app.py                live Trackio dashboard app
│   └── requirements.txt      Trackio dashboard dependency
├── train.py                  HF Space GRPO training, local replay fallback, artifact upload
├── eval.py                   deterministic baseline rollout harness -> eval.json
└── viz.py                    renders gap, terminal, trajectory, and training curves
```

## mcp action space

> deep dive: [docs/architecture.md](docs/architecture.md)


| tool                 | args                               | semantics                                                                          |
| -------------------- | ---------------------------------- | ---------------------------------------------------------------------------------- |
| `query_network`      | `source_id: str, dest_id: str`     | returns `{exists, lead_time, unit_cost, capacity}`                                 |
| `dispatch_inventory` | `routes: [{src, dst, qty}]`, max 8 | strict int qty ≥ 1; deducts inv, schedules shipment                                |
| `advance_cycle`      | none                               | ticks time, processes arrivals, deducts demand, accrues costs; finalize at step 30 |


max 5 calls per cycle; `advance_cycle` resets the per-cycle counter.

## observation

```
{
  "step": 0,
  "network_status": "nominal" | "disrupted",
  "nodes": [
    {"id", "type": "supplier" | "warehouse" | "retail",
     "inventory", "max_capacity", "holding_cost", "demand_forecast"}
  ],
  "pipeline": [{"src", "dst", "qty", "arrival_step"}],
  "reward": float, "done": bool,
  "metadata": {"tier", "agent_cost", "optimal_cost", "terminal", "calls_this_cycle"}
}
```

## reward rubric

> deep dive: [docs/reward-spec.md](docs/reward-spec.md)


| component      | type     | value                   | trigger                                      | cap             |
| -------------- | -------- | ----------------------- | -------------------------------------------- | --------------- |
| r_schema       | dense    | +0.05                   | valid pydantic-parsed tool call              | sum dense ≤ 0.4 |
| r_valid        | dense    | +0.10                   | dispatch with existing edge + inv sufficient | sum dense ≤ 0.4 |
| r_terminal     | sparse   | `clip(opt/agent, 0, 1)` | step == 30                                   | —               |
| r_neg_exploit  | terminal | −1.0 + done             | qty ≤ 0 or float                             | —               |
| r_phantom_edge | terminal | 0 + done                | dispatch over edge not in adjacency          | —               |


## curriculum

> deep dive: [docs/curriculum.md](docs/curriculum.md)


| tier | suppliers | warehouses | retail | lead time | demand   | disruptions     |
| ---- | --------- | ---------- | ------ | --------- | -------- | --------------- |
| 1    | 1         | 1          | 1      | L=1       | static   | none            |
| 2    | 3         | 5          | 10     | L=1       | gaussian | none            |
| 3    | 5         | 10         | 20     | L∈[1..5]  | gaussian | capacity jitter |
| 4    | 7         | 14         | 28     | L∈[1..7]  | seasonal | severe strikes  |


## milp formulation

> deep dive: [docs/milp-formulation.md](docs/milp-formulation.md)

```
min   Σ_{e,t} c_e · x[e,t]
    + Σ_{n,t} h_n · I[n,t]
    + Σ_{n,t} P   · u[n,t]

s.t.  I[n, 0]    = I0_n
      I[n, t+1] = I[n, t] + arrivals(n, t) − departures(n, t) − d[n, t] + u[n, t]
      I[n, t]   ≤ cap_n
      x[e, t]   ≤ cap_e
      Σ_{e: src=s, t} x[e, t] ≤ sup_cap     (supplier)
      arrivals(n, t) = Σ_{e: dst=n, t−L_e ≥ 0} x[e, t − L_e]
```

solver: `pulp.PULP_CBC_CMD(msg=0, timeLimit=30)`.

## anti-hacking hard-gates

> deep dive: [docs/anti-hacking.md](docs/anti-hacking.md)


| vector                      | defense                                                                                   |
| --------------------------- | ----------------------------------------------------------------------------------------- |
| negative / zero / float qty | pydantic `strict=True, ge=1` + pre-mutation `_is_underflow_qty` check → reward −1.0, done |
| cyclic reward farming       | `MAX_CALLS_PER_CYCLE=5` + `DENSE_CAP=0.4`; holding cost > dense reward                    |
| phantom edge hallucination  | immutable `_adjacency: frozenset` built at reset → dispatch off-graph ends episode        |


## evaluation

```
make eval N=20 TIERS="1 2 3"
make viz
```

produces:

- `eval.json` with per-rollout cost, gap, terminal
- `assets/gap_hist.png` per-tier gap histograms
- `assets/terminal_bars.png` mean terminal reward bars by policy × tier

baseline optimality gap histogram

caption: non-trained baseline policies leave large optimality gaps; `optimal_replay` is the MILP-derived upper-bound behavior.

baseline terminal reward by policy

caption: terminal reward has clear headroom for RL; greedy behavior is far below the MILP replay ceiling on tier 1.

baseline numbers on 5 seeds (tier 1 / tier 2):


| policy         | tier 1 gap / terminal | tier 2 gap / terminal |
| -------------- | --------------------- | --------------------- |
| zero_op        | 4.48 / 0.19           | 1.01 / 0.51           |
| greedy         | 1.59 / 0.39           | 0.35 / 0.75           |
| optimal_replay | **0.07 / 0.94**       | **0.02 / 0.98**       |


greedy ↔ optimal_replay gap is the rl learning target (~0.55 terminal reward points on tier 1).

## tests

```
make test
```

43 tests across models, env, solver:

- strict-int qty rejection (5 cases)
- action envelope parsing (3 tool kinds + rejection)
- tier 1/2 shape invariants + determinism
- all 3 anti-hack gates under step()
- horizon termination with milp finalize
- milp correctness: `optimal_cost ≤ greedy_cost` on random tier-1 scenarios
- bipartite edge topology

## credits

meta pytorch openenv team. huggingface trl team. unsloth team. coin-or cbc. apache-2.0.