---
title: openenv-dsc-co
emoji: "📦"
colorFrom: indigo
colorTo: red
sdk: docker
app_port: 7860
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

| artifact | url |
|---|---|
| live hf space (env server) | https://huggingface.co/spaces/AceofStades/dsc_co |
| kaggle training notebook | https://www.kaggle.com/code/cycl0p5/metahack |
| github source | https://github.com/CYCLOP5/metascaler-hack |
| trained lora adapter | https://huggingface.co/AceofStades/dsc-co-grpo-lora (after kaggle run) |
| trackio live training dashboard | https://huggingface.co/spaces/AceofStades/dsc-co-trackio (auto-created on first run) |
| blog post | [BLOG.md](BLOG.md) |
| 2-minute demo script | [stufftodo/VIDEO.md](stufftodo/VIDEO.md) |

## tl;dr

| aspect | value |
|---|---|
| action space | 3 mcp tools, strict pydantic v2 validation |
| observation | typed json, partial-observable via `query_network` |
| horizon | 30 discrete steps |
| reward | dense shaping (≤ 0.4) + terminal `clip(opt/agent, 0, 1)` |
| verifier | coin-or cbc milp, zero-variance signal |
| trainer | trl grpo + unsloth + qwen2.5-coder-7b-instruct |
| curriculum | 4 procedurally-generated tiers with ema gating |

## why this environment

llms default to step-wise greedy decisions. give a 7b instruct model a 30-step supply chain and early moves permanently truncate the viable solution space. this env measures and trains through that failure mode with a zero-variance, math-optimal reward.

| measurement | value |
|---|---|
| baseline greedy gap (tier 1, n=5) | **159%** |
| baseline zero-op gap (tier 1, n=5) | **448%** |
| milp-replay gap (tier 1, n=5) | **7%** |
| gradient headroom | ~0.55 terminal reward points |

![baseline terminal reward](assets/terminal_bars.png)

## quick start

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

live hf space: https://huggingface.co/spaces/AceofStades/dsc_co

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

## training on kaggle

live kaggle notebook: https://www.kaggle.com/code/cycl0p5/metahack

**mac / apple silicon cannot build xformers** (clang rejects `-fopenmp`). the env server runs fine on mac via pip, but grpo needs a cuda linux host. use kaggle t4x2 or p100.

workflow:
1. open `notebooks/train_kaggle.ipynb` on kaggle (or use the hosted link above)
2. `settings` → accelerator = `t4 x2` (or `p100`), internet = `on`
3. add two secrets under `add-ons → secrets`:
   - `HF_TOKEN` — a write-scope token from https://huggingface.co/settings/tokens
   - `DSC_HF_REPO` — `AceofStades/dsc-co-grpo-lora` (the repo the trained lora will be pushed to)
4. optional: also set `DSC_HF_SPACE=AceofStades/dsc_co` so the notebook clones the live env repo instead of github
5. `run all`. the notebook:
   - installs unsloth + trl + vllm + pulp
   - runs baseline rollouts (zero-op, greedy, milp-replay) to produce a pre-training gap curve
   - loads qwen2.5-coder-7b 4-bit, trains with grpo for `MAX_STEPS=120` steps by default (~1-2h on t4x2)
   - saves the lora adapter to `/kaggle/working/grpo_dsc_co/final` and uploads to `DSC_HF_REPO`
   - runs trained-policy rollouts for the post-training gap

training stack:
- `unsloth/Qwen2.5-Coder-7B-Instruct` 4-bit qlora, r=32
- `num_generations=8`, `max_completion_length=4096`, `beta=0.04`, `loss_type=bnpo`
- `use_vllm=True`, `vllm_mode=colocate`
- `trackio.log({...})` streams `reward/mean_cumulative`, `reward/mean_step`, `reward/mean_terminal` every grpo step to a public hf space that auto-creates on first call (or set `DSC_TRACKIO_SPACE=AceofStades/dsc-co-trackio`)

fall back to `unsloth/Qwen2.5-Coder-3B-Instruct` if t4 vram is tight.

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
├── pyproject.toml            openenv-core[core] + pulp deps, pytest pythonpath
├── uv.lock                   pinned resolution for openenv-base docker builds
├── openenv.yaml              spec_version: 1, type: space, runtime: fastapi, app: server.app:app, port: 8000
├── __init__.py               package marker
├── models.py                 re-export shim for openenv push structural check
├── client.py                 MCPToolClient CLI (tools, loop, query, dispatch, probe, health)
├── README.md                 this file (hf space frontmatter at top)
├── BLOG.md                   hackathon submission post
├── VIDEO.md                  2-minute demo script
├── Makefile                  install, serve, test, eval, viz, bench, docker, push-space, train
├── Dockerfile                python 3.11-slim + coinor-cbc + non-root, port 7860 (hf space direct)
├── requirements.txt          env-only pip deps (mac-safe)
├── requirements-train.txt    cuda training deps (torch, trl, unsloth)
├── server/
│   ├── __init__.py
│   ├── app.py                create_app(DSCEnv, CallToolAction, CallToolObservation) + fastapi fallback
│   ├── Dockerfile            openenv-base multi-stage build for `openenv push`
│   ├── dsc_environment.py    DSCEnv(MCPEnvironment) + 4-tier curriculum + 3 fastmcp tools
│   ├── models.py             pydantic v2 schemas (DSCAction RootModel, strict int qty)
│   ├── solver.py             pulp time-expanded min-cost flow + greedy baseline
│   └── policies.py           zero_op, greedy, optimal_replay baseline rollouts
├── tests/
│   ├── test_models.py        strict-int qty, action envelope parsing, observation schema
│   ├── test_env.py           reset shapes, anti-hack gates, valid flow, horizon termination
│   └── test_solver.py        milp correctness, tier shapes, bipartite edges
├── notebooks/
│   ├── train_kaggle.ipynb    end-to-end grpo run on kaggle t4x2/p100
│   └── demo.ipynb            before/after rollout plots
├── docs/
│   ├── architecture.md
│   ├── reward-spec.md
│   ├── milp-formulation.md
│   ├── curriculum.md
│   └── anti-hacking.md
├── assets/                   plots (gap_hist.png, terminal_bars.png, ...)
├── train.py                  trl grpo + unsloth + trackio + qwen2.5-coder-7b
├── eval.py                   batch rollout harness -> eval.json
└── viz.py                    gap histograms, terminal bars, training curves
```

## mcp action space

| tool | args | semantics |
|---|---|---|
| `query_network` | `source_id: str, dest_id: str` | returns `{exists, lead_time, unit_cost, capacity}` |
| `dispatch_inventory` | `routes: [{src, dst, qty}]`, max 8 | strict int qty ≥ 1; deducts inv, schedules shipment |
| `advance_cycle` | none | ticks time, processes arrivals, deducts demand, accrues costs; finalize at step 30 |

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

| component | type | value | trigger | cap |
|---|---|---|---|---|
| r_schema | dense | +0.05 | valid pydantic-parsed tool call | sum dense ≤ 0.4 |
| r_valid | dense | +0.10 | dispatch with existing edge + inv sufficient | sum dense ≤ 0.4 |
| r_terminal | sparse | `clip(opt/agent, 0, 1)` | step == 30 | — |
| r_neg_exploit | terminal | −1.0 + done | qty ≤ 0 or float | — |
| r_phantom_edge | terminal | 0 + done | dispatch over edge not in adjacency | — |

## curriculum

| tier | suppliers | warehouses | retail | lead time | demand | disruptions |
|---|---|---|---|---|---|---|
| 1 | 1 | 1 | 1 | L=1 | static | none |
| 2 | 3 | 5 | 10 | L=1 | gaussian | none |
| 3 | 5 | 10 | 20 | L∈[1..5] | gaussian | capacity jitter |
| 4 | 7 | 14 | 28 | L∈[1..7] | seasonal | severe strikes |

## milp formulation

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

| vector | defense |
|---|---|
| negative / zero / float qty | pydantic `strict=True, ge=1` + pre-mutation `_is_underflow_qty` check → reward −1.0, done |
| cyclic reward farming | `MAX_CALLS_PER_CYCLE=5` + `DENSE_CAP=0.4`; holding cost > dense reward |
| phantom edge hallucination | immutable `_adjacency: frozenset` built at reset → dispatch off-graph ends episode |

## evaluation

```
make eval N=20 TIERS="1 2 3"
make viz
```

produces:
- `eval.json` with per-rollout cost, gap, terminal
- `assets/gap_hist.png` per-tier gap histograms
- `assets/terminal_bars.png` mean terminal reward bars by policy × tier

baseline numbers on 5 seeds (tier 1 / tier 2):

| policy | tier 1 gap / terminal | tier 2 gap / terminal |
|---|---|---|
| zero_op | 4.48 / 0.19 | 1.01 / 0.51 |
| greedy | 1.59 / 0.39 | 0.35 / 0.75 |
| optimal_replay | **0.07 / 0.94** | **0.02 / 0.98** |

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

## judging criteria alignment

| criterion | weight | evidence |
|---|---|---|
| environment innovation | 40% | time-expanded graph with milp oracle; 3 hard-gated exploitation vectors; 4-tier procedural curriculum with ema gating |
| storytelling | 30% | `BLOG.md` + `VIDEO.md` with explicit before/after arc and numbers |
| showing improvement | 20% | `make eval` + `make viz` produces gap histograms; trackio live dashboard hook; near-optimal replay baseline provides ground-truth upper bound |
| reward and training pipeline | 10% | 43 green tests; composite rubric + anti-hack gates; `train.py` with 3 reward functions + trackio + strict grpo config |

## credits

meta pytorch openenv team. huggingface trl team. unsloth team. coin-or cbc. apache-2.0.
