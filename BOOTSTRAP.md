# bootstrap

full setup path for local environment work, openenv deployment, and the final grpo training workflow.

## modes

| mode | purpose | command path |
| ---- | ------- | ------------ |
| local env | tests, server, cli, eval, plots | `requirements.txt`, `make test`, `make serve`, `make eval` |
| openenv space | public runnable environment | `openenv push -r AceofStades/dsc_co --exclude .openenvignore` |
| training space | full grpo run on gpu | root `Dockerfile` on a huggingface **A100 docker space** |
| notebooks | reference / experiments | `notebooks/train_hf_space.ipynb`, `notebooks/train_kaggle.ipynb` |

the final submitted training run was done on huggingface spaces with A100 hardware using the root `Dockerfile`. the notebooks are provided, but they are not the canonical final training route.

## local install

```
git clone https://github.com/CYCLOP5/metascaler-hack.git
cd metascaler-hack

python3 -m venv env
source env/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` is the environment/runtime install: fastapi, openenv-core, pydantic, pulp, gymnasium, matplotlib, requests, and related deps. it does not install the full cuda training stack.

## tests

```
make test
```

coverage includes action parsing, observation shape, invalid-action gates, valid dispatches, horizon termination, solver correctness, and tier topology checks.

## local server

```
make serve
```

from a second shell:

```
python client.py reset --tier 1 --seed 7
python client.py query S0 W0
python client.py dispatch S0 W0 50
python client.py advance
python client.py tools
```

public hosted env:

```
https://huggingface.co/spaces/AceofStades/dsc_co
```

## baselines and plots

```
make eval N=10 TIERS="1 2"
make viz
```

outputs:

- `eval.json`
- `assets/gap_hist.png`
- `assets/terminal_bars.png`

final committed training plot:

- `assets/training_curve.png`

## openenv space deploy

this deploys the runnable environment server, not the gpu trainer:

```
huggingface-cli login
openenv push -r AceofStades/dsc_co --exclude .openenvignore
```

`.openenvignore` excludes training-only files from the public env space: root `Dockerfile`, `app.py`, `train.py`, `requirements-train.txt`, notebooks, trackio app, local envs, and large local outputs.

the openenv deploy path uses:

- `server/app.py`
- `server/Dockerfile`
- `openenv.yaml`

## training space deploy

intended full training path:

| field | value |
| ----- | ----- |
| platform | huggingface spaces |
| sdk | docker |
| hardware | A100 recommended for the final/full run |
| build file | root `Dockerfile` |
| entrypoint | root `app.py`, which starts `train.py` in a background thread |
| logs | huggingface space logs |

create a new huggingface space with docker sdk and A100 hardware, then configure:

```
HF_TOKEN=<write-scope huggingface token>
DSC_HF_REPO=AceofStades/dsc-co-grpo-lora
DSC_TRACKIO=openenv-dsc-co
DSC_TRACKIO_SPACE=AceofStades/dsc-co-trackio
```

push the repo to the training space:

```
git remote add training-space https://huggingface.co/spaces/<user>/<training-space>
git push training-space main
```

on startup, the root `Dockerfile` builds the cuda/unsloth/training image. `app.py` exposes a small port `7860` health server and launches `python train.py`. watch the space logs for unsloth/trl output.

## final run preset

```
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

for future qualitative before/after demos, set `DSC_LOG_COMPLETIONS=1` so the run preserves exact trained JSON action traces. the final submitted run used `DSC_LOG_COMPLETIONS=0`, so its improvement evidence is the metrics CSV/JSON and reward curves.

training stack:

- `unsloth/Llama-3.2-3B-Instruct-bnb-4bit`
- trl `GRPOTrainer`
- unsloth qlora
- pulp/cbc terminal verifier
- trackio live metric logging

## artifacts

trained adapter:

```
https://huggingface.co/AceofStades/dsc-co-grpo-lora
```

exported by `train.py` at the end of training:

- `training_metrics.csv`
- `training_metrics.json`
- `training_summary.json`
- `training_curve.png`

committed copies:

- `results/training_metrics.csv`
- `results/training_metrics.json`
- `results/training_summary.json`
- `assets/training_curve.png`

## notebooks

provided notebooks:

- `notebooks/train_hf_space.ipynb`
- `notebooks/train_kaggle.ipynb`

these are for reproducibility and experimentation. the final evidence run used the huggingface docker space path with the root `Dockerfile` on A100 hardware, because that path matches the submitted build, dependency stack, logging path, and artifact upload behavior.

## trackio space

dashboard:

```
https://huggingface.co/spaces/AceofStades/dsc-co-trackio
```

source:

```
trackio_space/
```

this is separate from both the public env space and the gpu training space.

## local docker smoke test

```
docker build -t openenv-dsc-co .
docker run --rm -p 7860:7860 openenv-dsc-co
```

on a cpu-only laptop this is only a build/config smoke test. full grpo training expects cuda gpu hardware, and the intended route is the huggingface A100 docker space.
