.PHONY: help install install-train serve eval viz test bench clean docker push-space train demo

PY ?= python3
PORT ?= 7860
N ?= 10
TIERS ?= 1 2

help:
	@echo "targets:"
	@echo "  install        env-only deps (mac/linux, no torch)"
	@echo "  install-train  full training deps (cuda linux)"
	@echo "  serve          uvicorn on :$(PORT)"
	@echo "  test           pytest"
	@echo "  eval           run baselines, write eval.json"
	@echo "  viz            generate plots into assets/"
	@echo "  bench          tier 1..3 greedy vs milp timing"
	@echo "  docker         build image"
	@echo "  push-space     openenv push to hf space (REPO=user/env-name)"
	@echo "  train          local grpo train (requires cuda)"
	@echo "  demo           full end-to-end eval + plots"
	@echo "  clean          remove caches, venvs, eval.json, assets"

install:
	$(PY) -m pip install -r requirements.txt

install-train:
	$(PY) -m pip install -r requirements-train.txt

serve:
	$(PY) -m uvicorn server.app:app --host 0.0.0.0 --port $(PORT) --reload

test:
	$(PY) -m pytest tests/ -q

eval:
	$(PY) eval.py --tiers $(TIERS) --n $(N) --out eval.json

viz:
	mkdir -p assets
	$(PY) viz.py gap --eval eval.json --out assets/gap_hist.png
	$(PY) viz.py terminal --eval eval.json --out assets/terminal_bars.png

bench:
	$(PY) -c "from server.dsc_environment import _build_tier; from server.solver import optimal_cost, greedy_cost; import random; \
	[print(f'tier {t}: greedy={greedy_cost(_build_tier(t, random.Random(7))):.1f}  optimal={optimal_cost(_build_tier(t, random.Random(7)), time_limit=20):.1f}') for t in [1,2,3]]"

docker:
	docker build -t openenv-dsc-co .

push-space:
	@[ -n "$(REPO)" ] || (echo "set REPO=user/env-name"; exit 1)
	openenv push -r $(REPO) --exclude .openenvignore

train:
	$(PY) train.py

demo: eval viz
	@echo ""
	@echo "plots written to assets/"
	@ls -la assets/

clean:
	rm -rf .venv_check env env_check env_test env_fresh __pycache__ server/__pycache__ tests/__pycache__ .pytest_cache eval.json *.egg-info
	find . -name "*.pyc" -delete
