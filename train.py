from __future__ import annotations

import json
import os
import random
from typing import Any, List, Optional


MODEL_NAME = os.environ.get("DSC_MODEL", "unsloth/Llama-3.2-3B-Instruct-bnb-4bit")
MAX_SEQ = int(os.environ.get("DSC_MAX_SEQ", "4096"))
LORA_R = int(os.environ.get("DSC_LORA_R", "32"))
DIFFICULTY = int(os.environ.get("DSC_TIER", "1"))
NUM_GEN = int(os.environ.get("DSC_NUM_GEN", "8"))
DATASET_SIZE = int(os.environ.get("DSC_DATA_N", "2000"))
MAX_STEPS = int(os.environ.get("DSC_MAX_STEPS", "0"))
ENV_URL = os.environ.get("DSC_ENV_URL", "")
TRACKIO_PROJECT = os.environ.get("DSC_TRACKIO", "openenv-dsc-co")
TRACKIO_SPACE = os.environ.get("DSC_TRACKIO_SPACE", "")
REPORT_TO = os.environ.get("DSC_REPORT", "none")


SYSTEM_PROMPT = (
    "you are a central supply chain planner over a 30 step horizon. "
    "you must minimize total operational cost: transit + holding + shortage penalty. "
    "available mcp tools:\n"
    "  query_network(source_id, dest_id) -> edge info\n"
    "  dispatch_inventory(routes=[{src,dst,qty}]) -> ship orders, qty must be positive integer\n"
    "  advance_cycle() -> tick time, process arrivals, deduct demand\n"
    "rules: only dispatch over edges returned by query_network. qty must be strict int>=1. "
    "plan ahead: lead times delay arrivals. stage inventory at warehouses before retail demand spikes. "
    "advance_cycle ticks time; episode ends at step 30."
)


def _obs_to_str(obs: Any) -> str:
    if hasattr(obs, "model_dump"):
        return json.dumps(obs.model_dump(), separators=(",", ":"))
    if isinstance(obs, dict):
        return json.dumps(obs, separators=(",", ":"))
    return str(obs)


class DSCToolEnv:
    def __init__(self):
        from server.dsc_environment import DSCEnv

        self._env = DSCEnv()
        self.reward = 0.0
        self.cumulative = 0.0
        self.terminal = 0.0
        self.done = False
        self._seed = random.randint(0, 2**31 - 1)

    def reset(self, **kwargs) -> Optional[str]:
        self.reward = 0.0
        self.cumulative = 0.0
        self.terminal = 0.0
        self.done = False
        self._seed = random.randint(0, 2**31 - 1)
        obs = self._env.reset(seed=self._seed, difficulty=DIFFICULTY)
        return _obs_to_str(obs)

    def _run(self, payload: dict) -> str:
        if self.done:
            return json.dumps({"done": True, "reason": "already_done"})
        obs = self._env.step({"action": payload})
        self.reward = float(obs.reward)
        self.cumulative += self.reward
        self.done = bool(obs.done)
        if obs.done:
            meta = obs.metadata or {}
            if "terminal" in meta:
                self.terminal = float(meta.get("terminal", 0.0))
                self.cumulative += self.terminal
        return _obs_to_str(obs)

    def query_network(self, source_id: str, dest_id: str) -> str:
        return self._run({"kind": "query_network", "source_id": source_id, "dest_id": dest_id})

    def dispatch_inventory(self, routes: List[dict]) -> str:
        return self._run({"kind": "dispatch_inventory", "routes": routes})

    def advance_cycle(self) -> str:
        return self._run({"kind": "advance_cycle"})


def reward_func(prompts, completions, environments=None, **kwargs) -> List[float]:
    out = []
    if environments is None:
        return [0.0] * len(prompts)
    for env in environments:
        out.append(float(getattr(env, "cumulative", 0.0)))
    if out:
        _log_trackio({"reward/mean_cumulative": sum(out) / len(out), "reward/max": max(out)})
    return out


def schema_reward_func(prompts, completions, environments=None, **kwargs) -> List[float]:
    out = []
    if environments is None:
        return [0.0] * len(prompts)
    for env in environments:
        out.append(float(getattr(env, "reward", 0.0)))
    if out:
        _log_trackio({"reward/mean_step": sum(out) / len(out)})
    return out


def terminal_reward_func(prompts, completions, environments=None, **kwargs) -> List[float]:
    out = []
    if environments is None:
        return [0.0] * len(prompts)
    for env in environments:
        out.append(float(getattr(env, "terminal", 0.0)))
    if out:
        _log_trackio({"reward/mean_terminal": sum(out) / len(out)})
    return out


def _build_dataset():
    from datasets import Dataset

    prompts = [
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "plan the supply chain. call tools to minimize cost."},
        ]
    ] * DATASET_SIZE
    return Dataset.from_dict({"prompt": prompts})


_TRACKIO = None


def _init_trackio():
    global _TRACKIO
    if _TRACKIO is not None:
        return _TRACKIO
    try:
        import trackio

        kwargs = {"project": TRACKIO_PROJECT}
        if TRACKIO_SPACE:
            kwargs["space_id"] = TRACKIO_SPACE
        _TRACKIO = trackio.init(**kwargs)
        print("init trackio")
        return _TRACKIO
    except Exception:
        print("no trackio")
        _TRACKIO = False
        return None


def _log_trackio(metrics: dict) -> None:
    t = _init_trackio()
    if t is None or t is False:
        return
    try:
        import trackio

        trackio.log(metrics)
    except Exception:
        pass


def main() -> None:
    print("init train")
    _init_trackio()

    try:
        from unsloth import FastLanguageModel, PatchFastRL
    except Exception as e:
        print("no unsloth")
        raise

    PatchFastRL("GRPO", FastLanguageModel)

    from trl import GRPOConfig, GRPOTrainer

    import torch as _torch
    _use_fast = _torch.cuda.get_device_capability()[0] >= 8
    print(f"load model fast_inference={_use_fast}")

    _load_kwargs = dict(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ,
        load_in_4bit=True,
        fast_inference=_use_fast,
    )
    if _use_fast:
        _load_kwargs["gpu_memory_utilization"] = 0.55
        _load_kwargs["max_lora_rank"] = LORA_R

    model, tokenizer = FastLanguageModel.from_pretrained(**_load_kwargs)

    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=LORA_R,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    dataset = _build_dataset()

    config = GRPOConfig(
        output_dir="./grpo_dsc_co",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=5e-6,
        num_generations=NUM_GEN,
        max_completion_length=2048,
        beta=0.04,
        temperature=0.7,
        use_vllm=_use_fast,
        vllm_mode="colocate" if _use_fast else None,
        chat_template_kwargs={"enable_thinking": False},
        log_completions=True,
        logging_steps=1,
        save_steps=50,
        num_train_epochs=1,
        max_steps=MAX_STEPS if MAX_STEPS > 0 else -1,
        report_to=REPORT_TO,
    )

    print("init grpo")
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        args=config,
        train_dataset=dataset,
        reward_funcs=[reward_func, schema_reward_func, terminal_reward_func],
        environment_factory=DSCToolEnv,
    )

    print("run train")
    trainer.train()
    print("save lora")
    trainer.save_model("./grpo_dsc_co/final")


if __name__ == "__main__":
    main()
