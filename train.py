from __future__ import annotations

import json
import os
import importlib.util
import inspect
import time
import ast
import torch
if not hasattr(torch, "int1"):
    torch.int1 = torch.int8
# torchao in some transformers builds references low-bit dtypes that
# are not present on torch 2.5.x wheels used in HF Spaces base images.
if not hasattr(torch, "int2"):
    torch.int2 = torch.int8
if not hasattr(torch, "int3"):
    torch.int3 = torch.int8
if not hasattr(torch, "int4"):
    torch.int4 = torch.int8
import random
from typing import Any, List, Optional


MODEL_NAME = os.environ.get("DSC_MODEL", "unsloth/Llama-3.2-3B-Instruct-bnb-4bit")
MAX_SEQ = int(os.environ.get("DSC_MAX_SEQ", "4096"))
LORA_R = int(os.environ.get("DSC_LORA_R", "32"))
DIFFICULTY = int(os.environ.get("DSC_TIER", "1"))
NUM_GEN = int(os.environ.get("DSC_NUM_GEN", "4"))
DATASET_SIZE = int(os.environ.get("DSC_DATA_N", "2000"))
MAX_STEPS = int(os.environ.get("DSC_MAX_STEPS", "0"))
MAX_COMPLETION_LENGTH = int(os.environ.get("DSC_MAX_COMPLETION", "512"))
SAVE_STEPS = int(os.environ.get("DSC_SAVE_STEPS", "200"))
TRAIN_BATCH_SIZE = int(os.environ.get("DSC_BATCH_SIZE", "1"))
GRAD_ACCUM = int(os.environ.get("DSC_GRAD_ACCUM", "8"))
LEARNING_RATE = float(os.environ.get("DSC_LR", "5e-6"))
BETA = float(os.environ.get("DSC_BETA", "0.04"))
TEMPERATURE = float(os.environ.get("DSC_TEMP", "0.7"))
NUM_TRAIN_EPOCHS = float(os.environ.get("DSC_EPOCHS", "1"))
ENV_URL = os.environ.get("DSC_ENV_URL", "")
TRACKIO_PROJECT = os.environ.get("DSC_TRACKIO", "openenv-dsc-co")
TRACKIO_SPACE = os.environ.get("DSC_TRACKIO_SPACE", "")
REPORT_TO = os.environ.get("DSC_REPORT", "none")
OUT_DIR = os.environ.get("DSC_OUT_DIR", "./grpo_dsc_co")
HF_REPO = os.environ.get("DSC_HF_REPO", "")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
RESUME_FROM_CHECKPOINT = os.environ.get("DSC_RESUME", "").strip()
LOG_COMPLETIONS = os.environ.get("DSC_LOG_COMPLETIONS", "0").lower() in {
    "1",
    "true",
    "yes",
}
DEBUG_ENABLED = os.environ.get("DSC_DEBUG", "0").lower() in {"1", "true", "yes"}
DEBUG_LOG_PATH = os.environ.get("DSC_DEBUG_LOG_PATH", "/tmp/dsc_train_debug.ndjson")
DEBUG_LOG_FALLBACK_PATH = "/tmp/dsc_train_debug_fallback.ndjson"
DEBUG_SESSION_ID = os.environ.get("DSC_DEBUG_SESSION_ID", "dsc-train")
DEBUG_RUN_ID = os.environ.get("DSC_DEBUG_RUN_ID", "baseline")


def _dbg_log(hypothesis_id: str, location: str, message: str, data: dict) -> None:
    if not DEBUG_ENABLED:
        return
    payload = {
        "sessionId": DEBUG_SESSION_ID,
        "runId": DEBUG_RUN_ID,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    try:
        log_dir = os.path.dirname(DEBUG_LOG_PATH)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, separators=(",", ":")) + "\n")
    except Exception:
        try:
            with open(DEBUG_LOG_FALLBACK_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, separators=(",", ":")) + "\n")
        except Exception:
            pass
    try:
        print(f"[DBG {hypothesis_id}] {location} :: {message}")
    except Exception:
        pass


def _completion_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict):
                parts.append(str(item.get("content", item)))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    if isinstance(value, dict):
        return str(value.get("content", value))
    return str(value)


_REPLAY_CACHE: dict[str, dict] = {}


def _json_objects_from_text(text: str) -> List[dict]:
    decoder = json.JSONDecoder()
    out: List[dict] = []
    idx = 0
    while idx < len(text):
        start = text.find("{", idx)
        if start < 0:
            break
        try:
            obj, end = decoder.raw_decode(text[start:])
            idx = start + max(end, 1)
        except Exception:
            idx = start + 1
            continue
        if isinstance(obj, dict):
            out.append(obj)
    return out


def _python_dicts_from_text(text: str) -> List[dict]:
    out: List[dict] = []
    for raw in text.splitlines():
        line = raw.strip().rstrip(",")
        if not line.startswith("{") or not line.endswith("}"):
            continue
        try:
            obj = ast.literal_eval(line)
        except Exception:
            continue
        if isinstance(obj, dict):
            out.append(obj)
    return out


def _normalize_action(obj: dict) -> Optional[dict]:
    kind = obj.get("kind")
    if kind == "query_network":
        source_id = obj.get("source_id") or obj.get("src")
        dest_id = obj.get("dest_id") or obj.get("dst")
        if isinstance(source_id, str) and isinstance(dest_id, str):
            return {"kind": "query_network", "source_id": source_id, "dest_id": dest_id}
    if kind == "dispatch_inventory":
        routes = obj.get("routes")
        if isinstance(routes, list):
            return {"kind": "dispatch_inventory", "routes": routes}
    if kind == "advance_cycle":
        return {"kind": "advance_cycle"}
    return None


def _extract_actions(text: str) -> List[dict]:
    candidates = _json_objects_from_text(text)
    if not candidates:
        candidates = _python_dicts_from_text(text)
    actions: List[dict] = []
    for obj in candidates:
        action = _normalize_action(obj)
        if action is not None:
            actions.append(action)
    return actions[:40]


SYSTEM_PROMPT = (
    "you are a central supply chain planner over a 30 step horizon. "
    "output only executable tool actions as json lines. do not write markdown, python, prose, or explanations. "
    "valid tier-1 node ids are S0 supplier, W0 warehouse, R0 retail. "
    "available actions:\n"
    "{\"kind\":\"query_network\",\"source_id\":\"S0\",\"dest_id\":\"W0\"}\n"
    "{\"kind\":\"dispatch_inventory\",\"routes\":[{\"src\":\"S0\",\"dst\":\"W0\",\"qty\":20}]}\n"
    "{\"kind\":\"advance_cycle\"}\n"
    "{\"kind\":\"query_network\",\"source_id\":\"W0\",\"dest_id\":\"R0\"}\n"
    "{\"kind\":\"dispatch_inventory\",\"routes\":[{\"src\":\"W0\",\"dst\":\"R0\",\"qty\":5}]}\n"
    "rules: only dispatch over edges returned by query_network. qty must be strict int >= 1. "
    "advance_cycle ticks time; episode ends only after 30 advance cycles. "
    "good plan: query S0->W0, ship small stock to W0, advance, query W0->R0, ship small stock to R0, "
    "then continue valid dispatches and advance_cycle until the 30-step episode finishes."
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
        obs_text = _obs_to_str(obs)
        # #region agent log
        _dbg_log(
            "H1,H2",
            "train.py:reset",
            "environment reset",
            {
                "difficulty": DIFFICULTY,
                "seed": self._seed,
                "obs_len": len(obs_text),
                "obs_preview": obs_text[:500],
            },
        )
        # #endregion
        return obs_text

    def _run(self, payload: dict) -> str:
        if self.done:
            return json.dumps({"done": True, "reason": "already_done"})
        before_calls = getattr(self._env, "_calls_this_cycle", None)
        before_step = getattr(self._env, "_current_step", None)
        obs = self._env.step({"action": payload})
        self.reward = float(obs.reward)
        self.cumulative += self.reward
        self.done = bool(obs.done)
        if obs.done:
            meta = obs.metadata or {}
            if "terminal" in meta:
                self.terminal = float(meta.get("terminal", 0.0))
                self.cumulative += self.terminal
        # #region agent log
        _dbg_log(
            "H2,H4",
            "train.py:_run",
            "tool step executed",
            {
                "action_kind": payload.get("kind"),
                "payload": payload,
                "before_step": before_step,
                "after_step": getattr(self._env, "_current_step", None),
                "before_calls_this_cycle": before_calls,
                "after_calls_this_cycle": getattr(self._env, "_calls_this_cycle", None),
                "reward": self.reward,
                "cumulative": self.cumulative,
                "terminal": self.terminal,
                "done": self.done,
                "obs_reward": float(getattr(obs, "reward", 0.0)),
                "obs_done": bool(getattr(obs, "done", False)),
                "obs_metadata": getattr(obs, "metadata", {}),
            },
        )
        # #endregion
        return _obs_to_str(obs)

    def query_network(self, source_id: str, dest_id: str) -> str:
        return self._run({"kind": "query_network", "source_id": source_id, "dest_id": dest_id})

    def dispatch_inventory(self, routes: List[dict]) -> str:
        return self._run({"kind": "dispatch_inventory", "routes": routes})

    def advance_cycle(self) -> str:
        return self._run({"kind": "advance_cycle"})


def _score_completion_without_trl_env(completion: Any, sample_idx: int) -> dict:
    text = _completion_text(completion)
    cache_key = text[:8192]
    if cache_key in _REPLAY_CACHE:
        return _REPLAY_CACHE[cache_key]

    actions = _extract_actions(text)
    env = DSCToolEnv()
    env.reset()
    executed = 0
    last_obs = ""
    for action in actions:
        if env.done:
            break
        last_obs = env._run(action)
        executed += 1

    score = {
        "cumulative": float(env.cumulative),
        "step_reward": float(env.reward),
        "terminal": float(env.terminal),
        "done": bool(env.done),
        "executed": executed,
        "parsed": len(actions),
        "completion_len": len(text),
        "completion_preview": text[:700],
        "last_obs_preview": last_obs[:500],
    }
    # #region agent log
    _dbg_log(
        "H2,H6",
        "train.py:_score_completion_without_trl_env",
        "local replay scored completion",
        {"sample_idx": sample_idx, **score},
    )
    # #endregion
    _REPLAY_CACHE[cache_key] = score
    return score


def reward_func(prompts, completions, environments=None, **kwargs) -> List[float]:
    out = []
    if environments is None:
        # #region agent log
        _dbg_log(
            "H2",
            "train.py:reward_func",
            "reward called without environments",
            {"num_prompts": len(prompts), "num_completions": len(completions)},
        )
        # #endregion
        out = [
            float(_score_completion_without_trl_env(c, idx)["cumulative"])
            for idx, c in enumerate(completions)
        ]
        if out:
            _log_trackio({"reward/mean_cumulative": sum(out) / len(out), "reward/max": max(out)})
        return out
    for idx, env in enumerate(environments):
        out.append(float(getattr(env, "cumulative", 0.0)))
        c = completions[idx] if idx < len(completions) else None
        c_full = _completion_text(c) if c is not None else ""
        c_text = c_full[:700]
        # #region agent log
        _dbg_log(
            "H1,H2,H3",
            "train.py:reward_func",
            "reward snapshot from env cumulative",
            {
                "sample_idx": idx,
                "num_prompts": len(prompts),
                "num_completions": len(completions),
                "num_environments": len(environments),
                "completion_type": type(c).__name__ if c is not None else "None",
                "completion_len": len(c_full),
                "env_cumulative": float(getattr(env, "cumulative", 0.0)),
                "env_reward": float(getattr(env, "reward", 0.0)),
                "env_terminal": float(getattr(env, "terminal", 0.0)),
                "env_done": bool(getattr(env, "done", False)),
                "env_step": getattr(getattr(env, "_env", None), "_current_step", None),
                "env_calls_this_cycle": getattr(getattr(env, "_env", None), "_calls_this_cycle", None),
                "completion_preview": c_text,
                "looks_like_tool_call": ("dispatch_inventory" in c_text)
                or ("query_network" in c_text)
                or ("advance_cycle" in c_text),
            },
        )
        # #endregion
    if out:
        _log_trackio({"reward/mean_cumulative": sum(out) / len(out), "reward/max": max(out)})
    return out


def schema_reward_func(prompts, completions, environments=None, **kwargs) -> List[float]:
    out = []
    if environments is None:
        # #region agent log
        _dbg_log(
            "H2",
            "train.py:schema_reward_func",
            "schema reward called without environments",
            {"num_prompts": len(prompts), "num_completions": len(completions)},
        )
        # #endregion
        out = [
            float(_score_completion_without_trl_env(c, idx)["step_reward"])
            for idx, c in enumerate(completions)
        ]
        if out:
            _log_trackio({"reward/mean_step": sum(out) / len(out)})
        return out
    for idx, env in enumerate(environments):
        out.append(float(getattr(env, "reward", 0.0)))
        # #region agent log
        _dbg_log(
            "H3",
            "train.py:schema_reward_func",
            "schema reward snapshot",
            {
                "sample_idx": idx,
                "step_reward": float(getattr(env, "reward", 0.0)),
                "env_cumulative": float(getattr(env, "cumulative", 0.0)),
                "done": bool(getattr(env, "done", False)),
                "env_step": getattr(getattr(env, "_env", None), "_current_step", None),
            },
        )
        # #endregion
    if out:
        _log_trackio({"reward/mean_step": sum(out) / len(out)})
    return out


def terminal_reward_func(prompts, completions, environments=None, **kwargs) -> List[float]:
    out = []
    if environments is None:
        # #region agent log
        _dbg_log(
            "H2",
            "train.py:terminal_reward_func",
            "terminal reward called without environments",
            {"num_prompts": len(prompts), "num_completions": len(completions)},
        )
        # #endregion
        out = [
            float(_score_completion_without_trl_env(c, idx)["terminal"])
            for idx, c in enumerate(completions)
        ]
        if out:
            _log_trackio({"reward/mean_terminal": sum(out) / len(out)})
        return out
    for idx, env in enumerate(environments):
        out.append(float(getattr(env, "terminal", 0.0)))
        # #region agent log
        _dbg_log(
            "H4",
            "train.py:terminal_reward_func",
            "terminal reward snapshot",
            {
                "sample_idx": idx,
                "terminal_reward": float(getattr(env, "terminal", 0.0)),
                "env_cumulative": float(getattr(env, "cumulative", 0.0)),
                "done": bool(getattr(env, "done", False)),
                "env_step": getattr(getattr(env, "_env", None), "_current_step", None),
            },
        )
        # #endregion
    if out:
        _log_trackio({"reward/mean_terminal": sum(out) / len(out)})
    return out


def _build_dataset():
    from datasets import Dataset

    prompts = [
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "return 35-45 json lines only. start with query_network S0 W0, then dispatch S0 W0, "
                    "then advance_cycle, then query W0 R0 and dispatch W0 R0. include enough advance_cycle "
                    "actions to finish the 30-step episode. no prose."
                ),
            },
        ]
    ] * DATASET_SIZE
    # #region agent log
    _dbg_log(
        "H1,H3",
        "train.py:_build_dataset",
        "dataset built",
        {
            "dataset_size": DATASET_SIZE,
            "max_completion_length": MAX_COMPLETION_LENGTH,
            "num_generations": NUM_GEN,
            "prompt_preview": prompts[0],
        },
    )
    # #endregion
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


def _push_to_hub(folder_path: str) -> None:
    if not HF_REPO:
        print("DSC_HF_REPO not set; skipping hub upload.")
        return
    if not HF_TOKEN:
        print("HF_TOKEN not set; skipping hub upload.")
        return

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(HF_REPO, token=HF_TOKEN, repo_type="model", exist_ok=True)
    api.upload_folder(
        folder_path=folder_path,
        repo_id=HF_REPO,
        repo_type="model",
        token=HF_TOKEN,
    )
    print(f"uploaded adapter to https://huggingface.co/{HF_REPO}")


def _resolve_resume_checkpoint() -> Optional[str]:
    if not RESUME_FROM_CHECKPOINT or RESUME_FROM_CHECKPOINT.lower() in {"0", "false", "no", "off"}:
        return None
    if RESUME_FROM_CHECKPOINT.lower() not in {"1", "true", "yes", "latest"}:
        return RESUME_FROM_CHECKPOINT

    try:
        from transformers.trainer_utils import get_last_checkpoint

        checkpoint = get_last_checkpoint(OUT_DIR)
    except Exception:
        checkpoint = None
    if checkpoint:
        print(f"resuming from checkpoint: {checkpoint}")
        return checkpoint
    print("DSC_RESUME requested, but no checkpoint was found; starting fresh.")
    return None


def _write_training_artifacts(trainer: Any, out_dir: str, train_output: Any = None) -> None:
    history = list(getattr(getattr(trainer, "state", None), "log_history", []) or [])
    train_metrics = getattr(train_output, "metrics", None) or {}
    if train_metrics:
        history.append({"step": getattr(getattr(trainer, "state", None), "global_step", len(history)), **train_metrics})
    if not history and not train_metrics:
        print("no trainer log history found; writing empty training artifact manifest.")

    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "training_metrics.json")
    csv_path = os.path.join(out_dir, "training_metrics.csv")
    png_path = os.path.join(out_dir, "training_curve.png")
    summary_path = os.path.join(out_dir, "training_summary.json")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, default=str)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_name": MODEL_NAME,
                "max_steps": MAX_STEPS,
                "dataset_size": DATASET_SIZE,
                "num_generations": NUM_GEN,
                "max_completion_length": MAX_COMPLETION_LENGTH,
                "save_steps": SAVE_STEPS,
                "trackio_project": TRACKIO_PROJECT,
                "trackio_space": TRACKIO_SPACE,
                "train_metrics": train_metrics,
                "num_log_history_rows": len(history),
            },
            f,
            indent=2,
            default=str,
        )

    fieldnames = sorted({key for row in history for key in row.keys()})
    preferred = [k for k in ["step", "epoch", "loss", "reward", "grad_norm", "learning_rate"] if k in fieldnames]
    fieldnames = preferred + [k for k in fieldnames if k not in preferred]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        import csv

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
        for row in history:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        series = {}
        for row_idx, row in enumerate(history):
            step = row.get("step", row_idx)
            for key, value in row.items():
                if key == "step" or not isinstance(value, (int, float)):
                    continue
                lowered = key.lower()
                if "reward" in lowered or lowered in {"loss", "grad_norm"}:
                    series.setdefault(key, []).append((step, float(value)))

        if series:
            fig, ax = plt.subplots(figsize=(9, 4.8))
            for key, points in sorted(series.items()):
                xs = [p[0] for p in points]
                ys = [p[1] for p in points]
                ax.plot(xs, ys, linewidth=1.3, label=key)
            ax.set_xlabel("training step")
            ax.set_ylabel("metric value")
            ax.set_title("GRPO training metrics")
            ax.grid(True, alpha=0.3)
            ax.legend()
            fig.tight_layout()
            fig.savefig(png_path, dpi=140)
            plt.close(fig)
        else:
            print("no numeric reward/loss series found for training curve.")
    except Exception as e:
        print(f"could not render training curve: {e}")

    print(f"wrote training artifacts to {out_dir}: {json_path}, {csv_path}, {summary_path}")


def _sync_trackio_space() -> None:
    if not TRACKIO_SPACE:
        return
    try:
        import trackio

        trackio.sync(project=TRACKIO_PROJECT, space_id=TRACKIO_SPACE, sdk="gradio")
        print(f"refreshed trackio dashboard at https://huggingface.co/spaces/{TRACKIO_SPACE}")
    except Exception as e:
        print(f"could not sync trackio dashboard: {e}")


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
    _has_vllm = importlib.util.find_spec("vllm") is not None
    _gpu_supports_fast = _torch.cuda.get_device_capability()[0] >= 8
    _use_fast = _gpu_supports_fast and _has_vllm
    if _gpu_supports_fast and not _has_vllm:
        print("vllm not installed; disabling fast_inference and using standard path.")
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

    try:
        model, tokenizer = FastLanguageModel.from_pretrained(**_load_kwargs)
    except ImportError as e:
        # Newer transformers may tighten bitsandbytes requirements.
        # Retry without 4-bit quantization so the run can proceed.
        if "bitsandbytes" in str(e) and "4-bit quantization" in str(e):
            print("4-bit quantization unavailable in this runtime; retrying with full precision.")
            _load_kwargs["load_in_4bit"] = False
            model, tokenizer = FastLanguageModel.from_pretrained(**_load_kwargs)
        else:
            raise

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

    _config_kwargs = dict(
        output_dir=OUT_DIR,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        num_generations=NUM_GEN,
        max_completion_length=MAX_COMPLETION_LENGTH,
        beta=BETA,
        temperature=TEMPERATURE,
        use_vllm=_use_fast,
        vllm_mode="colocate" if _use_fast else None,
        chat_template_kwargs={"enable_thinking": False},
        log_completions=LOG_COMPLETIONS,
        logging_steps=1,
        save_steps=SAVE_STEPS,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        max_steps=MAX_STEPS if MAX_STEPS > 0 else -1,
        report_to=REPORT_TO,
    )
    _supported = set(inspect.signature(GRPOConfig.__init__).parameters.keys())
    _config_kwargs = {k: v for k, v in _config_kwargs.items() if k in _supported}
    # #region agent log
    _dbg_log(
        "H2,H3",
        "train.py:main",
        "grpo config resolved",
        {
            "model_name": MODEL_NAME,
            "max_seq": MAX_SEQ,
            "max_steps": MAX_STEPS,
            "data_size": DATASET_SIZE,
            "use_fast": _use_fast,
            "has_vllm": _has_vllm,
            "debug_enabled": DEBUG_ENABLED,
            "resume_from_checkpoint": RESUME_FROM_CHECKPOINT,
            "supported_config": _config_kwargs,
        },
    )
    # #endregion
    config = GRPOConfig(**_config_kwargs)

    print("init grpo")
    # Some TRL builds expect this mutable warnings registry on the model.
    # PEFT wrappers may not expose it, so initialize it explicitly.
    if not hasattr(model, "warnings_issued"):
        model.warnings_issued = {}
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        args=config,
        train_dataset=dataset,
        reward_funcs=[reward_func, schema_reward_func, terminal_reward_func],
        environment_factory=DSCToolEnv,
    )

    print("run train")
    resume_checkpoint = _resolve_resume_checkpoint()
    if resume_checkpoint is None:
        train_output = trainer.train()
    else:
        train_output = trainer.train(resume_from_checkpoint=resume_checkpoint)
    print("save lora")
    final_dir = os.path.join(OUT_DIR, "final")
    trainer.save_model(final_dir)
    _write_training_artifacts(trainer, final_dir, train_output)
    _sync_trackio_space()
    _push_to_hub(final_dir)


if __name__ == "__main__":
    main()
