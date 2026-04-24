from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load(path: str) -> dict:
    return json.loads(Path(path).read_text())


def gap_histogram(eval_json: str, out_path: str) -> None:
    print("viz gap")
    d = _load(eval_json)
    rollouts = d["rollouts"]
    policies = sorted({r["policy"] for r in rollouts})
    tiers = sorted({r["tier"] for r in rollouts})

    fig, axes = plt.subplots(1, len(tiers), figsize=(6 * len(tiers), 4), squeeze=False)
    for col, tier in enumerate(tiers):
        ax = axes[0][col]
        for policy in policies:
            gaps = [r["gap"] for r in rollouts if r["tier"] == tier and r["policy"] == policy and r["gap"] != float("inf")]
            if not gaps:
                continue
            ax.hist(gaps, bins=20, alpha=0.5, label=policy)
        ax.set_title(f"tier {tier} optimality gap")
        ax.set_xlabel("gap  = (agent - optimal) / optimal")
        ax.set_ylabel("rollouts")
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"save {out_path}")


def terminal_bars(eval_json: str, out_path: str) -> None:
    print("viz term")
    d = _load(eval_json)
    rollouts = d["rollouts"]
    policies = sorted({r["policy"] for r in rollouts})
    tiers = sorted({r["tier"] for r in rollouts})
    means: Dict[str, List[float]] = {p: [] for p in policies}
    for tier in tiers:
        for p in policies:
            sel = [r["terminal"] for r in rollouts if r["tier"] == tier and r["policy"] == p]
            means[p].append(sum(sel) / max(len(sel), 1))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    width = 0.25
    x = list(range(len(tiers)))
    for i, p in enumerate(policies):
        offs = [xi + i * width for xi in x]
        ax.bar(offs, means[p], width=width, label=p)
    ax.set_xticks([xi + width for xi in x])
    ax.set_xticklabels([f"tier {t}" for t in tiers])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("mean terminal reward")
    ax.set_title("baseline policies: terminal reward by tier")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"save {out_path}")


def reward_trajectory(rollout_json: str, out_path: str) -> None:
    print("viz traj")
    data = _load(rollout_json)
    traj = data.get("trajectory") or data.get("rollouts", [])
    steps = list(range(len(traj)))
    rewards = [x.get("reward", 0.0) for x in traj]
    agent_cost = [x.get("agent_cost", 0.0) for x in traj]

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes[0].plot(steps, rewards, marker="o", linewidth=1)
    axes[0].set_ylabel("step reward")
    axes[0].set_title("rollout reward per step")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(steps, agent_cost, color="crimson", linewidth=1.2)
    axes[1].set_xlabel("env step")
    axes[1].set_ylabel("cumulative agent cost")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"save {out_path}")


def training_curve(log_csv: str, out_path: str) -> None:
    print("viz train")
    import csv

    steps: List[int] = []
    mean_r: List[float] = []
    with open(log_csv, "r") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            steps.append(int(row.get("step", len(steps))))
            mean_r.append(float(row.get("mean_reward", 0.0)))

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(steps, mean_r, linewidth=1.2)
    ax.set_xlabel("grpo step")
    ax.set_ylabel("mean cumulative reward")
    ax.set_title("grpo training curve")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"save {out_path}")


def main() -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    ph = sub.add_parser("gap")
    ph.add_argument("--eval", required=True)
    ph.add_argument("--out", default="gap_hist.png")

    pt = sub.add_parser("terminal")
    pt.add_argument("--eval", required=True)
    pt.add_argument("--out", default="terminal_bars.png")

    pr = sub.add_parser("traj")
    pr.add_argument("--rollout", required=True)
    pr.add_argument("--out", default="traj.png")

    pc = sub.add_parser("train")
    pc.add_argument("--csv", required=True)
    pc.add_argument("--out", default="train_curve.png")

    args = p.parse_args()
    if args.cmd == "gap":
        gap_histogram(args.eval, args.out)
    elif args.cmd == "terminal":
        terminal_bars(args.eval, args.out)
    elif args.cmd == "traj":
        reward_trajectory(args.rollout, args.out)
    elif args.cmd == "train":
        training_curve(args.csv, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
