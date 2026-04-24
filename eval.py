from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

from server.policies import greedy_rollout, optimal_replay_rollout, zero_op_rollout


POLICIES = {
    "zero_op": zero_op_rollout,
    "greedy": greedy_rollout,
    "optimal_replay": optimal_replay_rollout,
}


def _agg(values):
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def run(policies, tiers, seeds, out_path):
    print("run eval")
    results = []
    t0 = time.time()
    for tier in tiers:
        for policy in policies:
            fn = POLICIES[policy]
            for seed in seeds:
                r = fn(seed=seed, difficulty=tier)
                results.append(
                    {
                        "policy": r.policy,
                        "tier": r.tier,
                        "seed": r.seed,
                        "agent_cost": r.agent_cost,
                        "optimal_cost": r.optimal_cost,
                        "gap": r.gap,
                        "terminal": r.terminal,
                        "cumulative_dense": r.cumulative_dense,
                        "steps": r.steps,
                    }
                )
    dt = time.time() - t0

    summary = {}
    for tier in tiers:
        for policy in policies:
            sel = [r for r in results if r["tier"] == tier and r["policy"] == policy]
            summary[f"tier{tier}/{policy}"] = {
                "gap": _agg([r["gap"] for r in sel if r["gap"] != float("inf")]),
                "terminal": _agg([r["terminal"] for r in sel]),
                "agent_cost": _agg([r["agent_cost"] for r in sel]),
                "optimal_cost": _agg([r["optimal_cost"] for r in sel]),
            }

    out = {
        "meta": {
            "policies": policies,
            "tiers": tiers,
            "n_seeds": len(seeds),
            "duration_s": round(dt, 2),
        },
        "summary": summary,
        "rollouts": results,
    }
    Path(out_path).write_text(json.dumps(out, indent=2))
    print(f"save {out_path}")

    for key, stats in summary.items():
        g = stats["gap"]
        t = stats["terminal"]
        if g.get("n"):
            print(f"{key}: gap mean={g['mean']:.4f} med={g['median']:.4f}  terminal mean={t['mean']:.4f}")
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--policies", nargs="+", default=list(POLICIES.keys()))
    p.add_argument("--tiers", nargs="+", type=int, default=[1, 2])
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--seed0", type=int, default=0)
    p.add_argument("--out", default="eval.json")
    args = p.parse_args()

    seeds = list(range(args.seed0, args.seed0 + args.n))
    run(args.policies, args.tiers, seeds, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
