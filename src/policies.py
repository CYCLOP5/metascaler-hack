from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import pulp

from .env import DSCEnv, HORIZON
from .solver import EdgeSpec, Scenario


@dataclass
class RolloutResult:
    policy: str
    tier: int
    seed: int
    steps: int
    agent_cost: float
    optimal_cost: float
    gap: float
    terminal: float
    cumulative_dense: float
    trajectory: List[Dict[str, Any]] = field(default_factory=list)


def _run(env: DSCEnv, seed: int, difficulty: int, act_fn, label: str) -> RolloutResult:
    print("run roll")
    obs = env.reset(seed=seed, difficulty=difficulty)
    dense = 0.0
    traj: List[Dict[str, Any]] = []
    last = obs
    steps = 0
    while not last.done and steps < HORIZON * 20:
        a = act_fn(env, last)
        last = env.step({"action": a})
        dense += float(last.reward) if not last.done else 0.0
        traj.append(
            {
                "step": last.step,
                "kind": a.get("kind", ""),
                "reward": float(last.reward),
                "agent_cost": float(last.metadata.get("agent_cost", 0.0)),
                "done": bool(last.done),
            }
        )
        steps += 1

    meta = last.metadata or {}
    agent_c = float(meta.get("agent_cost", env._agent_cost))
    opt_c = float(meta.get("optimal_cost", env._opt_cost or 0.0))
    term = float(meta.get("terminal", last.reward if last.done else 0.0))
    gap = (agent_c - opt_c) / opt_c if opt_c > 0 else float("inf")
    return RolloutResult(
        policy=label,
        tier=difficulty,
        seed=seed,
        steps=steps,
        agent_cost=agent_c,
        optimal_cost=opt_c,
        gap=gap,
        terminal=term,
        cumulative_dense=dense,
        trajectory=traj,
    )


def zero_op_rollout(seed: int = 0, difficulty: int = 1) -> RolloutResult:
    env = DSCEnv()

    def act(e: DSCEnv, obs) -> dict:
        return {"kind": "advance_cycle"}

    return _run(env, seed, difficulty, act, "zero_op")


def _greedy_act(e: DSCEnv, obs) -> dict:
    retail = [n for n in obs.nodes if n.type == "retail"]
    warehouses = [n for n in obs.nodes if n.type == "warehouse"]
    suppliers = [n for n in obs.nodes if n.type == "supplier"]

    sc = e._scenario
    if sc is None:
        return {"kind": "advance_cycle"}
    edges_by_pair = {(edge.src, edge.dst): edge for edge in sc.edges}

    routes: List[dict] = []
    used_qty = {w.id: 0 for w in warehouses}
    used_sup = {s.id: 0 for s in suppliers}

    for r in retail:
        need = sum(r.demand_forecast) + max(0, 5 - r.inventory)
        if need <= 0:
            continue
        best = None
        for w in warehouses:
            pair = (w.id, r.id)
            if pair not in edges_by_pair:
                continue
            avail = w.inventory - used_qty[w.id]
            if avail <= 0:
                continue
            edge = edges_by_pair[pair]
            cand = min(avail, edge.capacity, need)
            if cand <= 0:
                continue
            if best is None or edge.lead_time < best[2].lead_time:
                best = (w, cand, edge)
        if best is not None:
            w, qty, edge = best
            routes.append({"src": w.id, "dst": r.id, "qty": int(qty)})
            used_qty[w.id] += qty

    for w in warehouses:
        if w.inventory - used_qty[w.id] > 60:
            continue
        for s in suppliers:
            pair = (s.id, w.id)
            if pair not in edges_by_pair:
                continue
            avail = s.inventory - used_sup[s.id]
            if avail <= 0:
                continue
            edge = edges_by_pair[pair]
            qty = min(avail, edge.capacity, 80)
            if qty <= 0:
                continue
            if len(routes) >= 8:
                break
            routes.append({"src": s.id, "dst": w.id, "qty": int(qty)})
            used_sup[s.id] += qty
            break
        if len(routes) >= 8:
            break

    if routes:
        return {"kind": "dispatch_inventory", "routes": routes[:8]}
    return {"kind": "advance_cycle"}


def greedy_rollout(seed: int = 0, difficulty: int = 1) -> RolloutResult:
    env = DSCEnv()
    calls = {"k": 0}

    def act(e: DSCEnv, obs) -> dict:
        calls["k"] += 1
        if calls["k"] % 3 == 0 or e._calls_this_cycle >= 4:
            return {"kind": "advance_cycle"}
        return _greedy_act(e, obs)

    return _run(env, seed, difficulty, act, "greedy")


def _solve_and_extract(sc: Scenario) -> Dict[int, List[Dict[str, Any]]]:
    H = sc.horizon
    E = list(sc.edges)
    prob = pulp.LpProblem("dsc_co_plan", pulp.LpMinimize)
    x = {
        (i, t): pulp.LpVariable(f"x_{i}_{t}", lowBound=0, cat=pulp.LpInteger)
        for i in range(len(E))
        for t in range(H)
    }
    I = {
        (n.id, t): pulp.LpVariable(f"I_{n.id}_{t}", lowBound=0, cat=pulp.LpInteger)
        for n in sc.nodes
        for t in range(H + 1)
    }
    u = {
        (n.id, t): pulp.LpVariable(f"u_{n.id}_{t}", lowBound=0, cat=pulp.LpInteger)
        for n in sc.nodes
        for t in range(H)
        if n.type == "retail"
    }

    prob += (
        pulp.lpSum(E[i].unit_cost * x[(i, t)] for i in range(len(E)) for t in range(H))
        + pulp.lpSum(n.holding_cost * I[(n.id, t)] for n in sc.nodes for t in range(H + 1))
        + pulp.lpSum(sc.penalty * u[k] for k in u)
    )

    for n in sc.nodes:
        prob += I[(n.id, 0)] == n.initial_inventory
        for t in range(H + 1):
            prob += I[(n.id, t)] <= n.max_capacity
    for i, e in enumerate(E):
        for t in range(H):
            prob += x[(i, t)] <= e.capacity
    for n in sc.nodes:
        if n.type == "supplier":
            for t in range(H):
                outs = [i for i, e in enumerate(E) if e.src == n.id]
                prob += pulp.lpSum(x[(i, t)] for i in outs) <= sc.supplier_step_cap
    for n in sc.nodes:
        ins = [i for i, e in enumerate(E) if e.dst == n.id]
        outs = [i for i, e in enumerate(E) if e.src == n.id]
        for t in range(H):
            arr = pulp.lpSum(
                x[(i, t - E[i].lead_time)] for i in ins if t - E[i].lead_time >= 0
            )
            dep = pulp.lpSum(x[(i, t)] for i in outs)
            if n.type == "retail":
                d_t = sc.demand.get(n.id, tuple())
                d = d_t[t] if t < len(d_t) else 0
                prob += I[(n.id, t + 1)] == I[(n.id, t)] + arr - dep - d + u[(n.id, t)]
            else:
                prob += I[(n.id, t + 1)] == I[(n.id, t)] + arr - dep

    prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=30))

    plan: Dict[int, List[Dict[str, Any]]] = {t: [] for t in range(H)}
    for (i, t), var in x.items():
        v = int(round(var.value() or 0))
        if v > 0:
            e = E[i]
            plan[t].append({"src": e.src, "dst": e.dst, "qty": v})
    return plan


def optimal_replay_rollout(seed: int = 0, difficulty: int = 1) -> RolloutResult:
    env = DSCEnv()
    obs = env.reset(seed=seed, difficulty=difficulty)
    sc = env._scenario
    assert sc is not None
    plan = _solve_and_extract(sc)

    dense = 0.0
    traj: List[Dict[str, Any]] = []
    last = obs
    steps = 0
    for t in range(HORIZON):
        routes = plan.get(t, [])
        chunks = [routes[i : i + 8] for i in range(0, len(routes), 8)] or [[]]
        for chunk in chunks:
            if chunk:
                last = env.step({"action": {"kind": "dispatch_inventory", "routes": chunk}})
                dense += float(last.reward) if not last.done else 0.0
                traj.append(
                    {"step": last.step, "kind": "dispatch_inventory", "reward": float(last.reward)}
                )
                steps += 1
                if last.done:
                    break
        if last.done:
            break
        last = env.step({"action": {"kind": "advance_cycle"}})
        dense += float(last.reward) if not last.done else 0.0
        traj.append({"step": last.step, "kind": "advance_cycle", "reward": float(last.reward)})
        steps += 1
        if last.done:
            break

    meta = last.metadata or {}
    agent_c = float(meta.get("agent_cost", env._agent_cost))
    opt_c = float(meta.get("optimal_cost", env._opt_cost or 0.0))
    term = float(meta.get("terminal", last.reward if last.done else 0.0))
    gap = (agent_c - opt_c) / opt_c if opt_c > 0 else float("inf")
    return RolloutResult(
        policy="optimal_replay",
        tier=difficulty,
        seed=seed,
        steps=steps,
        agent_cost=agent_c,
        optimal_cost=opt_c,
        gap=gap,
        terminal=term,
        cumulative_dense=dense,
        trajectory=traj,
    )
