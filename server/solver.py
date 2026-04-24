from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

import pulp


@dataclass(frozen=True)
class NodeSpec:
    id: str
    type: str
    initial_inventory: int
    max_capacity: int
    holding_cost: float


@dataclass(frozen=True)
class EdgeSpec:
    src: str
    dst: str
    lead_time: int
    unit_cost: float
    capacity: int


@dataclass(frozen=True)
class Scenario:
    nodes: Tuple[NodeSpec, ...]
    edges: Tuple[EdgeSpec, ...]
    demand: Dict[str, Tuple[int, ...]]
    horizon: int
    penalty: float
    supplier_step_cap: int


def _nodes_by_id(sc: Scenario) -> Dict[str, NodeSpec]:
    return {n.id: n for n in sc.nodes}


def _edge_key(e: EdgeSpec) -> Tuple[str, str]:
    return (e.src, e.dst)


def optimal_cost(sc: Scenario, time_limit: int = 30, msg: int = 0) -> float:
    print("calc milp")
    H = sc.horizon
    ni = _nodes_by_id(sc)
    E = list(sc.edges)
    T_disp = list(range(H))
    T_inv = list(range(H + 1))

    prob = pulp.LpProblem("dsc_co", pulp.LpMinimize)

    x = {
        (i, t): pulp.LpVariable(f"x_{i}_{t}", lowBound=0, cat=pulp.LpInteger)
        for i, _ in enumerate(E)
        for t in T_disp
    }
    I = {
        (n.id, t): pulp.LpVariable(f"I_{n.id}_{t}", lowBound=0, cat=pulp.LpInteger)
        for n in sc.nodes
        for t in T_inv
    }
    u = {
        (n.id, t): pulp.LpVariable(f"u_{n.id}_{t}", lowBound=0, cat=pulp.LpInteger)
        for n in sc.nodes
        for t in T_disp
        if n.type == "retail"
    }

    transit = pulp.lpSum(E[i].unit_cost * x[(i, t)] for i in range(len(E)) for t in T_disp)
    holding = pulp.lpSum(
        ni[n_id].holding_cost * I[(n_id, t)]
        for n_id in ni
        for t in T_inv
    )
    shortage = pulp.lpSum(sc.penalty * u[k] for k in u)
    prob += transit + holding + shortage

    for n in sc.nodes:
        prob += I[(n.id, 0)] == n.initial_inventory

    for n in sc.nodes:
        for t in T_inv:
            prob += I[(n.id, t)] <= n.max_capacity

    for i, e in enumerate(E):
        for t in T_disp:
            prob += x[(i, t)] <= e.capacity

    for n in sc.nodes:
        if n.type == "supplier":
            for t in T_disp:
                out_edges = [i for i, e in enumerate(E) if e.src == n.id]
                prob += pulp.lpSum(x[(i, t)] for i in out_edges) <= sc.supplier_step_cap

    for n in sc.nodes:
        in_edges = [i for i, e in enumerate(E) if e.dst == n.id]
        out_edges = [i for i, e in enumerate(E) if e.src == n.id]
        for t in T_disp:
            arrivals = pulp.lpSum(
                x[(i, t - E[i].lead_time)]
                for i in in_edges
                if t - E[i].lead_time >= 0
            )
            departures = pulp.lpSum(x[(i, t)] for i in out_edges)
            if n.type == "retail":
                d_t = sc.demand.get(n.id, tuple())[t] if t < len(sc.demand.get(n.id, tuple())) else 0
                prob += (
                    I[(n.id, t + 1)]
                    == I[(n.id, t)] + arrivals - departures - d_t + u[(n.id, t)]
                )
            else:
                prob += (
                    I[(n.id, t + 1)] == I[(n.id, t)] + arrivals - departures
                )

    solver = pulp.PULP_CBC_CMD(msg=msg, timeLimit=time_limit)
    status = prob.solve(solver)
    if pulp.LpStatus[status] not in ("Optimal", "Not Solved"):
        print("err milp")
    val = pulp.value(prob.objective)
    if val is None:
        return float("inf")
    return float(val)


def greedy_cost(sc: Scenario) -> float:
    print("calc greedy")
    H = sc.horizon
    ni = _nodes_by_id(sc)
    inv = {n.id: n.initial_inventory for n in sc.nodes}
    pipeline: list[tuple[str, str, int, int]] = []
    total = 0.0
    edges_by_pair = {(e.src, e.dst): e for e in sc.edges}

    for t in range(H):
        for r in [n for n in sc.nodes if n.type == "retail"]:
            dvec = sc.demand.get(r.id, tuple())
            d = dvec[t] if t < len(dvec) else 0
            if d <= 0:
                continue
            warehouses = [n for n in sc.nodes if n.type == "warehouse"]
            for w in warehouses:
                pair = (w.id, r.id)
                if pair in edges_by_pair and inv[w.id] > 0:
                    e = edges_by_pair[pair]
                    q = min(d, inv[w.id], e.capacity)
                    if q <= 0:
                        continue
                    inv[w.id] -= q
                    pipeline.append((w.id, r.id, q, t + e.lead_time))
                    total += e.unit_cost * q
                    d -= q
                    if d <= 0:
                        break
            if d > 0:
                total += sc.penalty * d

        new_pipe: list[tuple[str, str, int, int]] = []
        for src, dst, q, arr in pipeline:
            if arr == t + 1:
                inv[dst] = min(inv[dst] + q, ni[dst].max_capacity)
            else:
                new_pipe.append((src, dst, q, arr))
        pipeline = new_pipe
        for n in sc.nodes:
            total += ni[n.id].holding_cost * inv[n.id]

    return total
