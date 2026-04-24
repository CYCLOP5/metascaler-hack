import random

import pytest

from src.env import _build_tier
from src.solver import EdgeSpec, NodeSpec, Scenario, greedy_cost, optimal_cost


def _trivial_scenario(h: int = 5) -> Scenario:
    nodes = (
        NodeSpec(id="S0", type="supplier", initial_inventory=100, max_capacity=1000, holding_cost=0.0),
        NodeSpec(id="W0", type="warehouse", initial_inventory=0, max_capacity=200, holding_cost=0.0),
        NodeSpec(id="R0", type="retail", initial_inventory=0, max_capacity=50, holding_cost=0.0),
    )
    edges = (
        EdgeSpec(src="S0", dst="W0", lead_time=1, unit_cost=1.0, capacity=50),
        EdgeSpec(src="W0", dst="R0", lead_time=1, unit_cost=1.0, capacity=50),
    )
    demand = {"R0": tuple([5] * h)}
    return Scenario(
        nodes=nodes,
        edges=edges,
        demand=demand,
        horizon=h,
        penalty=100.0,
        supplier_step_cap=100,
    )


class TestOptimalCost:
    def test_trivial_closed_form(self):
        sc = _trivial_scenario(h=5)
        cost = optimal_cost(sc, time_limit=10)
        assert cost >= 0.0
        assert cost < 1e6

    def test_tier1_monotone(self):
        for seed in [1, 2, 3]:
            sc = _build_tier(1, random.Random(seed))
            g = greedy_cost(sc)
            o = optimal_cost(sc, time_limit=10)
            assert o <= g + 1e-6, f"optimal must be <= greedy, seed {seed}"

    def test_tier2_feasible(self):
        sc = _build_tier(2, random.Random(7))
        o = optimal_cost(sc, time_limit=30)
        assert o > 0.0
        assert o != float("inf")


class TestCurriculum:
    @pytest.mark.parametrize("tier,ns,nw,nr", [(1, 1, 1, 1), (2, 3, 5, 10), (3, 5, 10, 20)])
    def test_tier_shapes(self, tier, ns, nw, nr):
        sc = _build_tier(tier, random.Random(0))
        sups = [n for n in sc.nodes if n.type == "supplier"]
        whs = [n for n in sc.nodes if n.type == "warehouse"]
        rets = [n for n in sc.nodes if n.type == "retail"]
        assert len(sups) == ns
        assert len(whs) == nw
        assert len(rets) == nr

    def test_tier_bipartite_edges(self):
        sc = _build_tier(2, random.Random(0))
        for e in sc.edges:
            src_type = next(n.type for n in sc.nodes if n.id == e.src)
            dst_type = next(n.type for n in sc.nodes if n.id == e.dst)
            assert (src_type, dst_type) in (("supplier", "warehouse"), ("warehouse", "retail"))

    def test_demand_nonnegative(self):
        sc = _build_tier(3, random.Random(11))
        for r_id, d in sc.demand.items():
            assert all(q >= 0 for q in d)
