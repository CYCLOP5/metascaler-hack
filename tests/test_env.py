import pytest

from server.dsc_environment import DSCEnv, HORIZON, DENSE_CAP, MAX_CALLS_PER_CYCLE


@pytest.fixture
def env():
    e = DSCEnv()
    e.reset(seed=42, difficulty=1)
    return e


class TestReset:
    def test_tier1_shape(self, env):
        ids = [n.id for n in env._nodes.values()]
        types = [n.type for n in env._nodes.values()]
        assert types.count("supplier") == 1
        assert types.count("warehouse") == 1
        assert types.count("retail") == 1
        assert env._current_step == 0
        assert env._agent_cost == 0.0
        assert not env._done

    def test_tier2_shape(self):
        e = DSCEnv()
        e.reset(seed=42, difficulty=2)
        types = [n.type for n in e._nodes.values()]
        assert types.count("supplier") == 3
        assert types.count("warehouse") == 5
        assert types.count("retail") == 10

    def test_determinism(self):
        e1 = DSCEnv()
        e2 = DSCEnv()
        o1 = e1.reset(seed=99, difficulty=2)
        o2 = e2.reset(seed=99, difficulty=2)
        assert [n.inventory for n in o1.nodes] == [n.inventory for n in o2.nodes]


class TestAntiHackNegative:
    @pytest.mark.parametrize("q", [-1, -100, 0])
    def test_nonpositive_int_penalty(self, env, q):
        obs = env.step({"action": {"kind": "dispatch_inventory", "routes": [{"src": "S0", "dst": "W0", "qty": q}]}})
        assert obs.reward == -1.0
        assert obs.done is True

    @pytest.mark.parametrize("q", [3.7, -2.1, 0.5])
    def test_float_penalty(self, env, q):
        obs = env.step({"action": {"kind": "dispatch_inventory", "routes": [{"src": "S0", "dst": "W0", "qty": q}]}})
        assert obs.reward == -1.0
        assert obs.done is True


class TestAntiHackPhantom:
    def test_phantom_supplier_to_retail(self, env):
        obs = env.step({"action": {"kind": "dispatch_inventory", "routes": [{"src": "S0", "dst": "R0", "qty": 5}]}})
        assert obs.done is True
        assert obs.reward == 0.0

    def test_query_phantom_returns_not_exists(self, env):
        r = env._handle_query("S0", "R0")
        assert r["exists"] is False


class TestAntiHackCap:
    def test_dense_reward_cap_stops(self, env):
        for _ in range(50):
            env.step({"action": {"kind": "query_network", "source_id": "S0", "dest_id": "W0"}})
        assert env._dense_accum <= DENSE_CAP + 1e-9


class TestValidFlow:
    def test_valid_dispatch_deducts_inventory(self, env):
        before = env._nodes["S0"].inventory
        obs = env.step({"action": {"kind": "dispatch_inventory", "routes": [{"src": "S0", "dst": "W0", "qty": 10}]}})
        assert obs.done is False
        assert env._nodes["S0"].inventory == before - 10
        assert len(env._pipeline) == 1

    def test_insufficient_inventory_noop(self, env):
        obs = env.step({"action": {"kind": "dispatch_inventory", "routes": [{"src": "S0", "dst": "W0", "qty": 99999}]}})
        assert obs.done is False

    def test_advance_ticks_time(self, env):
        env.step({"action": {"kind": "advance_cycle"}})
        assert env._current_step == 1

    def test_full_horizon_ends(self, env):
        last = None
        for _ in range(HORIZON + 2):
            last = env.step({"action": {"kind": "advance_cycle"}})
            if last.done:
                break
        assert last.done
        assert "terminal" in last.metadata
        assert 0.0 <= last.metadata["terminal"] <= 1.0


class TestAdjacencyImmutability:
    def test_adjacency_frozen_after_reset(self, env):
        adj = env._adjacency
        assert isinstance(adj, frozenset)
        with pytest.raises((AttributeError, TypeError)):
            adj.add(("X", "Y"))
