import pytest
from pydantic import ValidationError

from server.models import (
    DSCAction,
    DSCObservation,
    DispatchOrder,
    Node,
    Shipment,
)


class TestDispatchOrderValidation:
    def test_valid_int_qty(self):
        o = DispatchOrder(src="S0", dst="W0", qty=5)
        assert o.qty == 5

    @pytest.mark.parametrize("q", [-1, -100, 0])
    def test_rejects_non_positive_int(self, q):
        with pytest.raises(ValidationError):
            DispatchOrder(src="S0", dst="W0", qty=q)

    @pytest.mark.parametrize("q", [3.7, 1.0, -0.5])
    def test_rejects_float(self, q):
        with pytest.raises(ValidationError):
            DispatchOrder(src="S0", dst="W0", qty=q)

    @pytest.mark.parametrize("q", [True, False])
    def test_rejects_bool(self, q):
        with pytest.raises(ValidationError):
            DispatchOrder(src="S0", dst="W0", qty=q)

    def test_rejects_string(self):
        with pytest.raises(ValidationError):
            DispatchOrder(src="S0", dst="W0", qty="5")


class TestActionEnvelope:
    def test_query_parses(self):
        a = DSCAction.model_validate({"kind": "query_network", "source_id": "S0", "dest_id": "W0"})
        assert a.root.kind == "query_network"
        assert a.root.source_id == "S0"

    def test_dispatch_parses(self):
        a = DSCAction.model_validate(
            {"kind": "dispatch_inventory", "routes": [{"src": "S0", "dst": "W0", "qty": 3}]}
        )
        assert len(a.root.routes) == 1
        assert a.root.routes[0].qty == 3

    def test_advance_parses(self):
        a = DSCAction.model_validate({"kind": "advance_cycle"})
        assert a.root.kind == "advance_cycle"

    def test_unknown_kind_rejected(self):
        with pytest.raises(ValidationError):
            DSCAction.model_validate({"kind": "hack_world"})

    def test_empty_routes_rejected(self):
        with pytest.raises(ValidationError):
            DSCAction.model_validate({"kind": "dispatch_inventory", "routes": []})

    def test_too_many_routes_rejected(self):
        routes = [{"src": "S0", "dst": "W0", "qty": 1}] * 9
        with pytest.raises(ValidationError):
            DSCAction.model_validate({"kind": "dispatch_inventory", "routes": routes})


class TestObservation:
    def test_accepts_valid(self):
        obs = DSCObservation(
            step=0,
            nodes=[Node(id="S0", type="supplier", inventory=10, max_capacity=100, holding_cost=0.1)],
            pipeline=[],
            reward=0.0,
            done=False,
        )
        assert obs.step == 0

    def test_rejects_step_out_of_range(self):
        with pytest.raises(ValidationError):
            DSCObservation(step=31, nodes=[], pipeline=[], reward=0.0, done=False)
