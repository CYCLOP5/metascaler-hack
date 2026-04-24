import pytest
from pydantic import ValidationError

from src.models import (
    DSCActionEnvelope,
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
        e = DSCActionEnvelope.model_validate(
            {"action": {"kind": "query_network", "source_id": "S0", "dest_id": "W0"}}
        )
        assert e.action.kind == "query_network"
        assert e.action.source_id == "S0"

    def test_dispatch_parses(self):
        e = DSCActionEnvelope.model_validate(
            {
                "action": {
                    "kind": "dispatch_inventory",
                    "routes": [{"src": "S0", "dst": "W0", "qty": 3}],
                }
            }
        )
        assert len(e.action.routes) == 1
        assert e.action.routes[0].qty == 3

    def test_advance_parses(self):
        e = DSCActionEnvelope.model_validate({"action": {"kind": "advance_cycle"}})
        assert e.action.kind == "advance_cycle"

    def test_unknown_kind_rejected(self):
        with pytest.raises(ValidationError):
            DSCActionEnvelope.model_validate({"action": {"kind": "hack_world"}})

    def test_empty_routes_rejected(self):
        with pytest.raises(ValidationError):
            DSCActionEnvelope.model_validate(
                {"action": {"kind": "dispatch_inventory", "routes": []}}
            )

    def test_too_many_routes_rejected(self):
        routes = [{"src": "S0", "dst": "W0", "qty": 1}] * 9
        with pytest.raises(ValidationError):
            DSCActionEnvelope.model_validate(
                {"action": {"kind": "dispatch_inventory", "routes": routes}}
            )


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
