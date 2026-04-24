from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

try:
    from openenv.core.env_server.mcp_environment import MCPEnvironment as _BaseEnv
    from openenv.core.env_server.types import State as _State
    from openenv.core.env_server.mcp_types import CallToolAction, ListToolsAction
    _OPENENV_OK = True
except Exception:
    _OPENENV_OK = False

    class _State:
        def __init__(self, episode_id: str = "", step_count: int = 0):
            self.episode_id = episode_id
            self.step_count = step_count

    class _BaseEnv:
        def __init__(self, mcp: Any = None):
            self._mcp = mcp

    class CallToolAction:
        pass

    class ListToolsAction:
        pass

try:
    from fastmcp import FastMCP
    _FASTMCP_OK = True
except Exception:
    _FASTMCP_OK = False

    class FastMCP:
        def __init__(self, name: str):
            self.name = name
            self._tools: Dict[str, Any] = {}

        def tool(self, fn):
            self._tools[fn.__name__] = fn
            return fn

from pydantic import ValidationError

from .models import (
    DispatchInventory,
    DispatchOrder,
    DSCAction,
    DSCActionEnvelope,
    DSCObservation,
    Node,
    Shipment,
)
from .solver import EdgeSpec, NodeSpec, Scenario, optimal_cost


DENSE_SCHEMA = 0.05
DENSE_VALID = 0.10
DENSE_CAP = 0.40
NEG_PENALTY = -1.0
PHANTOM_PENALTY = 0.0
MAX_CALLS_PER_CYCLE = 5
HORIZON = 30
EPS = 1e-6


def _is_underflow_qty(q: Any) -> bool:
    if isinstance(q, bool):
        return False
    if isinstance(q, float):
        return True
    if isinstance(q, int) and q < 1:
        return True
    return False


@dataclass
class _RuntimeNode:
    id: str
    type: str
    inventory: int
    max_capacity: int
    holding_cost: float
    demand: Tuple[int, ...] = field(default_factory=tuple)


@dataclass
class _RuntimeShipment:
    src: str
    dst: str
    qty: int
    arrival_step: int


def _build_tier(difficulty: int, rng: random.Random) -> Scenario:
    if difficulty <= 1:
        n_sup, n_wh, n_ret = 1, 1, 1
        lead_range = (1, 1)
        demand_lam = 8.0
        sup_cap = 200
    elif difficulty == 2:
        n_sup, n_wh, n_ret = 3, 5, 10
        lead_range = (1, 1)
        demand_lam = 10.0
        sup_cap = 400
    elif difficulty == 3:
        n_sup, n_wh, n_ret = 5, 10, 20
        lead_range = (1, 5)
        demand_lam = 12.0
        sup_cap = 600
    else:
        n_sup, n_wh, n_ret = 7, 14, 28
        lead_range = (1, 7)
        demand_lam = 14.0
        sup_cap = 800

    suppliers = [
        NodeSpec(
            id=f"S{i}",
            type="supplier",
            initial_inventory=rng.randint(200, 500),
            max_capacity=1000,
            holding_cost=0.05,
        )
        for i in range(n_sup)
    ]
    warehouses = [
        NodeSpec(
            id=f"W{i}",
            type="warehouse",
            initial_inventory=rng.randint(20, 80),
            max_capacity=400,
            holding_cost=0.2,
        )
        for i in range(n_wh)
    ]
    retails = [
        NodeSpec(
            id=f"R{i}",
            type="retail",
            initial_inventory=rng.randint(10, 40),
            max_capacity=120,
            holding_cost=0.5,
        )
        for i in range(n_ret)
    ]
    nodes = tuple(suppliers + warehouses + retails)

    edges: List[EdgeSpec] = []
    for s in suppliers:
        for w in warehouses:
            edges.append(
                EdgeSpec(
                    src=s.id,
                    dst=w.id,
                    lead_time=rng.randint(*lead_range),
                    unit_cost=round(rng.uniform(0.5, 2.0), 3),
                    capacity=rng.randint(40, 120),
                )
            )
    for w in warehouses:
        for r in retails:
            edges.append(
                EdgeSpec(
                    src=w.id,
                    dst=r.id,
                    lead_time=max(1, rng.randint(*lead_range) - 1),
                    unit_cost=round(rng.uniform(0.3, 1.5), 3),
                    capacity=rng.randint(20, 80),
                )
            )

    demand: Dict[str, Tuple[int, ...]] = {}
    if difficulty <= 1:
        d_vec = tuple(rng.randint(3, 8) for _ in range(HORIZON))
        for r in retails:
            demand[r.id] = d_vec
    else:
        for r in retails:
            base = rng.uniform(demand_lam * 0.6, demand_lam * 1.4)
            if difficulty >= 4:
                season = [max(0, int(base + 4.0 * math.sin(t / 3.0) + rng.gauss(0, 2))) for t in range(HORIZON)]
                demand[r.id] = tuple(season)
            else:
                demand[r.id] = tuple(max(0, int(rng.gauss(base, 2.5))) for _ in range(HORIZON))

    penalty = 25.0
    return Scenario(
        nodes=nodes,
        edges=tuple(edges),
        demand=demand,
        horizon=HORIZON,
        penalty=penalty,
        supplier_step_cap=sup_cap,
    )


class DSCEnv(_BaseEnv):
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        mcp = FastMCP("dsc_co")

        @mcp.tool
        def query_network(source_id: str, dest_id: str) -> dict:
            return self._handle_query(source_id, dest_id)

        @mcp.tool
        def dispatch_inventory(routes: list) -> dict:
            return self._handle_dispatch(routes)

        @mcp.tool
        def advance_cycle() -> dict:
            return self._handle_advance()

        if _OPENENV_OK:
            super().__init__(mcp)
        else:
            self._mcp = mcp

        self._state = _State(episode_id=str(uuid4()), step_count=0)
        self._difficulty: int = 1
        self._scenario: Optional[Scenario] = None
        self._nodes: Dict[str, _RuntimeNode] = {}
        self._pipeline: List[_RuntimeShipment] = []
        self._adjacency: frozenset = frozenset()
        self._edge_info: Dict[Tuple[str, str], EdgeSpec] = {}
        self._current_step: int = 0
        self._agent_cost: float = 0.0
        self._dense_accum: float = 0.0
        self._calls_this_cycle: int = 0
        self._done: bool = False
        self._last_reward: float = 0.0
        self._last_metadata: Dict[str, Any] = {}
        self._opt_cost: Optional[float] = None
        self._rng = random.Random(0)

    @property
    def state(self) -> _State:
        return self._state

    def _step_impl(self, action: Any, timeout_s: Optional[float] = None, **kwargs: Any):
        return self.step(action)

    async def step_async(self, action: Any, timeout_s: Optional[float] = None, **kwargs: Any):
        if _OPENENV_OK and isinstance(action, (ListToolsAction, CallToolAction)):
            self._state.step_count += 1
            return await super().step_async(action, timeout_s=timeout_s, **kwargs)
        return self.step(action, timeout_s=timeout_s, **kwargs)

    def reset(self, seed: Optional[int] = None, difficulty: int = 1, **kwargs: Any) -> DSCObservation:
        print("reset tier")
        self._difficulty = max(1, min(4, int(difficulty)))
        self._rng = random.Random(seed if seed is not None else 0)
        self._scenario = _build_tier(self._difficulty, self._rng)

        self._nodes = {
            n.id: _RuntimeNode(
                id=n.id,
                type=n.type,
                inventory=n.initial_inventory,
                max_capacity=n.max_capacity,
                holding_cost=n.holding_cost,
                demand=self._scenario.demand.get(n.id, tuple()),
            )
            for n in self._scenario.nodes
        }
        self._adjacency = frozenset((e.src, e.dst) for e in self._scenario.edges)
        self._edge_info = {(e.src, e.dst): e for e in self._scenario.edges}
        self._pipeline = []
        self._current_step = 0
        self._agent_cost = 0.0
        self._dense_accum = 0.0
        self._calls_this_cycle = 0
        self._done = False
        self._last_reward = 0.0
        self._opt_cost = None
        self._state = _State(episode_id=str(uuid4()), step_count=0)
        self._last_metadata = {"tier": self._difficulty, "agent_cost": 0.0}
        return self._observation(reward=0.0, done=False)

    def step(self, action: Any, timeout_s: Optional[float] = None, **kwargs: Any) -> Any:
        if _OPENENV_OK and isinstance(action, (ListToolsAction, CallToolAction)):
            self._state.step_count += 1
            return super().step(action, timeout_s=timeout_s, **kwargs)

        if isinstance(action, DSCAction):
            inner_dict = action.root.model_dump()
        elif isinstance(action, dict):
            inner_dict = action.get("action", action) if "action" in action else action
        elif hasattr(action, "model_dump"):
            dumped = action.model_dump()
            inner_dict = dumped.get("action", dumped) if isinstance(dumped, dict) and "action" in dumped else dumped
        else:
            inner_dict = getattr(action, "__dict__", {})

        if isinstance(inner_dict, dict) and inner_dict.get("kind") == "dispatch_inventory":
            for r in inner_dict.get("routes", []) or []:
                q = r.get("qty") if isinstance(r, dict) else None
                if _is_underflow_qty(q):
                    print("term neg")
                    self._done = True
                    self._last_reward = NEG_PENALTY
                    return self._observation(reward=NEG_PENALTY, done=True)

        try:
            parsed = DSCAction.model_validate(inner_dict)
        except ValidationError:
            print("err schema")
            return self._observation(reward=0.0, done=self._done)

        act = parsed.root
        kind = act.kind
        shaping = self._credit_schema()
        if kind == "query_network":
            self._handle_query(act.source_id, act.dest_id)
        elif kind == "dispatch_inventory":
            routes_payload = [r.model_dump() for r in act.routes]
            res = self._handle_dispatch(routes_payload)
            if res.get("ok"):
                shaping += self._credit_valid()
            if res.get("terminal"):
                print("end ep")
                self._done = True
                self._last_reward = float(res.get("reward", 0.0))
                return self._observation(reward=self._last_reward, done=True)
        elif kind == "advance_cycle":
            self._handle_advance()

        if self._done:
            return self._observation(reward=self._last_reward, done=True)

        self._last_reward = shaping
        return self._observation(reward=shaping, done=False)

    def _credit_schema(self) -> float:
        if self._dense_accum >= DENSE_CAP:
            return 0.0
        inc = min(DENSE_SCHEMA, DENSE_CAP - self._dense_accum)
        self._dense_accum += inc
        return inc

    def _credit_valid(self) -> float:
        if self._dense_accum >= DENSE_CAP:
            return 0.0
        inc = min(DENSE_VALID, DENSE_CAP - self._dense_accum)
        self._dense_accum += inc
        return inc

    def _handle_query(self, source_id: str, dest_id: str) -> dict:
        print("query net")
        self._state.step_count += 1
        self._calls_this_cycle += 1
        e = self._edge_info.get((source_id, dest_id))
        if e is None:
            return {"exists": False, "lead_time": 0, "unit_cost": 0.0, "capacity": 0}
        return {
            "exists": True,
            "lead_time": e.lead_time,
            "unit_cost": e.unit_cost,
            "capacity": e.capacity,
        }

    def _handle_dispatch(self, routes: List[dict]) -> dict:
        print("step req")
        self._state.step_count += 1
        self._calls_this_cycle += 1
        if self._done:
            return {"ok": False, "reason": "done", "terminal": False}

        if self._calls_this_cycle > MAX_CALLS_PER_CYCLE:
            return {"ok": False, "reason": "cap", "terminal": False}

        try:
            validated = DispatchInventory(routes=[DispatchOrder.model_validate(r) for r in routes])
        except ValidationError as ve:
            msg = str(ve)
            if "qty" in msg:
                print("term neg")
                self._done = True
                return {"ok": False, "reason": "neg", "terminal": True, "reward": NEG_PENALTY}
            print("err schema")
            return {"ok": False, "reason": "schema", "terminal": False}

        for order in validated.routes:
            if (order.src, order.dst) not in self._adjacency:
                print("bad edge")
                self._done = True
                return {"ok": False, "reason": "phantom", "terminal": True, "reward": PHANTOM_PENALTY}
            src_node = self._nodes.get(order.src)
            if src_node is None:
                print("bad edge")
                self._done = True
                return {"ok": False, "reason": "phantom", "terminal": True, "reward": PHANTOM_PENALTY}
            if src_node.inventory < order.qty:
                print("no inv")
                return {"ok": False, "reason": "inv", "terminal": False}
            e = self._edge_info[(order.src, order.dst)]
            if order.qty > e.capacity:
                print("no cap")
                return {"ok": False, "reason": "cap", "terminal": False}

        for order in validated.routes:
            e = self._edge_info[(order.src, order.dst)]
            src_node = self._nodes[order.src]
            src_node.inventory -= order.qty
            self._pipeline.append(
                _RuntimeShipment(
                    src=order.src,
                    dst=order.dst,
                    qty=order.qty,
                    arrival_step=self._current_step + e.lead_time,
                )
            )
            self._agent_cost += e.unit_cost * order.qty
            print("disp ok")

        return {"ok": True, "shaping": DENSE_VALID, "terminal": False}

    def _handle_advance(self) -> dict:
        print("adv cyc")
        self._state.step_count += 1
        if self._done:
            return {"done": True}

        self._current_step += 1
        self._calls_this_cycle = 0

        remaining: List[_RuntimeShipment] = []
        for ship in self._pipeline:
            if ship.arrival_step == self._current_step:
                dst = self._nodes.get(ship.dst)
                if dst is not None:
                    space = dst.max_capacity - dst.inventory
                    land = min(space, ship.qty)
                    dst.inventory += land
            else:
                remaining.append(ship)
        self._pipeline = remaining

        for n in self._nodes.values():
            if n.type == "retail":
                t = self._current_step - 1
                d = n.demand[t] if 0 <= t < len(n.demand) else 0
                if n.inventory >= d:
                    n.inventory -= d
                else:
                    unmet = d - n.inventory
                    n.inventory = 0
                    if self._scenario is not None:
                        self._agent_cost += self._scenario.penalty * unmet

        for n in self._nodes.values():
            self._agent_cost += n.holding_cost * n.inventory

        if self._current_step >= HORIZON:
            self._finalize()
            return {"done": True, "agent_cost": self._agent_cost}

        return {"done": False, "step": self._current_step}

    def _finalize(self) -> None:
        print("opt gap")
        self._done = True
        if self._scenario is None:
            self._last_reward = 0.0
            return
        try:
            opt = optimal_cost(self._scenario, time_limit=30, msg=0)
        except Exception:
            print("err milp")
            opt = float("inf")
        self._opt_cost = opt
        agent = max(self._agent_cost, EPS)
        terminal = 0.0 if not math.isfinite(opt) else max(0.0, min(1.0, opt / agent))
        self._last_reward = terminal
        self._last_metadata = {
            "tier": self._difficulty,
            "agent_cost": self._agent_cost,
            "optimal_cost": opt,
            "terminal": terminal,
        }

    def _observation(self, reward: float, done: bool) -> DSCObservation:
        nodes = [
            Node(
                id=n.id,
                type=n.type,
                inventory=int(n.inventory),
                max_capacity=int(n.max_capacity),
                holding_cost=float(n.holding_cost),
                demand_forecast=(
                    list(n.demand[self._current_step : self._current_step + 3])
                    if n.type == "retail"
                    else []
                ),
            )
            for n in self._nodes.values()
        ]
        pipeline = [
            Shipment(src=s.src, dst=s.dst, qty=int(s.qty), arrival_step=int(s.arrival_step))
            for s in self._pipeline
        ]
        meta = dict(self._last_metadata)
        meta["agent_cost"] = round(self._agent_cost, 4)
        meta["tier"] = self._difficulty
        meta["calls_this_cycle"] = self._calls_this_cycle
        return DSCObservation(
            step=self._current_step,
            network_status="nominal",
            nodes=nodes,
            pipeline=pipeline,
            reward=float(reward),
            done=bool(done),
            metadata=meta,
        )


def create_dsc_env() -> DSCEnv:
    return DSCEnv()
