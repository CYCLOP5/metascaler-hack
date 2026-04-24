from __future__ import annotations

from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, RootModel, conint, conlist, model_validator


NodeType = Literal["supplier", "warehouse", "retail"]
NetStatus = Literal["nominal", "disrupted"]

StrictPosInt = conint(strict=True, ge=1)
StrictNonNegInt = conint(strict=True, ge=0)


class Node(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(..., min_length=1, max_length=16)
    type: NodeType
    inventory: int = Field(..., ge=0)
    max_capacity: int = Field(..., ge=1)
    holding_cost: float = Field(..., ge=0.0)
    demand_forecast: list[int] = Field(default_factory=list)


class Shipment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    src: str
    dst: str
    qty: int = Field(..., ge=1)
    arrival_step: int = Field(..., ge=0)


class DSCObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    step: int = Field(..., ge=0, le=30)
    network_status: NetStatus = "nominal"
    nodes: list[Node] = Field(default_factory=list)
    pipeline: list[Shipment] = Field(default_factory=list)
    reward: float = 0.0
    done: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class QueryNetwork(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["query_network"] = "query_network"
    source_id: str
    dest_id: str


class DispatchOrder(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    src: str = Field(..., min_length=1, max_length=16)
    dst: str = Field(..., min_length=1, max_length=16)
    qty: StrictPosInt

    @model_validator(mode="before")
    @classmethod
    def _reject_float_qty(cls, data: Any) -> Any:
        if isinstance(data, dict) and "qty" in data and isinstance(data["qty"], bool):
            raise ValueError("err qty")
        if isinstance(data, dict) and "qty" in data and isinstance(data["qty"], float):
            raise ValueError("err qty")
        return data


class DispatchInventory(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["dispatch_inventory"] = "dispatch_inventory"
    routes: conlist(DispatchOrder, min_length=1, max_length=8)


class AdvanceCycle(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["advance_cycle"] = "advance_cycle"


_DSCActionUnion = Annotated[
    Union[QueryNetwork, DispatchInventory, AdvanceCycle],
    Field(discriminator="kind"),
]


class DSCAction(RootModel[_DSCActionUnion]):
    pass


DSCActionEnvelope = DSCAction


class EdgeInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")
    exists: bool
    lead_time: int = 0
    unit_cost: float = 0.0
    capacity: int = 0


class DispatchResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ok: bool
    shaping: float = 0.0
    reason: str = ""
