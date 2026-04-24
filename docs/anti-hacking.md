# anti-hacking

three classes of specification-gaming attacks are anticipated and each has a hard programmatic gate.

## 1. negative integer underflow

attack: llm emits `dispatch_inventory` with `qty = -10000`. a naive subtract `inv -= qty` creates infinite virtual inventory.

defense, layered:
- pydantic model `DispatchOrder.qty` is `conint(strict=True, ge=1)` rejecting 0, negatives, booleans, floats, strings
- `DispatchOrder` `model_validator(mode='before')` explicitly rejects bool and float qty even if pydantic might coerce
- on `ValidationError` with `"qty"` in the message: env terminates with `reward = -1.0`, `done = true`
- no mutation of `self._nodes` occurs before validation passes

emits: `print("term neg")`.

## 2. cyclic dense-reward farming

attack: llm loops `warehouse_a -> warehouse_b -> warehouse_a` harvesting `r_valid = +0.10` per dispatch.

defense:
- per-cycle tool-call cap: `MAX_CALLS_PER_CYCLE = 5`. after exceeding, `dispatch_inventory` returns `ok=false`, `reason="cap"`
- per-episode dense reward cap: `DENSE_CAP = 0.40`. once the accumulated `r_schema + r_valid` reaches `0.40`, further dense rewards are zero
- holding cost (`h_n` at warehouses = 0.20) strictly exceeds the dense shaping gain per unit-step, so cyclic routing strictly reduces total reward once the cap is hit
- terminal reward (optimality ratio) dominates; agents that loop finish with poor `agent_cost` and low `r_terminal`

## 3. phantom edge hallucination

attack: llm invents a direct supplier -> retail edge that bypasses warehouse lead time.

defense:
- immutable `self._adjacency: frozenset[(str, str)]` built at reset from the generated scenario
- any `dispatch_inventory` route with `(src, dst)` not in `_adjacency` ends the episode: `done = true`, `reward = 0.0`, `reason = "phantom"`
- `query_network` on a non-existent edge returns `exists: false` so the agent cannot pretend ignorance
- `_edge_info` dict is source of truth for `lead_time`, `unit_cost`, `capacity`; the env never consults agent-provided edge parameters

emits: `print("bad edge")`.

## audit surface

- `metadata.calls_this_cycle` reported on every observation
- `metadata.agent_cost` reported on every observation
- terminal observation includes `optimal_cost` and `terminal` for external verification
- `print` tracing uses short lowercase tokens: `init sim`, `reset tier`, `query net`, `step req`, `disp ok`, `adv cyc`, `opt gap`, `err qty`, `err schema`, `err milp`, `term neg`, `bad edge`, `end ep`, `no inv`, `no cap`
