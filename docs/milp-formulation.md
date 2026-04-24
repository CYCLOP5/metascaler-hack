# milp formulation

## sets

- N = set of nodes (suppliers, warehouses, retail)
- E = set of directed edges (supplier -> warehouse, warehouse -> retail)
- T_disp = {0, 1, ..., H - 1}  (dispatch steps)
- T_inv  = {0, 1, ..., H}      (inventory snapshots)
- H = 30

## parameters

- c_e         unit transit cost on edge e
- L_e         lead time on edge e (integer steps)
- cap_e       per-step flow capacity on edge e
- cap_n       inventory capacity at node n
- h_n         holding cost per unit per step at node n
- d[n, t]     retail demand at node n at step t (0 if n is not retail)
- I0_n        initial inventory at node n
- sup_cap     per-step supplier outflow cap
- P           shortage penalty (large, > max transit cost)

## decision variables

- x[e, t] in Z>=0       dispatched flow on e at step t
- I[n, t] in Z>=0       inventory at end of step t
- u[n, t] in Z>=0       unmet demand at retail n at step t

## objective

```
min  sum_{e in E, t in T_disp} c_e * x[e, t]
   + sum_{n in N, t in T_inv } h_n * I[n, t]
   + sum_{n in retail, t in T_disp} P * u[n, t]
```

## constraints

### initial state

```
I[n, 0] = I0_n                   for all n in N
```

### inventory capacity

```
I[n, t] <= cap_n                 for all n in N, t in T_inv
```

### edge capacity

```
x[e, t] <= cap_e                 for all e in E, t in T_disp
```

### supplier step cap

```
sum_{e: src(e) = s} x[e, t] <= sup_cap        for all s supplier, t in T_disp
```

### flow conservation

for every n in N and every t in T_disp:

```
arrivals(n, t)   = sum_{e: dst(e) = n, t - L_e >= 0} x[e, t - L_e]
departures(n, t) = sum_{e: src(e) = n} x[e, t]
```

retail:

```
I[n, t + 1] = I[n, t] + arrivals(n, t) - departures(n, t) - d[n, t] + u[n, t]
```

non-retail:

```
I[n, t + 1] = I[n, t] + arrivals(n, t) - departures(n, t)
```

## solver

```
pulp.PULP_CBC_CMD(msg=0, timeLimit=30)
```

returns `pulp.value(prob.objective)`. non-optimal status returns `float('inf')` at the python layer.

## complexity

for tier 2 (3 suppliers, 5 warehouses, 10 retail, H=30):
- edges ~ 3*5 + 5*10 = 65
- flow vars: 65 * 30 = 1950
- inventory vars: 18 * 31 = 558
- unmet vars: 10 * 30 = 300

cbc dispatches this in well under a second on a single core. tier 3 fits the same 30s budget. tier 4 is gated to manual use.
