# 04 — Phase 2: the experimental apparatus

```bash
cd ~/projects/swarm-ctde
source .venv/bin/activate
```

---

## What Phase 2 builds

The machinery that makes the experiment interpretable — the task, the
independent variable, and the confound controls.

| Component | File | Role |
|---|---|---|
| Coupling model | `src/ctde/envs/coupling.py` | Ranks teammates by aerodynamic influence |
| Formation task | `src/ctde/envs/formation_aviary.py` | The environment agents learn in |
| Observability ladder | `src/ctde/models/observability.py` | **The independent variable (`k`)** |
| Actor / Critic | `src/ctde/models/networks.py` | Networks, with capacity control |

---

## Inspect the apparatus

```bash
python scripts/inspect_ladder.py
```

Prints four things. Read them in order — together they *are* the experimental
design.

### 1. The observability ladder

```
  rung     teammates   privileged   critic input dim
  k0       0           False        12
  k1       1           False        18
  k2       2           False        24
  k3       3           False        30
  k3+P     3           True         33
```

`k0` is a fully decentralized critic; `k3+P` adds privileged simulator state
(the wind vector) no real drone could measure.

**Both endpoints matter.** Claim 1 requires the interior optimum to beat *both*
— beating only `k0` would merely re-show that centralization helps, which has
been known since MADDPG (2017).

### 2. Capacity control ← the most important output

```
  input_dim    projection     trunk          total
  12           832            82,689         83,521
  30           1,984          82,689         84,673
  PASS: trunk identical across all rungs (82,689 params)
```

**Why this exists.** If a wider critic input simply meant a bigger network,
then `k=3` would have more *capacity* than `k=1`, and we would be measuring
capacity rather than information. That is the single most likely reviewer
objection to the paper.

The fix is a fixed-width projection layer: every rung projects its
variable-width input to the same dimension before an identical trunk.

Note the direction of the residual difference: the 1,344-parameter spread
favours **high** `k`. So if performance still drops at high `k`, extra capacity
cannot explain it. The bias works against our own hypothesis, which is what
makes the control credible.

### 3. Neighbour ranking

```
  0.6 m above, 0.00 m offset (near, aligned)   0.21067 N
  0.6 m above, 0.10 m offset (near, offset)    0.00000 N
  1.5 m above, 0.00 m offset (far,  aligned)   0.03371 N
  0.5 m BELOW (wake flows downward)            0.00000 N
```

This is Phase 1's finding turned into a design decision. A drone 0.6 m above
but 10 cm off-axis exerts **zero** force; one 1.5 m above and aligned exerts
real force.

So `k=1` means *"the teammate whose wake affects me most"* — not the
Euclidean-nearest. Ranking by distance would hand the critic an irrelevant
teammate and silently corrupt the ladder.

### 4. Claim 2 manipulation check

```
  spacing (m)    total coupling (N)
  0.70           0.5579
  1.00           0.2737
  2.50           0.0438
  Coupling contrast: 12.7x     Monotonic: YES
```

Claim 2 says `k*` tracks coupling strength, which we vary via formation
spacing. This proves the knob changes the **physics**, not just the commanded
geometry — otherwise Claim 2 would rest on an unverified assumption.

---

## Run the tests

```bash
pytest tests/ -v                      # everything (~6 s)
pytest tests/ -m "not slow"           # skip simulator-dependent tests
pytest tests/test_phase2.py -v        # Phase 2 only
```

Two tests guard the paper's central controls and must never be deleted:

| Test | Guards |
|---|---|
| `test_critic_trunk_identical_across_ladder` | Capacity is not confounded with information |
| `test_actor_identical_across_ladder` | Only the critic's input varies with `k` |

If a later refactor breaks either, the sweep becomes uninterpretable — the
tests exist so that failure is loud rather than silent.

---

## Design decisions worth knowing

### Why the task is harder than the old one

The previous project's task would have produced a flat curve at every `k`.
Four fixes:

| Old | New | Why |
|---|---|---|
| One-way reward coupling | Vertical stack in shared downwash | Real mutual influence |
| `ONE_D_RPM`, ±5% hover | 3-D position offset (±0.2 m) over a PID stack | Policy learns *strategy*, not attitude control |
| 5 s episodes | 15 s | Coupling and wind accumulate |
| No disturbance | Per-episode random wind (0–12 m/s, **applied**) | Something for teammate state to be *informative about* |

The principle in one sentence: **a centralized critic can only help if there is
something only it can see that genuinely predicts return.**

### Why wind is unobservable

Wind is resampled every episode and appears in **no** agent's observation. It is
a hidden common cause — it pushes all drones simultaneously, so a teammate's
velocity carries information about a disturbance the agent cannot sense
directly. Exactly the variance a centralized critic might explain away
(Charter §1.5).

### Stack geometry

The formation is built **upward** from a floor clearance, not downward from a
fixed leader altitude. Building downward puts the lowest drone underground as
soon as `num_drones × spacing` exceeds the leader's altitude. `base_altitude_m`
is the altitude of the **lowest** drone; the leader sits above the stack.

---

## Next

Phase 3 — the MAPPO trainer and the pilot sweep (`k ∈ {0, 1, N-1}`, 5 seeds).
That is the **decision gate**: if the curve is flat, we stop and pivot before
spending months.
