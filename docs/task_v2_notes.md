# Task version v2-pidassist — design notes

**Date:** 28 August 2026
**Supersedes:** v1-raw4d (marked `invalid`, kept as evidence)
**Status:** verified; learnability confirmation pending

---

## Why v1 was abandoned

Measured over 300,000 steps × 3 seeds:

| Symptom | Value |
|---|---|
| Episode length | 20–100 of 720 steps (2.8–14%) |
| Seed 1 trajectory | 32 → 38 steps (got *worse*) |
| Seed variance | 139% of the mean (kill-criterion K3) |
| Return "improvement" | +59% — **an artifact** |

### Three compounding causes

**1. The task was unlearnable from scratch.** Direct 4-D rotor control asked a
randomly-initialised network to discover attitude stabilisation *and*
formation-keeping simultaneously. A quadrotor is underactuated and falls within
~1 s of imperfect thrust, so it never cleared step one.

**2. The reward paid the agent to crash.** Per-step penalty + early termination
meant dying at step 30 accumulated far fewer penalties than flying badly for
720 steps. **Returns rose 59% while episode length stayed flat — the policy was
learning to die faster.**

**3. Wind was never applied.** See the separate section below; this was found
only after the first two were fixed.

> **Generalisable lesson:** when return improves but episode length does not,
> suspect the termination/reward interaction. Rising return alone is *not*
> evidence of learning.

---

## Fix 1 — PID-assisted attitude control

```
policy → desired position offset from slot (3-D)
              ↓
        DSL PID inner loop
              ↓
        rotor RPMs → physics
```

The policy now learns **formation strategy** — "given the wind and my
neighbours, where should I aim?" — instead of how to keep a quadrotor upright.

**Why this strengthens rather than weakens the paper:**

- The coupling question is untouched. Downwash still perturbs drones; the
  critic's job is still to explain variance the actor cannot observe.
- It is **more realistic**. Every deployed drone stack has an inner
  stabilisation loop, so Claim 4 (sim-to-real) becomes more credible.
- It matches standard practice in quadrotor-swarm RL (Batra et al. 2021,
  CrazyMARL 2025), so reviewers will not find it unusual.

**Action space changed: 4-D rotor RPM → 3-D position offset.**

---

## Fix 2 — Survival shaping

Two changes make dying strictly worse than flying:

- per-step **survival bonus** (`+1.0`)
- **crash penalty** scaled by worst-case remaining return

### Sizing the crash penalty

A flat penalty of 50 was not enough. Computing the break-even:

| Sustained lateral error | Full-episode return | vs crashing at step 60 |
|---|---|---|
| 0.30 m | +72 | better |
| **0.31 m** | — | **break-even** |
| 0.40 m | −432 | *worse — would rather crash* |

That hole sat at 31 cm of sustained error — outside where a working policy
lives, but a real gap in the incentive landscape, and exactly the class of bug
that killed v1. So the penalty now scales with the **worst-case remaining
return**, closing it everywhere rather than pushing it out of reach.

After the fix, crashing at step 60 scores **−58,730** — surviving always wins.

---

## Fix 3 — The sidestep exploit

**Identified during design review, before it could corrupt any results.**

At 1 m separation, a 10 cm lateral offset removes **86% of the downwash** for a
reward cost of only 0.01 — about 0.45% of a typical return.

The optimal policy would therefore be to **step aside**, which would eliminate
the very coupling this paper studies. The k-sweep would come back flat, and
that flatness would look like "no effect" when it was really "the agents
engineered the effect away."

**Fix:** `lateral_weight = 10.0`, so lateral error costs 10× vertical error.

| Offset | Downwash removed | Reward cost | Benefit | Pays? |
|---|---|---|---|---|
| 0.05 m | 39% | 18.0 | 1.7 | no |
| 0.10 m | 86% | 72.0 | 3.8 | no |
| 0.20 m | 100% | 288.0 | 4.4 | no |

Holding the slot always beats dodging the wake.

> **Analytic bound only.** The real test is measuring `eval_lateral_error_m` in
> *trained* policies — now logged at every evaluation.

---

## Fix 4 — Wind was never applied ⚠️

**Found 28 Aug 2026, after Fixes 1–3.** Inherited from v1.

`_wind_vec` was sampled every episode, stored, and passed to the critic as the
privileged variable — but **never applied as a force**. It was decorative.

### How it was caught

A pure-PID policy (zero action) was scored at several wind levels:

```
wind_max = 0.0 m/s  ->  598.6 ± 0.00
wind_max = 2.0 m/s  ->  598.6 ± 0.00
wind_max = 5.0 m/s  ->  598.6 ± 0.00
```

Identical to one decimal, **zero variance**. Wind had no effect whatsoever.

### Why this would have silently faked kill-criterion K1

Wind is the project's **hidden common cause** (Charter §1.5): it pushes every
drone at once without appearing in any agent's observation, which is precisely
the unexplained variance a centralized critic is supposed to explain away.

With no wind actually applied:

- a `k=0` critic could predict returns almost perfectly,
- teammate information had nothing left to add,
- **the k-sweep would have come back flat for a reason unrelated to Claim 1.**

We would have fired K1 and abandoned the primary claim because of a missing
`applyExternalForce` call.

### The fix

Drag force from relative airspeed, applied every control step before the PID
computes its response:

```
F = 0.5 · ρ · C_d · A · |v_rel| · v_rel        v_rel = wind − drone_velocity
```

Applied *before* control so the PID must genuinely react, rather than being
handed a pre-corrected state.

---

## Wind calibration

With wind now real, its magnitude had to be chosen. Pure-PID performance:

| `wind_max` | PID return | Seed std | Headroom | Crash rate |
|---|---|---|---|---|
| 5 m/s | 592.8 | 11.0 | 17.7% | 0% |
| 8 m/s | 573.3 | 39.4 | 20.4% | 0% |
| **12 m/s** | **520.3** | **122.3** | **27.7%** | **0%** |
| 16 m/s | 453.7 | 218.1 | 37.0% | 17% |
| 20 m/s | 419.7 | 241.5 | 41.7% | 17% |

**Selected: 12 m/s.**

The requirement is *headroom* — the PID must not already be optimal, or the
policy has nothing to learn and every rung saturates (kill-criterion K2). At
12 m/s there is 27.7% of the ceiling left to win, substantial seed variance for
the critic to explain, and no crashes.

Below ~5 m/s the PID absorbs everything. Above ~16 m/s crashes appear, which
reintroduces termination effects we would rather avoid.

---

## Verification

`scripts/verify_task_fixes.py` checks all four fixes:

```
PASS  PID assist              722/720 steps survived (v1: 39)
PASS  Survival shaping        ordering A > B > C holds
PASS  Sidestep exploit        dodging never pays at any offset
PASS  Coupling knob           12.7× contrast, monotonic
```

| Metric | v1 | v2 |
|---|---|---|
| Episode completion | 5.4% | **100%** |
| Crash rate | ~100% | **0%** |
| Coupling preserved | — | 0.254 N |

---

## New guards against recurrence

**`check_learning.py` now requires >80% episode completion to pass.** This is
the diagnostic that would have caught v1 — the script existed specifically to
prevent that failure and used the wrong criterion.

**Every evaluation logs:**

| Field | Catches |
|---|---|
| `eval_lateral_error_m` | the sidestep exploit in *trained* policies |
| `eval_crash_rate` | the v1 "learning to die faster" mode |
| `eval_coupling_N` | whether coupling survives training |

---

## Parameters

| Parameter | Value | Rationale |
|---|---|---|
| Action space | 3-D offset | Fix 1 |
| `max_offset_m` | 0.5 | Bounds commanded setpoints |
| `lateral_weight` | 10.0 | Fix 3 — closes the sidestep exploit |
| `survival_bonus` | 1.0 | Fix 2 |
| `crash_penalty` | 50.0 + worst-case scaling | Fix 2 |
| `wind_max_ms` | 12.0 | Calibrated for headroom |
| `formation_spacing_m` | 1.0 | Above the 0.6875 m validity floor |

---

## Open question

The v2 learning check (before the wind fix) showed the policy **starting near
the PID ceiling and degrading** — 591 → 327, oscillating. That is consistent
with a task the PID already solved, leaving exploration able only to make
things worse.

The wind fix should resolve this by creating genuine headroom. **This must be
re-verified before any sweep.**
