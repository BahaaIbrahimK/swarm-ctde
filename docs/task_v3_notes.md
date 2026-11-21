# Task version v3-chain-gust — design notes

**Date:** 31 August 2026
**Supersedes:** v2-pidassist (v1-raw4d marked `invalid` earlier)
**Status:** verified; learnability confirmation running

---

## Why v2 was superseded

Two problems, both found by review rather than by a failing test. Neither would
have crashed anything — both would have produced a flat `k`-sweep that looked
like kill-criterion K1.

### Problem 1 — "decentralized execution" was not honest

v2 computed **every** drone's position error against `_formation_targets(t)` —
the analytical leader trajectory:

```python
targets = self._formation_targets(self._sim_time_sec)   # closed-form in t
obs[i, 0:3] = state[0:3] - targets[i]
```

So a follower implicitly knew the leader's entire future path. Three
consequences:

- **The leader was redundant.** Drone 2's target was `leader_target(t) − 2×gap`,
  computable without drone 1 existing. The formation would fly with the leader
  switched off.
- **Not deployable as described.** A real follower would need the trajectory
  formula baked in, or a radio link. Neither is "local observation only."
- **No following dynamics.** No lag, no error propagation, none of what makes
  formation flight interesting.

### Problem 2 — the PID cancelled every disturbance

Measured with a pure-PID policy (policy output forced to zero):

| Disturbance | Seed std |
|---|---|
| Steady wind 3 m/s | 1.66 |
| Steady wind 5 m/s | 1.95 |
| **Downwash at any spacing** | **0.00** |

Downwash at 0.7 m spacing is 0.155 N — 47% of the drone's *spare* thrust, which
sounds substantial. But it is **steady**, and a PID with integral action cancels
a constant force exactly. Turning downwash off entirely changed the return by
9 points out of ~620 (1.5%).

**A critic can only help by explaining variance the actor cannot predict.** With
steady disturbances there is none, so every rung performs identically and the
sweep is flat for reasons unrelated to Claim 1.

> This is why the Phase 1 probe was misleading in isolation: it measured drift
> with **no controller at all**. Under closed-loop control the same force is
> almost entirely rejected.

---

## Fix 1 — Chain topology

```
    🚁 LEADER   ← follows its own absolute trajectory (it has the mission)
    ↑
    🚁 F1       ← observes its gap to the LEADER
    ↑
    🚁 F2       ← observes its gap to F1
    ↑
    🚁 F3       ← observes its gap to F2
```

Every follower quantity is now a **relative measurement to the drone ~1 m
above** — exactly what UWB ranging or onboard vision provides. No global frame,
no trajectory formula, no radio link.

### Why chain rather than star (all followers tracking the leader)

| | Star | Chain |
|---|---|---|
| Sensing range | grows with N (7 m at 8 drones, through occluding bodies) | **constant ~1 m** |
| Failure mode | lose the leader → all followers blind | localised to one link |
| Sensing vs physics | mismatched | **the drone you sense is the drone whose wake hits you** |
| Error propagation | none — followers independent | **yes — string dynamics** |

That last row is the scientific argument. In a chain:

```
leader lags → F1 sees the gap open, accelerates → F1 overshoots
   → F2 reacts to F1's overshoot → error propagates down the chain
```

This is **string stability**, well studied in vehicle platooning. A `k=0` critic
sees "my gap opened" and cannot tell a transient from an amplifying cascade. A
critic seeing upstream neighbours can. Star topology makes followers
independent, leaving far less for a centralized critic to explain.

### Four places had to change together

Changing only the observation would have been cosmetic:

| Location | v2 | v3 |
|---|---|---|
| Observation | error vs. formula | gap to predecessor |
| Reward | distance from formula slot | gap error |
| **PID setpoint** | formula slot | **chain reference** |
| Termination | distance from formula | gap distance |

The PID setpoint matters most — had it kept aiming at the formula, the inner
loop would still be using global trajectory knowledge.

**Reward/observation consistency:** both now call `_reference_positions()`.
Rewarding a quantity the actor cannot observe would make the task partially
unlearnable by construction.

---

## Fix 2 — Stochastic gust field

An Ornstein–Uhlenbeck process per drone:

```
dg = -(1/tau)·g·dt + sigma·√dt·noise
```

A mean-reverting random walk: smooth, temporally correlated fluctuation on a
timescale of `tau` seconds. This is what real air does; white noise is not.

### Parameters are physical, not process coefficients

| Parameter | Meaning | Default |
|---|---|---|
| `gust_std_ms` | typical gust speed (m/s) | **4.0** |
| `gust_tau_sec` | how long a gust persists | 1.0 |
| `gust_spatial_corr` | similarity at neighbouring drones | 0.7 |
| `wind_mean_ms` | steady background component | 1.0 |

Raw σ is meaningless to a reader. `gust_std_ms` is a speed measurable with an
anemometer — which matters because Claim 4 is hardware transfer.

**Indoor reference air speeds:**

```
0.1–0.3 m/s   still room
0.5–1.5 m/s   HVAC draught
2–4   m/s     desk fan at 1–2 m      ← our range
5–8   m/s     fan close up / strong vent
10+   m/s     industrial fan
```

### Spatial correlation — the parameter that matters for this paper

Each drone's gust blends a **shared** and an **independent** component:

- `corr = 0` — independent noise. A neighbour's state tells you nothing about
  your own disturbance, so teammate information is worthless to the critic.
- `corr = 1` — identical gusts. A neighbour being pushed perfectly predicts
  that you are being pushed.

Real air is in between: a gust front sweeping the room hits nearby drones at
correlated times. **That correlation is the mechanism by which a teammate-aware
critic explains variance a local critic cannot.**

---

## Calibration (pure PID, chain topology, 5 seeds)

### Gust strength

| `gust_std_ms` | Return | Seed std | Crash% | Headroom |
|---|---|---|---|---|
| 0.0 | 588.8 | 1.73 | 0 | 18.2% |
| 1.5 | 584.8 | 1.70 | 0 | 18.8% |
| 2.5 | 579.4 | **8.12** | 0 | 19.5% |
| **4.0** | **559.5** | **21.66** | **0** | **22.3%** |

Selected **4.0 m/s** — best variance with no crashes. Compare to v2's 0.00.

### Spatial correlation

| `corr` | Return | Seed std |
|---|---|---|
| 0.0 | 585.1 | 3.92 |
| 0.3 | 581.3 | 5.30 |
| **0.7** | 579.4 | **8.12** |
| 1.0 | 579.3 | 7.88 |

Correlation roughly doubles the variance from 0.0 to 0.7, then saturates.

> **Honest caveat:** the effect is real but weaker than hoped. Gust *strength*
> is the dominant lever; correlation is secondary. If a future experiment sweeps
> correlation as an independent variable, expect a modest effect.

---

## Fix 3 — Two PyBullet force bugs

Both in `_apply_wind()`, both found on 31 Aug.

### Bug A — wrong frame

`LINK_FRAME` interprets the force in the **body** frame, so the wind direction
rotated with the drone. A tilting drone could partly "escape" the wind — not how
air works. Fixed to `WORLD_FRAME`.

### Bug B — wrong application point (introduced by fixing A)

With `WORLD_FRAME`, `posObj` is **also** world coordinates. The code passed
`[0,0,0]` — the world origin. A drone at 3.8 m therefore had its wind force
applied 3.8 m below itself.

**Observed:** a 0.03 N gust (about 1/10th body weight) spun a drone from 0.27 to
**1.70 rad of roll in three control steps**, then terminated. The force was
negligible; the 3.8 m moment arm was not.

Fixed by passing the drone's own position, so the force acts at the centre of
mass and produces pure translation.

> **Lesson:** a partial fix can be worse than the original bug. `verify_task_fixes.py`
> should be run after *any* physics edit, not just at milestones.

---

## Effect on the task

Pure-PID return dropped from **616 (v2) to 572 (v3)** at the same settings.
Expected and desirable: followers now chase a *moving* reference rather than a
perfect formula, so the task is genuinely harder and there is more for a policy
to win.

---

## Consequences for prior results

**The EXP03 pilot (2026-08-30) was produced under v2 and is superseded.** Its
interior optimum (k1 = 674.7 beating k0 = 645.5 and k3 = 626.6) may well be
real, but it was measured with:

- followers tracking an analytical trajectory (not honest decentralization),
- a disturbance the PID cancelled almost entirely,
- wind force applied in the wrong frame at the wrong point.

It must be re-run under v3 before any of it is reported.

---

## Parameters

| Parameter | Value | Rationale |
|---|---|---|
| `chain_topology` | True | Fix 1 |
| `gust_std_ms` | 4.0 | Calibrated for variance with no crashes |
| `gust_tau_sec` | 1.0 | ~1 s gusts, realistic indoors |
| `gust_spatial_corr` | 0.7 | Real air is partly correlated |
| `wind_mean_ms` | 1.0 | Light steady background |
| `FRONTAL_AREA` | 0.0025 m² | Corrected from 0.01 (4× too large) |
| `formation_spacing_m` | 1.0 | Above the 0.6875 m model-validity floor |
