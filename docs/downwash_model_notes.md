# Downwash model — characterisation notes

**Date:** 23 August 2026
**Phase:** 1 (premise check)
**Status:** premise CONFIRMED, probe recalibrated

---

## Summary

The `gym-pybullet-drones` downwash model (Zhou / DSL) produces **very strong**
inter-agent coupling — far stronger than initially assumed. The premise of the
paper holds comfortably. However, the first probe configuration measured in a
saturated regime and had to be recalibrated.

This document records what was found, because two of these points must be
disclosed in the paper.

---

## The model

From `BaseAviary._downwash()`:

```python
delta_z  = pos[i].z - pos[nth].z          # vertical separation
delta_xy = ||pos[i].xy - pos[nth].xy||    # lateral offset

if delta_z > 0 and delta_xy < 10:
    alpha = DW_COEFF_1 * (PROP_RADIUS / (4*delta_z))**2
    beta  = DW_COEFF_2 * delta_z + DW_COEFF_3
    force_z = -alpha * exp(-0.5 * (delta_xy/beta)**2)
```

Crazyflie 2.x coefficients, loaded from the URDF:

| Constant | Value |
|---|---|
| `DW_COEFF_1` | 2267.18 |
| `DW_COEFF_2` | 0.16 |
| `DW_COEFF_3` | −0.11 |
| `PROP_RADIUS` | 0.023135 m |
| drone weight | 0.265 N (27 g) |

**Structure:** magnitude follows an inverse-square law in vertical separation;
lateral falloff is Gaussian with width `beta`.

---

## Finding 1 — coupling is extremely strong at close range

Force directly beneath the upper drone:

| Separation | Force (N) | As multiple of drone weight |
|---|---|---|
| 0.10 m | 7.584 | **28.6×** |
| 0.15 m | 3.371 | **12.7×** |
| 0.20 m | 1.896 | **7.2×** |
| 0.30 m | 0.843 | 3.2× |
| 0.60 m | 0.211 | 0.80× |
| 1.00 m | 0.076 | 0.29× |
| 1.50 m | 0.034 | 0.13× |
| 2.50 m | 0.012 | 0.05× |

At tight formation spacings the lower drone is not *perturbed* — it is
**overwhelmed**. No hover command can counteract 12× body weight.

### Consequence: the first probe run was saturated

The initial sweep (0.10–1.50 m) produced a flat, distance-independent reading:

```
0.15 m separation -> differential drift -0.109 m
1.00 m separation -> differential drift -0.118 m      ratio 0.9x
```

This *looked* like "no coupling detected", but was the opposite: the drone was
in free fall at every point, and the drift simply measured how far it fell in
the measurement window. A **ceiling effect**, not an absence of effect.

A single-drone control test confirmed the rig itself was sound:

```
solo drone, pyb              drift = -0.000005 m
solo drone, pyb_gnd_drag_dw  drift = +0.000690 m
```

Hover RPM is well calibrated; the sag appears only when a second drone exists.

### Recalibration

`configs/probe_downwash.yaml` now sweeps **0.60 – 2.50 m**, where the force is
a genuine perturbation (0.80× down to 0.05× body weight), with short settle and
measurement windows (0.25 s each) so the drone does not leave the commanded
geometry during measurement.

Result after recalibration — clean monotonic decay:

| Separation | Differential drift (m) | Mean v_z (m/s) |
|---|---|---|
| 0.60 m | −0.4616 | −1.846 |
| 0.80 m | −0.3139 | −1.255 |
| 1.00 m | −0.2207 | −0.883 |
| 1.50 m | −0.1068 | −0.427 |
| 2.00 m | −0.0613 | −0.245 |
| 2.50 m | −0.0385 | −0.154 |

**12× decay across the range.** This is the structure the mechanism argument
(Charter §1.5) requires: strong, local, sharply distance-dependent coupling.

---

## Finding 2 — `beta` goes negative below 0.688 m ⚠️

```
beta = 0.16 * delta_z - 0.11   ->   beta = 0  at  delta_z = 0.6875 m
```

Below that separation `beta` is **negative**. Because it is squared inside the
Gaussian, no NaN or crash occurs and the force stays finite — the failure is
silent.

| Separation | beta |
|---|---|
| 0.10 m | −0.094 |
| 0.30 m | −0.062 |
| 0.60 m | −0.014 |
| **0.688 m** | **≈ 0** |
| 1.00 m | +0.050 |

Physically `beta` is a Gaussian *width*, so a negative value is meaningless.
The model is being evaluated outside the range Zhou calibrated it on.

**Implication for the paper.** Our formation spacings must stay above ~0.69 m
to remain inside the model's valid domain, OR we must disclose the
extrapolation explicitly. Given that Claim 2 rests on varying coupling
strength, a reviewer *will* ask whether the coupling model is trustworthy at
the spacings used. We should:

1. Keep Claim 2 spacings **≥ 0.69 m** where possible, and
2. State this limitation openly in the paper's limitations section, and
3. Note that Phase 8 hardware validation is what ultimately grounds the claim.

This is a good example of why the probe runs before the trainer: the constraint
is now a *design input* rather than a reviewer's objection.

---

## Finding 3 — implications for the observability ladder

Coupling decays as 1/Δz². Using 0.60 m as reference:

| Separation | Relative coupling |
|---|---|
| 0.60 m | 1.00 |
| 1.00 m | 0.36 |
| 1.50 m | 0.16 |
| 2.50 m | 0.06 |

A teammate at 2.5 m exerts ~6% of the influence of one at 0.6 m, yet costs the
critic just as many input dimensions.

**This is the interior-optimum argument made concrete.** Adding near neighbours
to the critic buys real variance reduction; adding distant ones buys almost
nothing while still paying full estimation cost — precisely the saturating-gain
vs. growing-cost trade in Charter §1.5.

The coupling horizon extracted from the full sweep becomes our **a-priori
prediction for k\***, fixed before any training runs.

---

## Actions taken

- `configs/probe_downwash.yaml` — separations moved to 0.60–2.50 m; settle and
  measure windows shortened to 0.25 s
- `configs/base.yaml` — added `probe_upper_altitude_m: 3.0` so the lower drone
  clears ground effect at the widest separations
- `src/ctde/physics/downwash_probe.py` — switched `BaseAviary` → `CtrlAviary`
  (BaseAviary is abstract); updated to the `pyb_freq`/`ctrl_freq` API

## Open question for Phase 2

Should the formation task operate **above** 0.69 m (staying inside the model's
valid domain, weaker coupling), or **below** it (stronger coupling, but
extrapolated physics)?

Recommendation: **above**, with the limitation disclosed. Claim 2 needs a
*range* of coupling strengths, not maximum absolute coupling — and 0.69–2.50 m
already provides a 12× contrast, which is ample.
