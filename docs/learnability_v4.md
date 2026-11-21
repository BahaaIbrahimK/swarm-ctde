# Learnability under v4-correctness — diagnostic record

**Date:** 16 September 2026
**Question:** after the September audit changed five things at once, does the
agent still learn?
**Short answer:** not yet within 30k steps — but the task is sound, a better
policy provably exists, and we now know exactly what to measure.

---

## Why this check was mandatory

v4 changed the observation (12→13-D), the action distribution (unbounded
Gaussian → tanh-squashed), the reward scale, wind application (1 of 5 physics
substeps → all 5), and added value normalization. Any one of those can stop
learning. None had been verified.

---

## Round 1 — FAILED, and the cause was the task, not the learner

| | |
|---|---|
| Completion | 72.9% (need > 80%) |
| Crashes | 17/30 held-out episodes |
| Explained variance | −0.387 (critic worse than predicting the mean) |

Return oscillated between ~30 and ~350 with no trend, tracking crash rate
exactly. That reads like "the agent cannot learn".

**It was not.** A pure PID with *zero RL action* at the same gust strength:

```
gust σ    crashes    completion    RMS
  4.0       40%         73%      22.46 cm
```

73% completion for a controller doing its textbook best — almost exactly the
72.9% the trained policies reached. **No policy could have learned that task.**

### Root cause: a calibration measured against a bug

`gust_std_ms = 4.0` was chosen on 31 Aug as "best variance, still 0% crashes"
— while wind reached the drones on **1 of 5 physics substeps**, i.e. at ~20%
strength. v4 fixed the substep bug. The real disturbance became ~5× what the
calibration described, and nobody re-measured it.

### Recalibration (pure PID, 12 episodes per point, v4 wind)

| σ | seed std | crashes | completion | RMS |
|---|---|---|---|---|
| 2.0 | 23.87 | 0% | 100% | 14.97 cm |
| 2.5 | 37.28 | 0% | 100% | 15.81 cm |
| **3.0** | **53.58** | **0%** | **100%** | **17.13 cm** ← chosen |
| 3.5 | 73.54 | 0% | 100% | 19.06 cm |
| 4.0 | — | 40% | 73% | 22.46 cm |

σ = 3.0 gives **31× the seed variance of the σ=0 baseline** (53.58 vs 1.73), so
a better critic has ample signal to explain, at zero crashes and with margin
before the cliff at 4.0. σ = 3.5 buys more variance for much less safety margin
— and crashes add variance reflecting **dice, not skill**, which is the opposite
of what this experiment needs.

Evidence kept at `results/_archive/20260916_learnability_gust4.0_FAILED/`.

---

## Round 2 — survival fixed, learning still absent

At σ = 3.0:

| Metric | Round 1 | Round 2 |
|---|---|---|
| Crashes | 17/30 | **2/30** |
| Completion | 72.9% | **87.7%** |
| Explained variance | 0.218 → −0.387 | 0.218 → **0.628** |
| RMS error | 19.87 cm | 17.65 cm |

The critic now **learns** (explained variance rises to 0.63, reaching 0.90 on
seed 0). Survival is fixed. But policy return did not improve:

```
seed   rel. improvement   expl. var   completion
  0           +4.0%          0.615       96.5%
  1          -50.2%          0.496       66.5%
  2           +1.8%         0.774      100.3%
```

### The pattern in the per-seed curves

**Control effort rises monotonically in every seed** (0.054→0.191, 0.044→0.233,
0.057→0.123) while RMS error stays flat at ~16 cm. The policy is learning to
*act more* and getting nothing for it. In seed 1 that drift eventually crosses
the crash threshold (33%→67%, return 140→−6).

PPO itself is healthy: entropy stable, approx-KL ~0.01, clip fraction ~0.11.
This is not optimizer instability.

---

## The decisive diagnostic: action authority

Is there anything better than the PID that the action space can express?

| Controller | Return | RMS | Crashes |
|---|---|---|---|
| Pure PID (action = 0) | 495.8 | 18.00 cm | 0/5 |
| Constant offset 0.05 | 485.8 | 18.39 cm | 0/5 |
| Random (init noise) | 481.3 | 18.56 cm | 0/5 |
| **P-control on gap error, gain 3** | **604.8** | **12.92 cm** | **0/5** |
| P-control, gain 8 | −436.8 | 16.97 cm | 5/5 |
| Full +1 all axes | −114.8 | 38.86 cm | 0/5 |
| Full −1 all axes | −661.1 | 20.17 cm | 5/5 |

**A trivial hand-written proportional controller on the observed gap error beats
the PID by +109 return and 28% lower RMS error.**

This is the most important number in this document. It establishes:

1. **The task is solvable.** A better policy exists.
2. **It is expressible** in the 3-D offset action space.
3. **It is reachable** from the 13-D observation — gain 3 on `obs[0:3]`, nothing
   more.
4. **The agent is not finding it** in 30k steps.

### Why the agent struggles

The reward landscape is **flat near zero and sharply punishing further out**.
Small actions (0.05, random noise) are indistinguishable from doing nothing;
large ones are catastrophic. The policy initialises at ~zero (orthogonal init,
final-layer gain 0.01) and sits in the flat region, receiving almost no gradient
signal. Gain 3 works and gain 8 crashes 5/5 — the useful band is narrow.

This is an **exploration/budget problem**, not a broken task and not a broken
learner.

---

## What this means for the experiment

**The good news for the paper.** The P-control result gives us something the
project lacked: a **non-trivial reference controller** with a known score
(604.8 / 12.92 cm). Every rung can now be reported against a meaningful
baseline rather than against return alone. A reviewer asking "is your RL doing
anything a simple controller cannot?" has a direct answer.

**The risk it exposes.** If no rung beats 604.8, the honest conclusion is that
this task does not need a learned policy — and that must be reported, not
buried. That is kill-criterion K1 applied to the *task* rather than to k.

**Why a flat ladder would have been uninterpretable.** Had we skipped this check
and run the sweep, every rung would have sat at the PID baseline, the curve
would have come back flat, and we would have fired K1 and abandoned Claim 1 —
concluding "critic information does not matter" when the real cause was that no
policy at any rung had learned anything. That is the single most expensive
mistake available in this codebase, and it was two hours from happening.

---

## Status

**The observability-ladder experiment is BLOCKED and must not be run yet.**

It compares how well different critics help a policy learn. If no policy learns
anything beyond a trivial controller, every rung sits at the same baseline, the
ladder returns flat, and the flatness says nothing about critic information.
Running the sweep now would burn hours to produce an uninterpretable figure —
and would invite exactly the wrong conclusion (kill-criterion K1).

The blocker is characterised, not mysterious. See "Options" above; a decision on
which to take is the next step, and it is a scientific choice rather than a
debugging one.

---

## Round 3 — the evaluation protocol was the problem

Seed 1 scored 130 where seeds 0 and 2 scored 450–545, across two independent
runs. That looked like a bad initialisation falling into a poor basin.

**It was not a learning failure at all.** Scoring seed 1's evaluation block with
the *pure PID*:

| Training seed | Eval seeds | PID return | Crashes |
|---|---|---|---|
| 0 | 10000–10002 | 458.5 | 1/3 |
| **1** | **11000–11002** | **119.7** | **1/3** |
| 2 | 12000–12002 | 540.9 | 0/3 |

Episode by episode, seed 1's block:

```
seed 11000: return  -612.0   crashed=True   steps= 87
seed 11001: return   556.7   crashed=False  steps=722
seed 11002: return   414.5   crashed=False  steps=722
```

One episode crashes at step 87 and scores −612, dragging the 3-episode mean to
119.7. Seed 0 also crashes, but at step 599, so it costs far less. **The policy
was being graded on a harder exam**, and we nearly recorded that as a learning
failure.

### Return is the wrong headline metric

Crashes make the return distribution **bimodal**. Across the crash/no-crash
mixture the std is ~263, so a single early crash in a small block moves the mean
by ~350 — more than **three times** the entire P-control-vs-PID effect (+109).
Even n = 40 cannot resolve +109 on return.

RMS position error has no such problem: it measures tracking quality
continuously rather than being dominated by whether a crash happened.

Measured on 16 held-out episodes per controller:

| Controller | Return | RMS error | Crashes |
|---|---|---|---|
| PID | 512.6 ± 63.1 | 16.61 ± 1.78 cm | 1/16 |
| P-control | 604.5 ± 61.8 | 11.95 ± 1.08 cm | 1/16 |

| Metric | Cohen's d | n per group for 80% power |
|---|---|---|
| Return | 1.42 | 8 |
| **RMS position error** | **3.07** | **2** |

**RMS error has 4× the discriminating power of return.** The charter already
made physical metrics the headline (§S.11) for reviewer-legibility; this shows
they are also the statistically correct choice.

### Changes made

* `eval_episodes` 3 → **8** (periodic)
* `final_eval_episodes` 10 → **24** (reported)
* Pinned by `tests/test_reporting.py`, including a check that seed blocks cannot
  overflow into one another as counts grow.

### Does a bigger budget fix the learning gap? No.

A 120k-step run (4× budget) on seed 0: return 469 → 541, **plateauing after
~60k**. Above the PID (495.8), still well short of the P-controller (604.8).
More steps alone does not close the gap — this is an exploration problem, not a
budget problem.

---

## Round 4 — clean measurement, negative result

With the corrected protocol (8 periodic / 24 final episodes) and σ = 3.0, k1 at
60k steps × 3 seeds. Final held-out evaluation, 24 episodes per seed:

| | Return | RMS error | Crashes |
|---|---|---|---|
| seed 0 | 515.2 | 15.73 cm | 8% |
| seed 1 | 408.2 | 16.73 cm | 25% |
| seed 2 | 528.0 | 15.09 cm | 4% |
| **RL mean** | **483.8** | **15.85 cm** | |
| Pure PID | 512.6 | 16.61 cm | 6% |
| **P-control** | **604.5** | **11.95 cm** | 6% |

**The learned policy does not beat a one-parameter P-controller.** It is 121
return and 3.9 cm behind it, and slightly *below* the do-nothing PID on return.

The seed-1 story also resolves cleanly. Under the 3-episode protocol it scored
130 and looked like a bad basin; under 8 episodes it starts at 406 — the low
number was a measurement artifact. But its *decline* (406 → 207, 25% crash rate)
is real, and matches the control-effort drift seen in every seed.

This is now a well-characterised negative result rather than a mystery.

### Diagnosis: exploration, not budget, task, or learner

Ruled out by measurement:

* **Task unsolvable** — no: P-control achieves 11.95 cm.
* **No action authority** — no: full-scale actions swing return from −661 to +604.
* **Insufficient budget** — no: 120k steps plateaued at ~540 after 60k.
* **PPO instability** — no: entropy stable, approx-KL ~0.01, clip fraction ~0.11.
* **Broken evaluation** — fixed, and the result survived the fix.

What remains: the reward landscape is **flat where the policy starts and
punishing where it must go**. Small actions are indistinguishable from doing
nothing (PID 512.6, constant-0.05 offset 485.8, random noise 481.3 — all within
one standard deviation). The useful band is narrow: P-gain 3 completes 5/5,
gain 8 crashes 5/5. A zero-initialised policy has almost no gradient pointing
toward the good region, and drifting outward is punished before it is rewarded.

### Options, in order of cost

1. **Behaviour-cloning warm start** from the P-controller, then PPO. Puts the
   policy inside the useful band where gradients are informative. Cheapest, and
   standard practice — but it must be applied *identically at every rung*, or it
   becomes a confound in its own right.
2. **Reward shaping** to give gradient in the flat region (e.g. a denser
   tracking term). Changes the task, so it means a new task version and
   re-running the calibration.
3. **Residual formulation** — learn a correction *on top of* the P-controller
   rather than on top of the PID. Makes the trivial solution the origin, so any
   improvement is unambiguously attributable to learning.

**Option 3 is the most defensible for the paper.** It makes "does the critic
help?" a question about improvement over a competent controller, which is what a
reviewer would want anyway, and it removes the "your RL barely matches a P
controller" objection entirely.

---

## Lessons recorded

1. **When a fix changes the magnitude of a physical quantity, every number
   calibrated against the old magnitude is stale** — including ones carefully
   measured and correctly recorded at the time.
2. **When a learner fails, check whether the task is solvable before touching
   the learner.** A pure-PID probe costs minutes and caught this immediately.
3. **Always establish a non-trivial reference controller.** "Better than doing
   nothing" is a much weaker claim than it sounds, and without the P-control
   number we would not know whether 470 was good or catastrophic.
4. **Score the evaluation set with a known controller before blaming the
   learner.** Seed 1 looked like a bad basin for two entire runs. One PID pass
   over its eval block showed the block itself was hard. This costs minutes.
5. **A metric dominated by rare catastrophic events cannot resolve small
   effects.** Return is bimodal here; no feasible episode count fixes that.
   Choose a metric whose distribution is unimodal over the thing being
   measured.
