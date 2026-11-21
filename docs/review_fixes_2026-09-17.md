# External review, 17 September 2026 — verification and fixes

An external reviewer read the code and the handoff and concluded the negative
result was **not supported**. I verified every load-bearing claim by measurement
before changing anything. The reviewer was right on all but one detail.

---

## Verified: confirmed by measurement

### R1. The sweep never performed held-out final evaluation — CONFIRMED

`sweep.py::_run_one()` defined `final_eval_return` as the mean of the last three
**periodic** evaluations. `Trainer.evaluate(final=True)` existed, was tested, and
was called only from `scripts/check_learning.py`.

Three things wrong at once:

1. It averaged **three different policies** (three checkpoints), so the reported
   number described no model that ever existed.
2. Those episodes are the **periodic** block — the ones used to monitor run
   health. Reporting them is selecting on the monitoring data.
3. The correct path existed and the sweep didn't call it.

**This is the fourth instance in this project of a correct fix that never
reached the running code** (padded critic input; `metrics.py`; the
`eval_episodes` override; this). The pattern is mine, not coincidence.

*Fixed:* `_run_one` calls `evaluate(final=True)`; curve diagnostics renamed to
`curve_*` so they cannot be mistaken for results; per-episode records saved to
`final_eval_episodes.json` for later paired tests.

### R2. Per-run final evaluation seeds — CONFIRMED, and backwards

Each training seed had its own final episodes, adding episode difficulty to what
we report as between-run variance. Already measured: training seed 1 scored 130
where others scored 450–545, because its block contained an episode crashing at
step 87 (−612). **The pure PID scores 119.7 on that same block.**

*Fixed:* common random numbers. Every policy, rung and baseline is scored on the
**same** episodes (`FINAL_EVAL_SEED_BASE = 900_000`). Periodic blocks stay
per-run, since they judge run health independently and are never reported.

Two variances must now be reported separately: across **training seeds** (the
primary experimental unit) and across **evaluation episodes** (deployment
variability). They must not be pooled.

### R3. The k=1 rung duplicates the actor's observation — CONFIRMED, with nuance

For a follower, the most-coupled teammate is its predecessor, and the actor
already observes the gap to and relative velocity of that predecessor:

```
e_i    = p_i - (p_{i-1} - Δ)     →  p_{i-1} - p_i = Δ - e_i
ė_i    = v_i - v_{i-1}           →  v_{i-1} - v_i = -ė_i
```

Numerically confirmed: the k=1 teammate's relative velocity is **bit-identical**
to the negated `obs[6:9]` for all three followers.

But a rank test (dimensions added beyond the actor's own observation) shows the
problem is not uniform:

| rung | agent 0 | agent 1 | agent 2 | agent 3 |
|---|---|---|---|---|
| k0 | 0 | 0 | 0 | 0 |
| **k1** | 6 | **0** | 6 | 6 |
| k2 | 12 | 6 | 12 | 12 |
| k3 | 18 | 12 | 18 | 18 |

Agent 1 gains **exactly zero** at k=1 — its predecessor is the leader, whose
motion the actor already tracks analytically. Agents 2 and 3 *do* gain, because
their predecessors are gust-perturbed, so absolute state carries what relative
state does not.

So the ladder is not uniformly broken; it is **inconsistent across agents**,
which for a controlled experiment is arguably worse: `k` denotes a different
amount of added information depending on which drone you are.

*Fixed:* added `information_added_by_rung()` so this is measurable, plus a test
that asserts the current degenerate state and **must be rewritten when the
ladder is redefined**. The redefinition itself is a scientific decision, not a
code fix — see "What to run", below.

### R4. The PID never receives target velocity — CONFIRMED

`computeControl()` is called with `target_pos` only. Desired velocity defaults
to zero while both leader and followers track moving references, so the
controller carries avoidable lag.

The reviewer's algebra also holds: the P-controller's effect reduces to roughly
`−1.6e` versus `−e`, i.e. a ~60% increase in outer-loop position feedback.

**This reframes my earlier conclusion.** "A trivial controller beats RL by 109"
is evidence the **baseline PID is under-specified**, not that RL has an
exploration problem. My recommendation to build a residual on top of the
P-controller would have baked an arbitrary gain retuning into the design.

*Not fixed —* completing and tuning the classical controller changes the task
and needs a decision first.

### R5. Episodes ran 722 steps, not 720 — CONFIRMED (mechanism differs)

Measured off-by-2 at every frequency tested. **Two** compounding causes:

* Upstream `BaseAviary.step()` increments `step_counter` *after* computing
  obs/reward/terminated/truncated, so `_sim_time_sec` returned the time **before**
  the step being described.
* The truncation test used strict `>`, so the boundary step didn't trigger it.

*Correction to the review:* the reviewer said simulation time is "one control
step behind" and the fix is `>=`. Both parts are individually right but neither
alone is sufficient — `>=` alone still gave 721. The fix needs both: post-step
time **and** `>=`.

*Fixed:* exact at 720/240/300/48/180 steps across five configurations, pinned by
parametrised tests.

### R8. Gust magnitude mislabelled — CONFIRMED

`gust_std_ms = 3.0` is the per-**component** standard deviation. Measured vector
speed: **5.32 m/s RMS** (≈ √3 × 3.0), not 3.0. My "desk fan at close range"
gloss described the wrong quantity.

Also confirmed: `EpisodeRecorder` took `np.linalg.norm()` of the whole (N,3)
array — a Frobenius norm growing like √N. Measured 5.689 where the true mean
per-drone speed was 2.486.

*Fixed:* recorder logs mean per-drone speed; the comment now states the vector
speed, since that is what an anemometer would read. **The calibration itself is
unaffected** — σ=3.0 was chosen by measuring PID crash rate and seed variance,
not by reasoning from the label.

---

## Partially disputed

### R7. "PPO is healthy" — the reviewer is right

I reported entropy/KL/clip-fraction as healthy, which was accurate, but let it
stand as shorthand for "the learner is fine". Late explained variance averages
−0.02 across seeds, which contradicts that. KL and clip fraction show only that
updates aren't exploding. **I conflated optimizer stability with learning.**

Their further points stand and are not yet addressed: inputs unnormalised
despite differing physical scales; logged entropy is the *unsquashed* Gaussian's,
so it cannot diagnose tanh saturation; no gradient norms, action saturation or
action/error correlation recorded; checkpoints omit value-normaliser and
optimiser state.

### R6. Crash shaping monotonicity — right in principle, not currently exploited

The terminal cost forfeits `50 + remaining_steps`, and future tracking rewards
can be strongly negative, so monotonicity is not *guaranteed*.

Probed with a deliberately bad-but-surviving policy versus a deliberate dive:

| policy | return |
|---|---|
| bad but surviving | +324.2 |
| deliberate dive | −653.4 |

Shaping holds here by a wide margin, but one probe is not a proof. The reviewer's
fixed-horizon absorbing-state formulation is the sound fix.

### One place the review understates the problem

They note the learnability diagnostic *did* use held-out final evaluation (true),
implying it is sound on that axis. But it was still contaminated by the per-run
eval blocks criticised in R2. **The negative result is weaker than even the
reviewer credits.**

---

## Fixed in this pass

| # | Defect | Status |
|---|---|---|
| R1 | Sweep reported periodic, not held-out, evaluation | **Fixed** |
| R2 | Per-run eval seeds confounded policy with episode difficulty | **Fixed** (common random numbers) |
| R5 | Episodes ran 722 steps; completion >100% | **Fixed** (exact at 5 configs) |
| R8a | Gust speed logged as Frobenius norm across drones | **Fixed** |
| R8b | σ=3.0 mislabelled as a 3 m/s gust | **Fixed** (5.32 m/s RMS) |
| R3 | Ladder degeneracy | **Made measurable**; redefinition pending decision |
| R4 | PID missing target velocity | **Not fixed** — needs a decision |
| R6 | Crash monotonicity not guaranteed | **Not fixed** — needs a decision |
| R7 | Learner instrumentation | **Not fixed** — needs a decision |

Test suite: **95 passing** (was 84).

All results produced before these fixes are marked `invalid`.

---

## Lesson

Four times now a fix has been correct and unreachable. The common thread: I test
the layer I just edited, not the layer that runs in the experiment. The tests
added here assert on **sweep output** and on **episode counts from the real
environment**, not on the APIs I happened to touch.

---

# Part 2 — the three repairs, in order

## Step 1: the observability ladder redefined

**Measured first.** Dimensions each rung adds beyond the actor's own
observation (rank of the critic input after projecting out the actor's):

| rung | agent 0 | agent 1 | agent 2 | agent 3 |
|---|---|---|---|---|
| k1 *(before)* | 6 | **0** | 6 | 6 |
| k1 *(after)* | 6 | **6** | 6 | 6 |

Agent 1 gained nothing because its predecessor is the leader, whose motion the
actor already tracks analytically.

**Fix:** exclude the chain predecessor when counting teammate slots. Verified
that this **loses no information** — giving a follower all teammates adds 12
dimensions; all-except-predecessor also adds 12. The predecessor block is pure
redundancy.

**Second defect found while verifying.** The top rung was `k = N-1`, but a
follower has only `N-2` informative teammates, so `k3` added nothing beyond
`k2` — a duplicate rung that would have consumed a fifth of the sweep budget
measuring the same condition twice, and could have manufactured a fake
"interior optimum". Ladder capped at `N-2`: **k0, k1, k2, k2+P** for N=4.

**Third defect.** The *unpadded* critic builder still fed the privileged rung
`wind_vec` (a single global mean) while the *padded* one — the path MAPPO
actually uses — had been corrected to per-drone `air_velocity`. Training was
right; diagnostics read a stale value. Two builders of the same vector silently
disagreeing is precisely the drift pattern this project keeps hitting.

Capacity control re-verified: all rungs 28-D, 84,545 params (N=4).

> **Team size.** N=4 now yields only three information levels (0/6/12), i.e. one
> interior point — thin for a claim about an *interior* optimum. N=6 gives
> k0–k4 all distinct (0/6/12/18/24). **If Claim 1 is to be tested seriously,
> N=6 is the defensible size.** This is a decision, not a fix.

## Step 2: the classical controller completed — and it overturns the negative result

`computeControl()` was called with `target_pos` only, so desired velocity
defaulted to **zero** while the leader tracked a moving circle and every
follower tracked a moving predecessor.

Passing `target_vel` (already available as `_reference_velocities()`):

| controller | return | RMS | crashes |
|---|---|---|---|
| PID, no feedforward *(old baseline)* | 512.6 | 16.61 cm | 1/16 |
| P-control on top *(old "winner")* | 604.5 | 11.95 cm | 1/16 |
| **PID + feedforward** | **614.7** | **12.16 cm** | **0/8** |
| P-control on top of feedforward | −561.8 | 11.60 cm | **8/8** |

Gain sweep on the corrected controller:

| gain | 0.00 | 0.10 | 0.25 | 0.50 | 1.00 | 2.00 | 3.00 |
|---|---|---|---|---|---|---|---|
| return | **602.5** | 550.2 | 455.1 | 286.1 | 62.7 | −471.3 | −559.1 |
| crashes | **0/6** | 1/6 | 1/6 | 2/6 | 5/6 | 6/6 | 6/6 |

**The best P gain is zero.** The P-controller's entire advantage was
substituting proportional gain for missing velocity feedback.

**This overturns the headline conclusion.** "A trivial controller beats RL,
therefore RL has an exploration problem" was a comparison against a
deliberately handicapped baseline. The real bar is the feedforward PID, and it
is considerably higher.

### The calibration was wrong again — same mistake, different layer

Re-measuring on 24 common episodes revealed σ=3.0 crashes the PID **6/24**. All
six crashes sit in episodes 8–23; the original calibration used 12 episodes and
saw none.

| σ | return | seed std | crashes | RMS |
|---|---|---|---|---|
| 1.5 | 704.2 | 3.2 | 0/24 | 5.98 cm |
| 2.0 | 688.0 | 7.7 | 0/24 | 7.32 cm |
| **2.5** | **656.5** | **17.7** | **0/24** | **9.65 cm** |
| 3.0 | 465.2 | 295.6 | 6/24 | 13.26 cm |

σ=3.0's apparent seed std of 295.6 is **crash variance — dice, not skill**.
Recalibrated to **σ = 2.5** (vector speed ≈ 4.3 m/s).

This is the *same small-sample error* already diagnosed in the evaluation
protocol, repeated in the calibration. Calibration is a measurement and needs
the same sample-size discipline.

## Step 3: the learner instrumented

Added and recorded per update: `actor_grad_norm`, `critic_grad_norm` (both
**pre**-clipping — post-clipping they are capped by construction and say
nothing), `action_saturation`, `action_abs_mean`, `action_std`.

`action_abs_mean` matters most: it rose monotonically in every v4 seed while
tracking error stayed flat — the "acting more, achieving nothing" signature.
The logged `entropy` is the **unsquashed** Gaussian's, so it cannot fall when
tanh saturates and cannot detect the failure it appears to watch.

**Representation check.** Supervised cloning of a P mapping onto the real actor
architecture reaches **R² > 0.998** at two different gains. The network can
express a competent controller, so a learning failure lies in the search, the
reward, or the critic — not in capacity.

---

## Status

**97 tests passing** (was 84). Every prior result is now marked `invalid`.

Nothing has been retrained. The learnability question is **reopened**: the
previous answer was measured against a handicapped baseline, a mis-calibrated
disturbance, and a degenerate ladder. All three are fixed; the experiment has
not been re-run.

---

# Part 3 — Gates 0 and 1

## Gate 0: remaining correctness items

Audited each item rather than assuming. Six were already fixed; **three were
not**, and the reviewer was right to ask.

| Item | Status |
|---|---|
| Episodes end at exactly 720 steps | OK |
| Post-physics reference time | OK |
| Sweep calls `evaluate(final=True)` | OK |
| Common final evaluation seeds | OK |
| Episode-level results saved | OK |
| Simulator commit + packages pinned | OK |
| **Checkpoint saves value normalizer** | **was MISSING — fixed** |
| **Wrong-rung checkpoint rejected** | **guard never fired — fixed** |
| **Failure cannot reward early termination** | **hole was real — fixed** |

### Checkpoints were incomplete and the rung guard was dead

The critic trains against normalised returns and its output is denormalised
with running mean/variance — learned state that was not saved. A restored
critic was silently mis-scaled. Optimizer state was missing too.

The rung guard compared `critic_obs_dim`, which since the padding fix is
**identical at every rung by design**. It could never fire: a k0 checkpoint
loaded into a k2 learner succeeded silently. Now the rung identity is checked.

### The monotonicity hole was real, and measured

The reviewer argued that forfeiting `50 + remaining_steps` does not *guarantee*
crashing is worse, because avoided future tracking costs can exceed it. Measured
directly:

```
worst per-step reward while ALIVE : -1.07
survival bonus forfeited by crashing:  1.00
```

A persistently-bad agent could **profit by terminating** — it avoided 1.07/step
at a price of 1.00/step. That is the v1 "learn to die" failure reachable through
a different door.

Fixed with a fixed-horizon absorbing cost (`FAILURE_STEP_COST = 3.0`, strictly
worse than the worst survivable rate). Verified: dive −2102.6 vs worst
surviving +177.2.

## Gate 1: headroom above the feedforward PID

100 common episodes, σ=2.5, identical seeds for every controller. RMS is
reported **conditional on completion**, because RMS is censored by crashes — a
controller that dies early accrues less error and looks accurate. (The earlier
P-control row showing 11.60 cm while crashing 8/8 is exactly that artefact.)

| controller | return | RMS \| completed | crashes | paired Δreturn |
|---|---|---|---|---|
| **feedforward PID (zero residual)** | 593.3 ± 320.5 | 9.76 cm | 4/100 | — |
| P gain 0.05 | 552.8 | 9.73 cm | 7/100 | −40.5 ± 74.6 |
| P gain 0.1 | 555.1 | 9.73 cm | 6/100 | −38.2 ± 57.6 |
| P gain 0.2 | 529.7 | 9.77 cm | 8/100 | −63.6 ± 64.1 |
| P gain 0.4 | 499.4 | 9.85 cm | 11/100 | −93.9 ± 71.4 |
| PD (0.05, 0.05) | 577.5 | 9.59 cm | 5/100 | −15.9 ± 33.6 |
| PD (0.1, 0.1) | 544.3 | 9.48 cm | 6/100 | −49.1 ± 75.9 |
| PD (0.2, 0.2) | 538.9 | 9.20 cm | 8/100 | −54.4 ± 59.6 |
| **PD (0.1, 0.3)** | **595.6** | **8.97 cm** | 5/100 | **+2.3 ± 37.6** |

### What this says

**No residual controller beats the feedforward PID on return.** Every paired
difference either favours the PID or is indistinguishable from zero. The best
candidate, PD(0.1, 0.3), is **+2.3 ± 37.6** — a confidence interval straddling
zero.

There is a *hint* of headroom in tracking: RMS falls monotonically with added
derivative gain, 9.76 → 8.97 cm (~8%). But it comes with no return benefit and
a slight crash increase, so it is not a **safe** improvement.

**The honest reading: remaining headroom above a properly-tuned classical
controller is small and not clearly safe.** The reviewer's warning applies
directly — if RL fails to beat this baseline, that would *not* establish broken
learning, because there may be very little left to win.

### The calibration was wrong a third time

σ=2.5 gives **4/100 PID crashes**, not the 0/24 I reported. All four sit beyond
episode 24. This is the third time a calibration has been set on too small a
sample:

| round | sample | verdict | actual |
|---|---|---|---|
| 1 | 12 episodes | σ=4.0 "0% crashes" | 40% crashes |
| 2 | 24 episodes | σ=3.0 "0/24" | 6/24 |
| 3 | 24 episodes | σ=2.5 "0/24" | **4/100** |

By the rule of three, 0/24 only bounds the crash rate below ~12.5%. The
reviewer's prescription — ≥100 episodes, and a **predefined acceptable failure
rate** rather than lowering σ until a small sample shows zero — is the right
standard. A 4% baseline failure rate is defensible if declared in advance;
what is not defensible is discovering it after the fact three times running.
