# HANDOFF — read this first

**Written for: whoever picks this project up next.**
**Last worked on: 17 September 2026. Written: 19 November 2026.**

Environment setup is a separate file: **`SETUP_NEW_MACHINE.md`**. Do that first.

---

## In one paragraph

The paper asks whether a centralized critic has an **optimal amount of
information** — not maximal, interior. The apparatus is a chain of quadrotors
holding vertical formation under aerodynamic downwash and stochastic gusts; the
independent variable `k` is how many teammates the critic may see. **No valid
experimental result exists yet.** Three audit rounds found defects that would
each have produced a plausible but wrong answer, and all of them are now fixed
(100 tests pass). Work stopped at a **genuine scientific fork**, not a bug: a
properly-tuned classical controller turns out to be close to optimal on this
task, so there may be too little headroom for any learned policy to demonstrate
anything. **That decision is open and needs a human.** See "The open decision".

---

## State of the code

| | |
|---|---|
| Task version | `v4-correctness` |
| Tests | **100 passing** |
| Observability ladder | N=4: `k0, k1, k2, k2+P` · N=6: `k0..k4, k4+P` |
| Gust σ | 2.5 per-component (≈4.3 m/s vector speed) |
| Evaluation | 8 periodic / 24 final episodes, **common seeds from 900000** |
| Critic capacity | identical at every rung — 28-D input, 84,545 params (N=4) |
| Valid results | **none** — every run in `results/_archive/` is marked invalid |

Nothing is currently training. Nothing is half-finished in the code.

---

## The open decision — this is where we stopped

**Gate 1 asked: is there safe room above a well-tuned classical controller?**
100 common episodes, σ=2.5, RMS reported *conditional on completion*:

| controller | return | RMS \| completed | crashes | paired Δreturn |
|---|---|---|---|---|
| **feedforward PID (zero residual)** | 593.3 ± 320.5 | 9.76 cm | 4/100 | — |
| P gain 0.1 | 555.1 | 9.73 cm | 6/100 | −38.2 ± 57.6 |
| P gain 0.4 | 499.4 | 9.85 cm | 11/100 | −93.9 ± 71.4 |
| PD (0.2, 0.2) | 538.9 | 9.20 cm | 8/100 | −54.4 ± 59.6 |
| **PD (0.1, 0.3)** | **595.6** | **8.97 cm** | 5/100 | **+2.3 ± 37.6** |

**Nothing beats the PID.** The best candidate's confidence interval straddles
zero. There is a hint of tracking headroom (9.76 → 8.97 cm, ~8%) but it buys no
return and costs a slightly higher crash rate, so it is not a *safe*
improvement.

This matters because of what it implies about a future negative result: **if RL
fails to beat this baseline, that would not prove the learner is broken.** There
may simply be nothing left to win. Running the ladder sweep in that state
produces an uninterpretable figure either way.

### The three options on the table

1. **Declare an acceptable baseline failure rate** (e.g. ≤5%), accept σ=2.5's
   4/100, and run the N=4 learnability gate knowing headroom is thin. Cheap,
   but a null result stays ambiguous.
2. **Make the task harder so headroom exists** — larger σ, tighter spacing, or
   stronger downwash. Changes the task (new version tag, fresh calibration on
   ≥100 episodes), but makes the experiment *capable* of showing something.
   ← **recommended**
3. **Report the finding as the result**: a well-tuned classical controller is
   near-optimal here, so this task cannot support Claim 1. Honest, publishable
   as a negative/benchmark contribution, and far cheaper than discovering it
   after a full sweep.

**Why option 2.** The incremental downwash force falls off steeply with
separation — roughly 0.076 N for the already-observed predecessor, then 0.019,
0.008, 0.005, 0.003 N for each further neighbour at 1 m spacing. At 1 m the
distant teammates are physically almost irrelevant, so teammate information
cannot matter much *whatever* the learner does. Tighter spacing or stronger
coupling is what would give `k` something to bite on. But this is a question
about what the paper is testing, not an implementation detail, so it is
decided deliberately rather than by default.

### Agreed sequence once the decision is made

```
controller calibration  ->  N=4 learnability gate  ->  N=6 pilot  ->  N=6 full study
```

N=4 first as a **correctness gate only** — it is a debugging experiment, not a
result. N=4 gives just three information levels (0/6/12 dimensions), i.e. a
single interior point, which is thin evidence for an *interior* optimum. N=6
gives k0..k4 (0/6/12/18/24) and three interior points, and is the defensible
size for the actual claim. Do not spend N=6 compute until N=4 shows the learner,
critic and evaluation protocol all work.

---

## What to run next

**Do not start with a training run.** Start by confirming the environment:

```bash
source .venv/bin/activate
python -m pytest tests/ -q          # expect: 100 passed
```

Then resolve the open decision above. If **option 1** is chosen and the N=4
gate is wanted immediately:

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python scripts/check_learning.py --steps 30000 --seeds 3 --rung k1
```

~30 min. But read "Pass criteria" below first — the script's own verdict is
necessary, not sufficient.

If **option 2** is chosen, the first task is re-calibrating the disturbance on
**≥100 episodes** with a **failure rate declared in advance**. Do not lower σ
until a small sample happens to show zero crashes — that mistake was made three
times (below).

### Pass criteria for the N=4 gate

Require all of these, not just rising return:

- critic explained variance **consistently positive** (it averaged −0.02 before)
- deterministic tracking error improves **in centimetres**
- crash rate does not increase
- policy beats zero-residual **on paired episodes**, same seeds
- action saturation and gradient norms stay healthy
- learned actions correlate sensibly with position/velocity error
- test **k0 and k2**, not only k1

Return alone cannot settle it: the distribution is **bimodal** (crash vs
no-crash, σ≈263), so even n=40 cannot resolve a +109 effect. RMS error has
Cohen's d = 3.07 versus 1.42 for return — about 4× the discriminating power.

---

## What was wrong before, so it isn't repeated

Four months of work produced no valid result because of defects that **did not
crash anything**. Grouped by the lesson, since the pattern repeats:

### A. Correct fixes that never reached the running code — four times

1. Padded critic input existed; MAPPO never called it (real learners were 1,344
   params apart while tests passed).
2. `metrics.py` was written, unit-tested, imported nowhere.
3. `eval_episodes` default was raised 3→8; three scripts still hardcoded 3, so a
   "corrected" run produced byte-identical numbers.
4. `Trainer.evaluate(final=True)` was written and tested — and the **sweep**,
   the code that produces the paper's numbers, never called it. It reported the
   mean of three *periodic* evaluations of three *different* checkpoints.

**Lesson: test the layer that runs in the experiment, not the layer you edited.**

### B. Calibrations set on too small a sample — three times

| round | sample | claimed | actual |
|---|---|---|---|
| 1 | 12 episodes | σ=4.0, 0% crashes | **40%** — task was unsurvivable |
| 2 | 24 episodes | σ=3.0, 0/24 | **6/24** |
| 3 | 24 episodes | σ=2.5, 0/24 | **4/100** |

Round 1 was the expensive one: a learnability run "failed" at 72.9% completion
and was read as the agent not learning. The **pure PID** completes 73% at that
σ. Nothing could have learned it. Cause: σ=4.0 had been calibrated while wind
reached the drones on 1 of 5 physics substeps (~20% strength); a later fix
applied it on all 5 and nobody re-measured.

**Lesson: ≥100 episodes, rule of three (0 in n only bounds the rate below 3/n),
and a failure rate declared in advance.**

### C. Comparisons against a handicapped baseline

The PID was called with `target_pos` only, so its desired velocity defaulted to
**zero** while the leader tracked a moving circle and every follower tracked a
moving predecessor. A one-parameter P-controller "beat RL by +109 return / 28%
lower RMS", and that was written up as evidence of an exploration problem.

Passing `target_vel` raised the PID to 614.7 / 12.16 cm — matching the
P-controller — and the best P gain became **zero**. The whole finding was the
missing feedforward term.

**Lesson: complete and tune the classical baseline before diagnosing the learner.**

### D. The ladder did not vary what it claimed to

In a vertical chain the most-coupled teammate **is** the predecessor, whose gap
and relative velocity the actor already observes. Measured dimensions added
beyond the actor's own observation:

```
rung   agent0  agent1  agent2  agent3
k1        6      0        6       6     <- agent 1 gained NOTHING
```

`k` meant a different amount of information depending on which drone you were.
Also `k3 ≡ k2` for followers — a duplicate rung that would have consumed a
fifth of the sweep budget and could have manufactured a fake interior optimum.
Fixed by excluding the predecessor (verified to lose no information) and capping
the ladder at N−2.

### E. Metrics that could not measure what they were used for

- Return is **bimodal**; one early crash in a 3-episode block moved the mean by
  ~350 — three times the effect being measured. One seed was nearly
  misdiagnosed as "not learning" when its evaluation block simply contained an
  early crash.
- Per-run evaluation seeds confounded policy quality with episode difficulty.
  Now **common random numbers**: every policy, rung and baseline is scored on
  the same episodes from 900000.
- RMS is **censored by crashes** — a controller that dies early accrues less
  error and looks accurate. One row showed 11.60 cm while crashing 8/8. Always
  report RMS conditional on completion, next to crash probability.
- Logged `entropy` is the **unsquashed** Gaussian's, so it cannot fall when tanh
  saturates and cannot detect the failure it appears to watch.
- Episodes ran **722 steps instead of 720**, so "completion" exceeded 100%.

### F. Things that were verified and are fine

Worth knowing so they are not re-investigated:

- The **task is solvable** and the **actor can express** a competent policy:
  supervised cloning of a P/PD mapping reaches R² > 0.998.
- **Action authority exists** — full-scale actions span −661 to +604 return.
- **PPO updates are stable** — approx-KL ~0.01, clip fraction ~0.11. (This is
  *not* the same as "the learner works": explained variance averaged −0.02.)
- **Capacity control is exact** — every rung, verified on real learners.
- **Crash shaping is now monotone** — a fixed-horizon absorbing cost; a
  measured hole (−1.07/step while alive vs 1.00/step forfeited) is closed.
- **Downwash premise holds** — ~12× force contrast across the useful range, and
  physics-ranked neighbours genuinely differ from Euclidean-ranked.

---

## Known open items (not blockers, but real)

- **Aerodynamic coupling and gust correlation are confounded** at
  `gust_spatial_corr = 0.7`. Teammate state is informative because drones share
  gusts, even with downwash disabled. The factorial (downwash off/on × corr
  0/0.7) in `PROJECT_CHARTER.md` §S.10 has **not** been run, so no benefit from
  centralized state can yet be attributed to aerodynamics.
- **The downwash model is a position-only proxy.** It ignores the source
  drone's thrust and applies force in the receiving drone's link frame. The
  paper must call it an empirical position-dependent proxy, not measured
  coupling. `coupling_matrix()` is an analytical quantity, not an applied force.
- **Observation inputs are not normalized** despite differing physical scales.
- **Statistical unit**: training seed is the experimental unit; episodes are
  nested repeated measures. 24 episodes from one policy are not 24 replications.
- **Power**: n=8 seeds gives only ~46% power at d=1.0. 12–20 needed.
- **Prior art**: `PROJECT_CHARTER.md` §S.8 — Lyu et al. (JAIR 2023), MADDPG-K,
  Qu et al. (2020). The claim "nobody has tested this" does **not** survive;
  §S.9 has the corrected, defensible framing.

---

## Document map

| File | What it holds |
|---|---|
| `SETUP_NEW_MACHINE.md` | environment, versions, failure modes |
| `PROJECT_CHARTER.md` | the plan. §S.* = status, newest first |
| `docs/review_fixes_2026-09-17.md` | **most recent audit** — Gates 0 and 1 |
| `docs/learnability_v4.md` | why training was halted |
| `docs/audit_2026-09-16.md` | the three earlier audit rounds |
| `docs/EXPERIMENT_DEFINITION.md` | task + RL problem, with a glossary |
| `results/_archive/README.md` | why each old result is invalid |
| `commands/` | terminal commands per stage |

Nothing in `results/` is quotable. The archive is kept as evidence of *why*
decisions were made — deleting it invites repeating the mistakes.
