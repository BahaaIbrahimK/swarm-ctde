# Project Charter — How Much Should a Centralized Critic See?

**Working title:** *The Observability Optimum: Rethinking How Much Information a Centralized Critic Should Receive in Multi-Robot Reinforcement Learning*

**Authors:** Bahaa Ibrahim, Eman Mohamed — KFUPM
**Status:** v3.0 — Phases 0–3 built; **all pilot results invalidated by a code audit (16 Sep 2026) and pending re-measurement.**
**Written:** 20 August 2026 · **Last updated:** 16 September 2026

> **The research argument in §1–§3 is unchanged and has held up.** What follows
> in §S is the implementation record: what was built, what was measured, and
> which design decisions were revised along the way. Read §S if you want the
> current state; read §1–§3 if you want the idea.

---

## §S. STATUS — what has actually happened

*Added 1 Sep 2026. This section did not exist in v1.0.*

### S.1 Phases

| Phase | Status | Outcome |
|---|---|---|
| **0 — Foundations** | ✅ done | Config system, seeding, 40 tests |
| **1 — Physics premise** | ✅ done | **Downwash confirmed:** 12× coupling contrast, 0.60–2.50 m |
| **2 — Apparatus** | ✅ done | Ladder built; both confound controls verified |
| **3 — Pilot** ⚠️ | ⚠️ **re-run required** | Interior optimum found, but **invalidated** — see §S.7 |
| 4 — Full sweep | ⬜ blocked | Blocked on re-running Phase 3 under v4 |
| 5–9 | ⬜ | Per §5 |

### S.2 The headline result (EXP03, v3-chain-gust, 1 Sep 2026) — ⚠️ INVALIDATED

> **Do not cite these numbers.** A code audit on 16 Sep 2026 found five defects
> that invalidate inference — see §S.7. The result below is retained only to
> record what was measured and why it must be repeated.

Evaluated at the 300k horizon that Phase 3 Stage 1 independently established as
converged:

| Rung | Mean | Std | 95% CI |
|---|---|---|---|
| `k0` (decentralized) | 653.56 | 32.25 | [613.5, 693.6] |
| **`k1` (interior)** | **677.20** | **4.62** | **[671.5, 682.9]** |
| `k3` (maximal) | 621.54 | 63.41 | [542.8, 700.3] |

```
k1 − k0 = +23.64   p = 0.177   d = +1.03
k1 − k3 = +55.66   p = 0.121   d = +1.24
```

**The interior optimum predicted by Claim 1 appeared.** Both effect sizes exceed
d = 1.0 (large). `k1`'s seed spread of 4.62 is notably tight — the middle rung is
not only better on average but more *reliable*.

**Not yet established.** Neither comparison reaches p < 0.05 at n = 5, and
kill-criterion K3 still fires: `k3`'s seed std (63.4) exceeds the between-rung
effect (55.7). This is exactly what a pilot should deliver — evidence that the
full sweep is worth running, not a finding.

### S.7 Code audit, 16 September 2026 — results invalidated

An external audit found five defects, **all confirmed by direct measurement**:

| # | Defect | Effect |
|---|---|---|
| 1 | GAE read `dones[t+1]` instead of `dones[t]` | Mid-rollout terminals leaked reward and value across episode boundaries |
| 2 | GAE recursion ran through environment resets | Truncation was never stored; unrelated episodes were blended |
| 3 | **Crash penalty was dead code** | `BaseAviary` calls `_computeReward()` *before* `_computeTerminated()`, so the flag was always `False`. Measured: crashing step returned +0.88, identical to the step before, with a configured penalty of 50.0 |
| 4 | Collisions were never detected | `crash_rate` did not measure collisions |
| 5 | Privileged critic got only the steady mean wind | The `k3+P` rung was missing the stochastic gust — the dominant disturbance term |
| 6 | Critic capacity was not exactly equal | Projection params grew 832 → 2,176; confounds the ascending side of the optimum |

**Why every existing test missed them:** each test exercised its component in a
way that bypassed the failure. The GAE test placed the terminal on the last
buffer element — the one index where the off-by-one is invisible. The crash test
set `env._crashed = True` by hand, testing the reward *formula* while bypassing
the call *order* that broke it.

**The lesson, recorded so it is not repeated:** confident comments and passing
tests were written around assumptions never verified end-to-end. The new suite
(`tests/test_correctness.py`, 11 tests, all failing pre-fix) follows one rule —
**test through the public path the experiment actually uses.**

All defects are fixed and the task version is now **`v4-correctness`**. Full
detail: [`docs/audit_2026-09-16.md`](docs/audit_2026-09-16.md).

**Three rounds, not one.** The audit ran three times because the first two
rounds each missed a different *class* of defect:

| Round | Class | Example | Caught by |
|---|---|---|---|
| 1 | Wrong **physics / learning** | GAE off by one; crash penalty was dead code | measurement against the public path |
| 2 | Correct fixes **never wired in** | padded critic input existed; MAPPO never called it — real learners were 1,344 params apart while tests passed | instantiating the *real* objects (`tests/test_wiring.py`) |
| 3 | Correct code, **corrupted report** | `aggregate_by_rung` averaged across formation spacing — Claim 2's own independent variable | checking what reaches the *table*, not the run |

The progression moves steadily away from where code crashes and toward where it
merely lies. Round 3's R1 is the sharpest case: it would have produced a
complete, plausible, publishable table answering a **different question than the
one we asked**, with no error and nothing to notice.

Test suite: **81 passing** (was 69 after round two, 55 before the audit).

### S.3b Learnability under v4 — the task was unsurvivable (16 Sep 2026)

Before any sweep, we checked whether the agent still learns after v4 changed
five things at once. **It did not** — and the cause was the *task*, not the
learner.

A pure PID with **zero RL action** at the configured `gust_std_ms = 4.0`
crashes 40% of the time and completes 73% of the episode, almost exactly
matching the trained policies' 72.9%. Nothing could have learned it.

`σ = 4.0` had been calibrated on 31 Aug as "best variance, still 0% crashes" —
while wind reached the drones on **1 of 5 physics substeps**. v4 fixed that
substep bug, making the real disturbance ~5× what the calibration described.
Recalibrated to **σ = 3.0** (seed std 53.58, 0% crashes, 31× the σ=0 baseline
variance). Crashes then fell 17/30 → 2/30 and explained variance rose to 0.63.

**The near-miss.** Had we skipped this check, every rung would have sat at the
PID baseline, the ladder would have come back flat, and we would have fired
kill-criterion K1 — concluding "critic information does not matter" when in
fact no policy at any rung had learned anything. That is the most expensive
mistake available in this codebase, and it was roughly two hours away.

### S.3c A real baseline now exists — and it raises the bar on the whole paper

Diagnosing the above produced something the project lacked: a **non-trivial
reference controller**, measured on held-out gust seeds at σ = 3.0.

| Controller | Return | RMS error |
|---|---|---|
| Pure PID (action = 0) | 495.8 | 18.00 cm |
| Random noise at init std | 481.3 | 18.56 cm |
| **P-control on gap error, gain 3** | **604.8** | **12.92 cm** |

A one-parameter proportional controller on the observed gap error beats the PID
by **+109 return and 28% lower tracking error**. This proves the task is
solvable, the solution is expressible in the action space, and it is reachable
from the observation — so when a learned policy fails to improve, the task is
not at fault.

It also **raises the bar the paper must clear**. Claim 1 is about which critic
learns best; but if *no rung* beats a controller with one tunable number, the
honest finding is that this task does not need a learned policy at all. That
must be reported, not buried. Every rung will therefore be reported against
this baseline, not merely against return — a reviewer asking "is your RL doing
anything a simple controller cannot?" gets a direct answer.

Implemented in [`src/ctde/baselines.py`](src/ctde/baselines.py), pinned by
`tests/test_baselines.py` (which also asserts the configured gust strength
leaves the PID surviving — the check whose absence cost the first run).

### S.3d BLOCKED — the learned policy does not beat a trivial controller

With the calibration fixed and the evaluation protocol corrected (3 → 8 periodic
episodes, 10 → 24 final, after measuring that 3 episodes cannot resolve a +109
effect), k1 at 60k steps × 3 seeds gives a clean, trustworthy measurement:

| | Return | RMS error |
|---|---|---|
| Learned policy (k1, 3 seeds) | 483.8 | 15.85 cm |
| Pure PID (action = 0) | 512.6 | 16.61 cm |
| **P-control, one parameter** | **604.5** | **11.95 cm** |

**RL does not beat the one-parameter baseline**, and sits slightly below the
do-nothing PID on return.

Ruled out by measurement: task unsolvable (no — P-control gets 11.95 cm), no
action authority (no — full-scale actions span −661 to +604), insufficient
budget (no — 120k steps plateaued after 60k), PPO instability (no — entropy,
KL and clip fraction all healthy), broken evaluation (fixed; result survived).

**Diagnosis: exploration.** The reward landscape is flat where the policy
initialises and punishing where it must go. Everything small scores the same
(PID 512.6, constant offset 485.8, random noise 481.3), and the useful band is
narrow (P-gain 3 completes 5/5; gain 8 crashes 5/5).

**The ladder experiment is blocked until this is resolved.** It measures how
much different critics help a policy learn; if no policy learns anything beyond
a trivial controller, every rung sits at the same baseline and a flat ladder
would be uninterpretable — while looking exactly like kill-criterion K1.

**Recommended resolution: a residual formulation** — learn a correction on top
of the P-controller rather than on top of the PID. That makes the trivial
solution the origin, so any improvement is unambiguously attributable to
learning, and it pre-empts the reviewer question "is your RL doing anything a
simple controller cannot?". The alternatives are a behaviour-cloning warm start
(cheapest, but must be identical at every rung or it becomes a confound) or
reward shaping (changes the task; new task version, new calibration).

Full diagnostic record:
[`docs/learnability_v4.md`](docs/learnability_v4.md).

---

### S.3 Task versions — three, because two were wrong

The task definition was revised twice. Results from different versions **must
never be pooled**; the version tag appears in every result path.

| Version | Status | Why it was abandoned |
|---|---|---|
| `v1-raw4d` | ❌ invalid | **Unlearnable.** Direct rotor control: episodes ended after 20–100 of 720 steps even after 300k steps. Per-step penalties also made crashing *cheaper* than flying badly — returns "improved" 59% while the policy was learning to die faster. |
| `v2-pidassist` | ❌ superseded | **(a)** Followers were measured against an *analytical leader trajectory*, so "decentralized execution" was not honest — a follower implicitly knew the leader's whole future path. **(b)** Steady wind and steady downwash are both cancelled by a PID: measured seed variance 0.00, leaving nothing for any critic to explain. **(c)** Wind force used the wrong PyBullet frame and application point. |
| `v3-chain-gust` | ❌ superseded | Correct in physics, wrong in learning machinery. The September audit found six defects that would have invalidated any result: GAE off by one step and running straight through episode resets; the crash penalty was dead code (measured +0.88 on the crashing step, identical to the step before); no collision detection at all; the privileged critic never received the gust it was supposed to be privileged about; and critic capacity was not exactly equal across rungs. |
| **`v4-correctness`** | ✅ **current** | Chain topology + stochastic gust field + corrected force application, plus: padded critic input so every rung has byte-identical capacity (34-D input, 84,929 parameters, verified on real learners); tanh-squashed Gaussian policy; value normalization; separate termination/truncation handling; collision detection; survival-scaled crash penalty; physical metrics in metres reported alongside return; held-out evaluation seeds. |

Full detail: [`docs/task_v2_notes.md`](docs/task_v2_notes.md),
[`docs/task_v3_notes.md`](docs/task_v3_notes.md),
[`docs/audit_2026-09-16.md`](docs/audit_2026-09-16.md).

### S.4 Design decisions revised since v1.0

These supersede statements in §4 below. **§4 has not been rewritten**, so where
the two disagree, this section is current.

**Control (§4.2 said "full 4-D per-rotor RPM. No PID crutch.")**
Now a three-layer stack: the RL policy commands a **3-D position offset**
(guidance), and a DSL PID runs position and attitude loops beneath it. Direct
rotor control proved unlearnable from scratch. This is also how real drone
stacks work, which *strengthens* Claim 4.

**Topology (not specified in v1.0)**
**Chain**: each follower observes its gap to the drone immediately above,
~1 m away — measurable by UWB or onboard vision. Not a trajectory formula. This
also produces error propagation down the chain (string dynamics), which is where
teammate information becomes genuinely valuable to a critic.

**Disturbance (§4.2 said "stochastic wind field")**
Now correctly stochastic: an Ornstein–Uhlenbeck gust field with a spatial
correlation parameter. v2's *steady* wind was cancelled entirely by the PID
(seed variance 0.00). Gusts restore it (std 21.7). Parameters are in physical
units (`gust_std_ms = 4.0`, a desk fan close up) so they can be checked against
reality — which matters for Claim 4.

**Platform: indoor, confirmed.** Bitcraze's own FAQ states Crazyflies are not
recommended outdoors. This bounds realistic wind magnitudes and makes UWB the
honest positioning choice.

### S.5 Open issues before Phase 4

1. **Late-training collapse — 4 of 15 runs.** The critic destabilises past ~350k
   steps (explained variance goes negative), which corrupts the policy gradient
   and drags performance down. Suspected cause: `lr_critic = 1e-3` too high.
   **Untested.** Mitigations: evaluate at the 300k converged horizon, and/or
   lower the critic learning rate.

2. **Statistical power.** n = 5 is insufficient; K3 fires. Phase 4 needs ≥8
   seeds, which at d ≈ 1.0–1.2 should reach significance.

3. **Spatial correlation is a weaker lever than hoped.** Seed std rises 3.92 →
   8.12 across corr 0 → 0.7, then saturates. Gust *strength* dominates. Relevant
   if a future experiment sweeps correlation as an independent variable.

4. **A limitation to disclose.** The downwash model is extrapolated below
   0.6875 m (its Gaussian width term goes negative — silently). Claim 2 spacings
   should stay above this, or the extrapolation must be stated.

### S.6 Reproducibility notes

- **Throughput:** ~46 steps/s/process at 4 workers with `OMP_NUM_THREADS=1`.
  **Thread capping is essential** — 6 unthrottled workers oversubscribed an
  8-core machine 2.3× and a sweep made no progress in 13 hours.
- Every result directory carries `MANIFEST.json` with git commit, task version,
  and status. Superseded runs are **marked, never deleted** — they are the
  evidence for why the design changed.

---

### S.8 External reviews, 16 September 2026 — prior art and the pivot question

Two independent reviews arrived: a **research/novelty** review and a **code
audit**. The code audit's findings are in §S.7. This section records the
research findings, which change the paper's framing.

#### The prior art is stronger than §1 claims

I verified each of these directly. **§1.1's claim that "nobody has ever tested
this" does not survive.**

| Work | What it already establishes |
|---|---|
| **Lyu et al., JAIR 2023** | *Theorem 1:* centralized and decentralized history critics have **identical expected gradients**. *Theorem 2:* `Var[ĝ_h] ≥ Var[ĝ_i]` — centralized critics have **provably higher** gradient variance. Abstract states plainly: *"critic centralization is not strictly beneficial, and using state values can be harmful."* |
| **MADDPG-K**, arXiv 2604.18190 (Apr 2026) | Restricts each critic to its `k` closest agents; concludes **optimal k is environment-dependent**. This is Claim 1, published. |
| **Qu et al. 2020**, PMLR 120 | κ-hop localized critics with **provable exponential decay** in neighbourhood radius. This is *theory predicting our interior optimum* — worse for us than mere overlap. |

**Consequence:** Claim 1 can remain an empirical phenomenon but **cannot be the
central novelty**. It must cite these as prior art, not as a gap.

#### Where novelty may survive

Narrower than the reviewer's "physics-aligned critics", and worth stating
precisely:

> Localized and k-nearest critics are established. What is unresolved is whether
> a critic's information set can be chosen from an **independently measured,
> directed, anisotropic physical-influence graph**, and whether that alignment
> predicts learning and hardware performance in continuous multi-robot control.

Our Phase 1 measurement supports exactly this distinction: a drone 0.6 m above
but 10 cm off-axis exerts **zero** force, while one 1.5 m above and aligned
exerts real force. Euclidean and physical neighbours genuinely differ here —
which Qu et al.'s *graph-distance* decay does not capture.

#### Do NOT pivot prematurely

The research reviewer recommends reframing around the coupling graph. The
theory reviewer cautions against deciding too early, and **I agree with the
caution**:

- Identical physics/distance rankings would **not** kill the observability
  question. It would mean only *"this geometry cannot compare two selection
  rules."*
- The charter asks *"how much teammate information should the critic receive?"*
  — not *"is physics-ranking better than distance-ranking?"*

**Decision: defer the framing choice until §S.10's diagnostics are in.**

#### A correction to the proposed method

Both reviewers proposed defining coupling as `w = E[‖∂F/∂T_j‖]`.

**In our simulator that is identically zero.** `downwash_force(dz, dxy)` has no
thrust argument — the Zhou model depends only on geometry. That matrix would be
all zeros. Verified.

Use instead:

| Quantity | Meaning |
|---|---|
| `W^F` | force magnitude — what `coupling.py` already computes |
| `W^p = ‖∂F/∂p_j‖` | geometry sensitivity, finite-difference verified |
| `H` | **closed-loop** impulse response with the PID running |

`H` matters most. Direct coupling being banded says **nothing** about
closed-loop propagation: D₀ perturbs D₁ → D₁'s PID reacts → its wake shifts →
D₂ feels it. `k>1` may matter even with a tridiagonal `W^F`. My earlier claim
that "near-diagonal `W` means k>1 is useless" was **wrong**.

A thrust-dependent wake model would be new physics needing its own validation —
out of scope.

---

### S.9 Mechanism framing — a theoretical correction

§1.5 currently argues that a critic seeing more than the actor produces an
"unrealizable advantage" that biases the policy gradient. **That framing is not
defensible as stated.**

With ideal conditional expectation and unlimited data, more conditioning
information cannot worsen the critic's optimal prediction — conditional
variance generally decreases as information increases.

**The defensible claim is finite-sample and function-approximation:**

- extra dimensions require more data to fit under a fixed budget;
- optimisation becomes harder;
- irrelevant or aliased information raises gradient variance;
- state-based critics can be biased under partial observability
  (Lyu et al., Theorem 2 is the formal anchor).

**The fixed sample budget is not a detail — it is central to the hypothesis.**
§1.5 must be rewritten accordingly before submission.

---

### S.10 Agreed plan — diagnostics before commitment

Ordered by cost. **Nothing downstream is decided until these run.**

| Step | What | Cost | Answers |
|---|---|---|---|
| **1** | **Environment-only factorial**: downwash × gust × correlation, pure PID, paired realisations | ~2 h, no RL | Is teammate state informative from *aerodynamics* or from *gust correlation*? Currently **completely confounded** at `gust_spatial_corr = 0.7` |
| **2** | Coupling matrices `W^F`, `W^p`, `H`; ranking overlap vs Euclidean | ~1 h, no RL | Does physics-ranking differ from distance-ranking in *our* geometry? How far does influence propagate? |
| **3** | Critic collapse | ~2 h | 4/15 runs collapsed past 350k. **Do not start by lowering `lr_critic`** — it may have been caused by the corrupted GAE targets now fixed. Re-run 3 seeds at 400k under v4 first |
| **4** | Choose the framing | — | With evidence rather than guesswork |

**Step 1 is decisive.** If the effect survives with downwash **off**, the
aerodynamic framing is wrong and both proposed pivots are moot.

**Why no RL in steps 1–2:** they characterise the environment for a fraction of
the cost, and the collapse is unresolved — training-based ablations would be
contaminated.

---

### S.11 Adopted regardless of any result

These are method improvements, not contingent on what the diagnostics show.

**String-stability metrics.** For gap error `e_i = p_{i-1} − p_i − Δ*_i`:

```
G_i = RMS(e_i) / RMS(e_{i-1})        G_i > 1 → errors amplify downstream
```

Call it **empirical downstream error amplification**, not string stability — a
formal claim needs controlled excitation and frequency-domain or induced-norm
analysis. Define `e_0` as the leader's own trajectory-tracking error so the
first follower has a valid denominator.

This is the control-systems contribution that differentiates the work from a
standard MARL ablation.

**Physical-unit reporting.** Per drone and per episode: RMS position error
(cm), max error, min separation, collision/abort rate, control effort, and
downstream amplification. **Return stays a training variable, not the headline
scientific result** — robotics reviewers dislike unitless return.

**Honest power analysis.** Verified arithmetic for a two-sided independent
t-test:

| d | power at n=8 | n for 80% |
|---|---|---|
| 1.00 | **46%** | 17 |
| 1.24 | **64%** | 12 |

My earlier "8 seeds should reach significance" was **wrong**. Plan 12–20 seeds,
powered from a **practically meaningful effect in centimetres**, not from the
pilot's Cohen's d — which is upward-biased because `k1` was the observed winner.
Report interval estimates and probability-of-improvement (Agarwal et al. 2021),
not `p < 0.05` alone.

**Ground-effect confound.** The main physics mode includes ground effect, and
changing formation spacing also changes individual altitudes. Either disable it
for the causal experiment or demonstrate by ablation that it is negligible.

---

### S.12 Deferred — good ideas, wrong time

| Item | Why deferred |
|---|---|
| Communication realism (packet age, validity mask, 20–100 ms delay, dropout) | Changes the **actor observation** → another task version → another calibration. Belongs in a robustness section. Both reviewers agree |
| Bidirectional coupling (suspended payload, tether) | Payload dynamics, tension constraints, much harder hardware. Separate paper |
| Formation reconfiguration (drones swapping vertical slots) | Good for Claim 3, but an extension |
| Cross-simulator validation (Crazyflow) | Strengthens "not a simulator artifact" but premature |

---

### S.13 Venue assessment

| Venue | Verdict |
|---|---|
| **RA-L** | **Realistic target** after reframing, resolving instability, and adding hardware |
| T-RO | Stretch — needs a formal selection method, multiple coupling regimes, mechanism evidence, strong baselines, substantial hardware |
| T-Cyber | Only with a broadly applicable algorithm validated beyond drones |
| JFR | **Poor fit** — its scope excludes validation restricted to controlled laboratory experiments |

**Baselines a reviewer will now expect:** MADDPG-K, attention/learned selection
(MAAC-style), and a downwash-aware classical controller. PID alone is no longer
a strong classical baseline.

---

## 0. How to read this document

This is the contract for the project. It exists so that we never reach the end and discover the work is not publishable. Every section is written to be attacked. If a section cannot survive your scrutiny now, it will not survive a reviewer's later.

Read it in this order:

1. **§1 Research gap** — is the hole real?
2. **§2 Claims** — are they falsifiable, and would we believe the opposite?
3. **§3 Kill criteria** — what result would end this project, and are we willing to accept it?
4. **§4–§7 Execution** — how we build it, in reviewable steps.
5. **§8–§11 Outputs, risks, venue** — what lands on paper.

**Nothing in §4 onward is written in code until you approve §1–§3.**

---

## 1. The research gap

### 1.1 What the field currently believes

Centralized Training with Decentralized Execution (CTDE) is the dominant paradigm in cooperative multi-agent RL. The standard recipe: give the critic everything available in simulation — all agents' states, all agents' actions, privileged simulator variables — while the actor sees only its own local observation. This is the design in MADDPG (2017), COMA (2018), MAPPO (2022), and in essentially every multi-robot RL system built since.

The implicit assumption behind that recipe is what this project attacks:

> **More critic information is never harmful. The critic should receive the maximum information the simulator can provide.**

This is stated nowhere as a formal claim and tested nowhere as a hypothesis. It is inherited practice.

### 1.2 What is already known (and therefore NOT our contribution)

We must be scrupulous here, because our previous submission was rejected on novelty. The following is **established** and will be cited as foundation, never claimed:

| Established finding | Source | Consequence for us |
|---|---|---|
| CTDE with a centralized critic mitigates non-stationarity | MADDPG, COMA | We use it; we do not claim it |
| Centralized critics are **not uniformly superior** to decentralized ones; they can raise gradient variance and introduce bias under partial observability | Lyu et al., *On Centralized Critics in MARL*, JAIR 2023 (extends AAMAS 2021) | **This is the closest prior work. Our gap sits directly on top of it.** |
| Asymmetric actor-critic: privileged critic input helps sim-to-real | Pinto et al. 2017 and successors | We use the mechanism; the *dosage question* is ours |
| Centralized-decentralized mismatch (CDM) degrades per-agent gradients | Multiple | Provides the mechanism we predict will cause the effect |
| Quadrotor swarm RL transfers zero-shot to Crazyflies | Batra et al. 2021, CrazyMARL 2025 | Our hardware protocol builds on theirs |
| Critic variants exist (concat / mean-field / counterfactual) | Various | These are *architectures*, not *observability levels* — a distinction we exploit |

### 1.3 The precise hole

Lyu et al. is the definitive treatment of "is a centralized critic better?" — and it has three boundaries that we verified directly against the paper and its published summary:

1. **It compares only the two extremes.** Fully centralized vs. fully decentralized. It does not study intermediate degrees of critic observability. There is no notion of a *spectrum*.
2. **It tests only toy domains.** Climb Game, Dec-Tiger, SMAC. No continuous control. No robot dynamics. No physical coupling between agents.
3. **It treats agents as symmetric.** No heterogeneous or role-asymmetric teams, no execution-time communication constraints, no sim-to-real.

Meanwhile the robotics side (Batra et al., CrazyMARL) builds working quadrotor swarm systems but treats the critic's input set as a fixed implementation detail — never as an experimental variable.

**So the hole is the intersection:**

> Nobody has treated *critic observability* as a continuous design axis, swept it on a system with genuine physical inter-agent coupling, and asked whether the optimum is at the maximum.

Two further sub-holes fall out of the same apparatus and are, as far as our search found, entirely untouched:

- **The asymmetry sub-hole.** When agents have structurally different roles (one whose optimal policy is independent of teammates, others whose policies depend on it), does a *uniform* critic design harm the independent agent? Nobody has asked whether critic observability should be **role-conditional**.
- **The transfer sub-hole.** Does the benefit of critic information measured in simulation *survive contact with hardware*? If critic centralization causes the policy to exploit inter-agent state correlations that are simulator artifacts, then simulation-only CTDE benchmarks systematically overstate real-world benefit.

### 1.4 Why the gap is open rather than merely unfashionable

A gap can be open because it's hard, or because it's uninteresting. This one is open because of a **field boundary**: the people who study critic theory (MARL/AAMAS/JAIR) work in gridworlds where physical coupling does not exist and the effect we predict cannot appear. The people who study drone swarms (RA-L/ICRA) treat the critic as plumbing and optimize for task success, not scientific ablation. The question requires both halves at once, and almost nobody sits in both.

That is a defensible answer to "why has nobody done this?" — the single hardest reviewer question.

### 1.5 The mechanism — why we expect an interior optimum

This is what separates a hypothesis from a guess. Three known effects act in opposite directions as critic observability `k` increases:

**Pushing performance up as `k` grows:**
- *Variance reduction.* A critic conditioned on teammate state can explain reward variation the local critic must absorb as noise. In physically coupled systems (downwash, formation drag) this component is large and real.

**Pushing performance down as `k` grows:**
- *Representational mismatch.* The critic learns `V(s_global)`, but the actor can only realize policies measurable w.r.t. its own observation. As `k` grows, the critic's target increasingly credits outcomes to information the policy provably cannot act on. The advantage signal acquires a component the actor cannot follow — it becomes noise in the policy gradient.
- *Estimation cost.* Critic input dimension grows with `k`; sample complexity to fit it grows accordingly. Under a fixed sample budget, a wider critic is a worse-fit critic.
- *CDM.* A teammate's exploratory noise enters this agent's value target, injecting variance that carries no signal for it.

Variance reduction saturates once the critic sees the teammate state that actually matters dynamically. Mismatch and estimation cost keep growing. **A sum of a saturating gain and a monotone cost has an interior maximum.** That is the claim, and it is why we expect the effect to be strongest exactly where physical coupling is strong but not total.

Critically, this mechanism predicts *where* the optimum sits: near the coupling horizon of the physics. That is a sharper, more falsifiable prediction than "somewhere in the middle," and it is what makes the result a design rule rather than a curiosity.

---

## 2. The claims

Each claim is stated so that the opposite outcome is plausible. If we cannot imagine the opposite, it is not a finding.

### Claim 1 — The observability optimum (primary)

> Task performance as a function of critic observability `k` is **non-monotonic**, with an interior maximum at partial centralization, under a fixed sample budget on physically coupled multi-robot control.

- **Opposite outcome (plausible):** performance is monotonically non-decreasing in `k`; maximal centralization is optimal; the field's practice is correct.
- **Measured by:** final task performance vs. `k`, ≥8 seeds, with the interior maximum required to survive a significance test against both endpoints.

### Claim 2 — The optimum tracks physical coupling

> The location of the optimum `k*` shifts **predictably with the strength of inter-agent physical coupling**: as formation spacing tightens (downwash grows), `k*` moves toward greater centralization.

- **Opposite outcome:** `k*` is constant, or moves unpredictably — the effect is an artifact of one configuration, not a principle.
- **Why it matters:** this converts Claim 1 from an observation into an actionable design rule. This is the claim that makes the paper *useful* rather than merely *interesting*.

### Claim 3 — Role-conditional observability (secondary)

> In role-asymmetric teams, **uniform** critic observability is dominated by **role-conditional** observability; specifically, agents whose optimal policy is independent of teammates are actively harmed by centralized critic input.

- **Opposite outcome:** uniform centralization is fine; role-conditioning yields no gain.
- **Note:** this claim is *born from the flaw in our previous project*. The one-way leader→follower dependency we identified as a weakness is precisely the regime where this becomes measurable. We are converting a defect into a contribution.

### Claim 4 — Transfer degradation (hardware, conditional)

> The simulation-measured benefit of critic centralization **does not transfer proportionally to hardware**; sim-to-real degradation increases with training-time critic observability.

- **Opposite outcome:** transfer is unaffected by critic design (critic is training-only, so a reviewer may reasonably expect exactly this — which is what makes a positive result striking).
- **Status:** included if hardware access materializes; the paper stands without it (see §11).

---

## 3. Kill criteria — decided in advance

We commit to these *now*, before we have any emotional investment in a result.

| # | Condition | Action |
|---|---|---|
| K1 | Pilot sweep (§5, Phase 3) shows performance flat or monotonic in `k` across ≥5 seeds, with overlapping CIs | **Claim 1 is dead.** Do not proceed to the full sweep. Pivot to Claim 3 as primary (§11 fallback A). |
| K2 | Task is solved to ceiling by all `k`, or by a PID baseline | Task is too easy. Return to §4 and increase difficulty. Do not run the sweep on a saturated task. |
| K3 | Seed variance exceeds the between-`k` effect size | Underpowered. Raise seed count or reduce scope; never report the effect as real. |
| K4 | Claim 1 holds but `k*` does not move with coupling (Claim 2 fails) | Paper is weaker but alive. Reframe as an empirical finding + mechanism study; target RA-L rather than T-RO. |
| K5 | A paper appears during our work that closes the gap | Reassess at every monthly literature sweep (§5, standing task). Pivot toward Claim 3/4, which are further from the crowd. |

**The rule:** a negative result on Claim 1, reported honestly with adequate power, is still publishable — "the field's assumption is correct, and here is the first evidence for it" is a real contribution. What is *not* publishable is an underpowered ambiguous result dressed up as a finding. We will not do that.

---

## 4. The experimental platform

### 4.1 Why the previous task cannot be reused

The old task fails every requirement, and understanding why defines the new one:

| Old property | Why it kills the experiment |
|---|---|
| Reward coupling is one-way only (leader ignores follower) | No genuine coordination; critic has nothing to gain |
| `ONE_D_RPM`, ±5% hover thrust, single scalar | Trivially solvable; every `k` saturates; no variance to reduce |
| 5-second episodes | Too short for coupling to accumulate |
| 2 agents, hardcoded | No scalability axis; `k` has only ~2 possible values |
| No disturbance | Nothing for teammate state to be informative *about* |

**The core insight for platform design:** the critic can only help if there is something *only it can see* that genuinely predicts return. That requires real physical coupling between agents. This is the single most important design constraint in the project.

### 4.2 Target task — tight-formation flight under disturbance

> ⚠️ **Partly superseded — see §S.4.** Control is now a three-layer stack (RL
> guidance over a PID position+attitude loop), topology is chain-based, and the
> disturbance is a gust field rather than steady wind. The *intent* below is
> unchanged; the implementation differs.

- **Agents:** N = 4 (primary), with 2 / 6 / 8 for scaling checks.
- **Control:** three-layer stack — RL policy commands a **3-D position offset**
  (guidance), a DSL PID runs position and attitude loops beneath it.
  *(Revised 28 Aug 2026: direct 4-D rotor control was tried as task version
  v1-raw4d and proved unlearnable — episodes ended after 20–100 of 720 steps
  even after 300k steps. See `docs/task_v2_notes.md`. The coupling question is
  unaffected, and a three-layer stack matches real swarm practice, which
  strengthens Claim 4.)*
- **Task:** maintain a commanded formation while tracking a leader trajectory that is *not* known to followers in advance.
- **Coupling (the essential ingredient):** formation spacing tight enough that **downwash** materially perturbs lower/trailing agents. `gym-pybullet-drones` implements downwash, drag, and ground effect — we will verify and, if necessary, strengthen the model. Spacing becomes our **coupling-strength knob** for Claim 2.
- **Disturbance:** stochastic gust field (Ornstein-Uhlenbeck, spatially
  correlated), resampled per episode. *Revised — see §S.4. A steady wind field
  was tried first and measured to be useless: a PID cancels a constant force
  exactly, leaving seed variance of 0.00.*
- **Episode:** 15–20 s, long enough for disturbance to accumulate.
- **Termination:** collision, altitude bound violation, or timeout.

### 4.3 The observability ladder — the independent variable

This is the heart of the design. Critic input at level `k`:

| Level | Critic sees | Interpretation |
|---|---|---|
| `k=0` | own observation only | Independent learner (decentralized critic) |
| `k=1` | own + nearest neighbor state | Minimal centralization |
| `k=2` | own + 2 nearest neighbors | Partial |
| `k=3` | own + 3 nearest neighbors | Partial |
| `k=N-1` | all teammate states | Standard CTDE ("give it everything") |
| `k=N-1+P` | all teammates + privileged sim state (wind vector, true downwash forces) | Maximal / asymmetric-critic style |

**Controls that make the sweep interpretable — non-negotiable:**

- **The actor is byte-identical across all `k`.** Same architecture, same input, same parameter count. Only the critic's input set changes. Any performance difference is therefore attributable to critic information alone.
- **Critic parameter count held constant** across `k` via a fixed-width input projection layer. Otherwise we are measuring critic capacity, not critic information — the most likely reviewer objection, pre-empted by design.
- **Neighbor ordering is distance-sorted and permutation-invariant** (or attention-pooled), so `k` varies information content, not encoding difficulty.
- **Identical sample budget, identical hyperparameters** across all `k`. Hyperparameters tuned once at `k=N-1` (the *baseline's* favor, not ours — tuning in favor of the standard practice we are challenging is the honest choice and blocks the "you sabotaged the baseline" objection).

### 4.4 Baselines

- PID / geometric formation controller (establishes that learning is needed at all)
- Independent PPO (`k=0`)
- Standard MAPPO (`k=N-1`) — the incumbent we are testing
- Asymmetric actor-critic with full privileged state (`k=N-1+P`)

---

## 5. Execution plan

> **Phases 0–3 are complete — see §S.1 for outcomes.** The plan below is the
> original schedule; treat it as intent, not status.

Built in reviewable steps. **Each phase ends with an artifact you and I inspect together before the next begins.** Nothing is written as a monolith; every file is explained line-by-line as we go, per your requirement.

### Phase 0 — Foundations *(week 1)*
Establish the literature position and the reproducibility spine.
- Full related-work matrix; register the gap claim precisely
- Repository skeleton, dependency pinning (**modern Ray 2.x — the old `ray.rllib.agents` API is deprecated and will fail reviewer reproduction**), seeding discipline, experiment config system
- **Exit artifact:** annotated bibliography + a runnable "hello world" training job with fixed seed producing bit-identical output twice.

### Phase 1 — Environment *(weeks 2–3)*
- Formation task, wind model, downwash verification
- **Physics validation:** empirically confirm downwash actually perturbs trailing agents at our spacings — plot force vs. separation. *If the coupling isn't real, the entire project's premise fails here, so we check it first.*
- **Exit artifact:** coupling-strength characterization plot + environment unit tests.

### Phase 2 — Algorithm core *(weeks 4–5)*
- MAPPO with parameterized critic observability `k`
- Fixed-width projection; permutation-invariant neighbor encoding
- **Verification:** confirm the critic actually receives what we think (gradient-flow tests, input-shape assertions, a deliberate "critic sees nothing" sanity run that must fail to learn)
- *This is where the previous project silently broke. We test the wiring explicitly rather than trusting it.*
- **Exit artifact:** unit-tested trainer + proof the critic input set changes with `k`.

### Phase 3 — Pilot sweep ⚠️ **DECISION GATE** *(week 6)*
- `k ∈ {0, 1, N-1}`, 5 seeds, reduced budget
- **Purpose: kill the project cheaply if the effect isn't there.**
- **Exit artifact:** a plot. If flat/monotonic → invoke K1 and pivot before spending months.

### Phase 4 — Full sweep *(weeks 7–9)*
- All `k`, ≥8 seeds, full budget, with CIs and significance tests
- **Exit artifact:** the paper's headline figure (performance vs. `k`).

### Phase 5 — Coupling sweep (Claim 2) *(weeks 10–11)*
- Cross `k` with 3–4 formation spacings
- **Exit artifact:** `k*` vs. coupling strength — the design rule.

### Phase 6 — Asymmetry (Claim 3) *(weeks 12–13)*
- Role-asymmetric variant; uniform vs. role-conditional observability
- **Exit artifact:** per-role learning curves.

### Phase 7 — Mechanism *(weeks 14–15)*
- Instrument gradient variance, critic TD error, advantage-signal realizability
- **Purpose:** show *why* the curve bends. Elevates the paper from "we observed" to "we explained." **This is what distinguishes Q1 from a workshop paper.**
- **Exit artifact:** mechanism decomposition figure.

### Phase 8 — Hardware (Claim 4, conditional) *(weeks 16–18)*
- Crazyflie deployment, ≥3 `k` levels, quantified sim-to-real gap

### Phase 9 — Writing *(weeks 19–22)*
- Draft, internal adversarial review, artifact release

**Standing task, monthly:** literature sweep for scoop risk (K5).

---

## 6. Expected outputs

### 6.1 Figures (the paper's spine)
1. **Performance vs. `k`** — the interior optimum. *The paper lives or dies here.*
2. **`k*` vs. coupling strength** — the design rule (Claim 2).
3. **Mechanism decomposition** — variance reduction vs. mismatch cost as `k` grows.
4. **Per-role curves under uniform vs. role-conditional critics** (Claim 3).
5. **Sim-to-real gap vs. `k`** (Claim 4, conditional).
6. Coupling characterization (downwash force vs. separation) — supplementary.

### 6.2 Numerical results
- Full `k` × seed × spacing table, mean ± 95% CI
- Significance tests for interior-optimum vs. both endpoints
- Sample-efficiency curves
- Baseline comparison including classical control

### 6.3 Artifacts
- Open-source environment + trainer, seeded and reproducible
- Complete config + log release
- Hardware protocol and flight logs (if Phase 8 runs)

---

## 7. Standard of evidence

Non-negotiable, because these are the things that get papers desk-rejected:

- **≥8 seeds** everywhere, mean ± 95% CI, always plotted
- **Significance testing** for every comparative claim; effect sizes reported
- **Identical budgets** across conditions
- **Hyperparameters tuned in the baseline's favor**, tuning protocol disclosed
- **Negative results reported**, including failed conditions
- **Compute cost disclosed** in full
- **Every figure regenerable** from released code with one command

---

## 8. Why this reaches Q1

| Reviewer question | Our answer |
|---|---|
| What is new? | Critic observability as a *design axis*; the interior optimum; the coupling-tracking rule |
| Why hasn't it been done? | Field boundary between MARL theory (toy domains) and swarm robotics (critic as plumbing) |
| Could the result have been otherwise? | Yes — monotonic improvement is the field's current default belief |
| Is it useful? | Yes — Claim 2 yields an actionable rule for choosing `k` from physical coupling |
| Do you explain it, or just observe it? | Phase 7 mechanism study |
| Is it rigorous? | §7 |
| Does it work on real robots? | Phase 8 (conditional) |

**Target venues:** IEEE T-RO or IEEE RA-L (primary); IEEE T-Cyber, JFR, or Autonomous Robots (alternates). **Not** SoftwareX-tier — that positioning contributed to the previous rejection.

---

## 9. What we explicitly are NOT claiming

Guarding against the previous rejection's failure mode:

- We do **not** claim to invent CTDE, centralized critics, or MAPPO
- We do **not** claim a new RL algorithm
- We do **not** claim "distributed training is faster" (a throughput fact, not a research finding — this was a core weakness of the inherited SoftwareX work)
- We do **not** claim superiority of one algorithm over another as a headline; algorithm choice is held fixed so that `k` is the only variable

---

## 10. Risks

| Risk | Severity | Mitigation |
|---|---|---|
| Effect doesn't exist | **Critical** | Phase 3 gate kills cheaply; K1 pivot ready |
| Task too easy → saturation | High | Phase 1 validates difficulty before sweeps |
| Coupling too weak to matter | High | Phase 1 measures it explicitly; tighten spacing or strengthen model |
| Confound: critic capacity not information | High | Fixed-width projection by design (§4.3) |
| Compute cost | Medium | Pilot first; scope seeds to available hardware; disclose |
| Scooped | Medium | Monthly sweep; Claims 3/4 are further from the crowd |
| Hardware unavailable | Low | Paper stands on Claims 1–3 (§11) |
| Ray API churn | Low | Pin versions; containerize |

---

## 11. Fallback ladder

- **Full success (Claims 1+2+3+4):** T-RO
- **No hardware (1+2+3):** RA-L or T-Cyber — still strong
- **Fallback A — Claim 1 dies, Claim 3 survives:** reframe around role-conditional critics for heterogeneous teams; RA-L
- **Fallback B — effect exists but is small:** reframe as a rigorous negative/boundary result: "centralization is not free, and here is when it stops paying." Honest, publishable, still useful
- **Fallback C — everything monotonic:** "the first rigorous evidence that the field's assumption holds, and why" — a genuine service to the field, publishable at a lower tier

---

## 12. Open questions for your review

> **Mostly answered.** Hardware: yes, Crazyflies at KFUPM, **indoor only**
> (Bitcraze advise against outdoor use). Simulator: resolved in §13. Compute:
> ~46 steps/s/process, 4 workers, thread-capped — see §S.6. Remaining open
> items are listed in §S.5.

Answer these before Phase 0 begins:

1. **Hardware:** do we have or can we get Crazyflies and a flight space at KFUPM? Determines whether Claim 4 is in scope.
2. **Compute:** what GPU/CPU budget is available? Directly sets seed count and sweep size — the single largest constraint on statistical power.
3. **Timeline:** the plan spans ~22 weeks. What is the real deadline?
4. **Team split:** which phases do you own vs. Eman?
5. ~~**Simulator commitment**~~ — **RESOLVED, see §13.**
6. **Is the gap acceptable to you?** If §1 does not convince you, it will not convince a reviewer — and now is the time to say so.

**Answered so far (21 Aug 2026):**
- **Q1 Hardware — YES.** Crazyflies are available at KFUPM. **Claim 4 and Phase 8 are IN SCOPE.** This is our strongest differentiator against Lyu et al. (toy domains only) and must be protected in every subsequent decision.
- **Q5 Simulator — RESOLVED**, see §13.

---

## 13. Simulator decision (resolved 21 Aug 2026)

**Decision: `gym-pybullet-drones` for Phases 1–7. `crazyflow` added for the Phase 8 hardware bridge only. No migration to MuJoCo.**

### The decisive criterion

Downwash is not a fidelity nicety in this project — **it is the independent variable of Claim 2.** The coupling-strength knob (§4.2) *is* inter-agent aerodynamic interaction. A simulator without it makes Claim 2 untestable and leaves the §1.5 mechanism with nothing to act on.

Every candidate was therefore screened on inter-agent aerodynamics first, speed second.

### Candidates evaluated

| Simulator | Speed | Downwash / inter-agent aero | CF sim-to-real validated | Verdict |
|---|---|---|---|---|
| **gym-pybullet-drones** | Slow (CPU only) | **Yes** — Zhou DSL model, validated vs. real CF data | Yes | **CHOSEN (primary)** |
| **Crazyflow** (JAX) | 914M steps/s, RTX 4090 | **No** — authors state "single-agent aerodynamics only" | **Yes — best measured** (−47.5% RMS pos. error vs. pybullet) | **Phase 8 bridge only** |
| **CrazyMARL** (MuJoCo MJX) | 16,384 parallel envs | **No** — rigid body, cables, contact only | Yes, zero-shot | Rejected |
| **MuJoCo-Drones-Gym** | GPU-accelerated | Yes — Gaussian model, CF2.x sysID | **No — "planned future work"** | Rejected (see below) |

### Why not MuJoCo-Drones-Gym (the only real contender)

It is the sole GPU-accelerated option that models downwash, and on paper it is what we want. Rejected on three grounds:

1. **No hardware validation.** Sim-to-real is explicitly future work. With Crazyflies confirmed and Claim 4 in scope, founding the project on an unvalidated-to-hardware simulator would undercut our strongest asset.
2. **Two months old** (arXiv June 2026). A 22-week project cannot absorb being an early stress-tester of a new codebase during Phase 1.
3. **Unquantified speed claim.** Parallelizability is asserted; no throughput benchmark against gym-pybullet-drones is published.
4. **Weaker downwash fidelity** — a Gaussian fit vs. PyBullet's validated Zhou DSL model. We would be trading accuracy in *the exact physics the paper is about* for an unmeasured speed gain.

### Why speed matters less than it appears

**Our bottleneck is seed count, not steps/second.** The full design is ~8 seeds × 6 observability levels × 4 spacings ≈ 200 runs. That is an *embarrassingly parallel* workload: 200 independent CPU jobs, not one job needing GPU vectorization. PyBullet parallelizes across cores adequately. GPU vectorization optimizes a dimension we are not constrained on — **pending confirmation of the KFUPM CPU allocation (Q2, still open).**

### Defensive benefit

Using the simulator with the best-validated downwash model is the honest answer to "is your coupling effect a simulation artifact?" — a question Claim 2 invites directly. Migrating to a faster simulator with weaker or absent aerodynamics would hand a reviewer that attack for free.

### Contingency

If the Phase 3 pilot shows we are genuinely compute-bound, re-evaluate MuJoCo-Drones-Gym at that point — with ~6 months of additional maturity on it and real measurements of what we need. **Do not pre-emptively migrate.**

### Consequent changes elsewhere in this charter

- §4.2 coupling knob is implemented via gym-pybullet-drones' downwash model; **Phase 1's exit artifact (force vs. separation) doubles as the calibration for Claim 2.**
- §5 Phase 8 gains a task: port the trained policy to `crazyflow` for the transfer bridge, exploiting its measured −47.5% sim-to-real gap.
- §10: "Ray API churn" risk is retained; PyBullet + modern Ray 2.x remains the pinned stack.

---

## Appendix A — Key references to secure first

- Lyu et al., *On Centralized Critics in Multi-Agent Reinforcement Learning*, JAIR 2023 — **the paper we must position against most carefully**
- Lowe et al., MADDPG, NeurIPS 2017
- Foerster et al., COMA, AAAI 2018
- Yu et al., *The Surprising Effectiveness of MAPPO*, NeurIPS 2022
- Pinto et al., *Asymmetric Actor Critic*, 2017
- Batra et al., *Decentralized Control of Quadrotor Swarms*, CoRL 2021
- Panerati et al., *Learning to Fly* (gym-pybullet-drones), IROS 2021
- CrazyMARL, 2025
- Amato, *An Introduction to CTDE*, 2024 (survey framing)
