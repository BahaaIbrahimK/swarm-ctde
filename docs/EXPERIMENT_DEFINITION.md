# Experiment Definition — Critic Observability in Formation Flight

**The reference document for this project.** Read this before running anything,
and come back to it whenever a term or a number is unclear.

| | |
|---|---|
| **Experiment** | Does a centralized critic have an *optimal* amount of information? |
| **Phases covered** | 1 (physics premise), 2 (apparatus), 3 (pilot decision gate) |
| **Task version** | **v4-correctness** (v1–v3 abandoned — see the version table in [`../PROJECT_CHARTER.md`](../PROJECT_CHARTER.md) §S.3) |
| **Status** | Phase 1 ✅ · Phase 2 ✅ · Phase 3 pilots ❌ **invalid** (ran under v3, before the September audit) · re-run pending |
| **Related** | [`../PROJECT_CHARTER.md`](../PROJECT_CHARTER.md) · [`downwash_model_notes.md`](downwash_model_notes.md) · [`task_v2_notes.md`](task_v2_notes.md) · [`task_v3_notes.md`](task_v3_notes.md) · [`audit_2026-09-16.md`](audit_2026-09-16.md) |

---

## Table of contents

1. [Glossary — read this first](#1-glossary--read-this-first)
2. [The problem](#2-the-problem)
3. [The RL problem, defined](#3-the-rl-problem-defined)
4. [The formation — who knows what](#4-the-formation--who-knows-what)
5. [The observability ladder](#5-the-observability-ladder)
6. [Why 1.0 m spacing](#6-why-10-m-spacing)
7. [What we expect](#7-what-we-expect)
8. [The phases](#8-the-phases)
9. [Reading the output](#9-reading-the-output)
10. [Kill criteria](#10-kill-criteria)

---

## 1. Glossary — read this first

Every term used later, defined once.

### Reinforcement learning basics

**Agent** — one decision-maker. Here, one drone.

**Policy** — the agent's brain. A function: `observation → action`. Training
means adjusting it so its actions earn more reward.

**State** — everything true about the world right now: all drones' positions,
velocities, the wind vector. The complete truth.

**Observation** — the slice *one agent* can actually see. Smaller than the
state. Drone 2 knows its own position and velocity, but not the wind, and not
what drone 3 is about to do.

**Partial observability** — when observation < state. Every real robot has
this. A real drone genuinely cannot sense the wind field.

**Reward** — a number after each step saying how well the agent did. Since
task version v2 it includes a **survival bonus**, so returns are **positive**:
a good policy scores ~600–670 out of a 720 ceiling. Higher is better.

**Return** — total reward accumulated over an episode. This is the headline
performance number.

**Episode** — one attempt, start to finish. Here: 15 seconds of flight (720
steps at 48 Hz), or until a crash.

**Rollout** — a batch of experience collected with the current policy, used for
one update and then discarded.

**Step / timestep** — one control decision. At 48 Hz control rate, one second
of flight = 48 steps.

**Sample budget** — total training steps. **Held identical across all rungs**,
otherwise a rung could win simply by getting more practice.

### Actor–critic

**Actor** — the policy. Chooses actions. **This is what flies the drone**, and
what would be deployed on real hardware.

**Critic** — a coach that never flies. Estimates "how good is this situation?"
so the actor knows which way to adjust. **Deleted after training.**

> **This is the single most important idea in the project.** Because the critic
> is thrown away, it can see things the drone could never know in reality —
> teammate states, the true wind vector. It is a coach with a god's-eye replay.

**Value** — the critic's estimate of expected future reward from a state.

**Advantage** — "was that action better or worse than expected?"
`advantage = what happened − what the critic predicted`. This is the signal the
actor learns from.

**GAE (Generalised Advantage Estimation)** — the standard way to compute
advantage, trading bias against variance.

**Explained variance** — how much of the return variation the critic actually
captures. `1.0` = perfect, `0.0` = no better than guessing the mean, negative =
worse than the mean. **Our key diagnostic for whether the critic is working.**

### Multi-agent

**CTDE (Centralized Training, Decentralized Execution)** — train with global
information, deploy with only local information. The dominant paradigm in
multi-agent RL.

**Non-stationarity** — when other agents are also learning, the environment
appears to change under you. An action that worked yesterday fails today, not
because it was wrong, but because everyone else changed. CTDE exists to
mitigate this.

**Privileged information** — data given to the critic that the actor can never
have at deployment (e.g. the wind vector). Legitimate precisely because the
critic is discarded.

**MAPPO** — Multi-Agent PPO. The algorithm we use, held **fixed** so that `k`
is the only variable.

**PPO clipping** — prevents catastrophically large policy updates by limiting
how far the new policy may move from the old one.

### This experiment's own terms

**`k`** — **the independent variable.** How many teammates the critic sees.
`k=0` means own-observation only; `k=3` means all three teammates.

**Rung** — one setting of `k`. The ladder is `k0, k1, k2, k3, k3+P`.

**`k*`** — the *optimal* `k`, i.e. where performance peaks. Claim 1 says this
sits in the middle, not at the maximum.

**Downwash** — the column of fast-moving air pushed down by a rotor. A drone
flying underneath gets shoved down and loses lift. **This is the physical
coupling the whole paper rests on.**

**Coupling horizon** — the separation beyond which downwash stops mattering.
Measured in Phase 1; predicts where `k*` should sit.

**Formation spacing** — vertical gap between drones. **The coupling-strength
knob for Claim 2**: tighter spacing = stronger coupling.

**Seed** — the starting number for the random number generator. Same seed =
identical run. Different seeds = genuinely different outcomes.

**Confidence interval (95% CI)** — the range the true value probably lies in.
**Overlapping CIs mean two rungs cannot be distinguished.**

**Confound** — an alternative explanation for a result. Our biggest risk: if
higher `k` also meant a *bigger* network, we would be measuring capacity rather
than information.

**Task version** — a tag (`v1-raw4d`, `v2-pidassist`) marking a change to the
task definition itself: reward, action space, termination, or control mode.
**Results from different task versions must never be pooled into one figure.**
The tag appears in every result path.

**Sidestep exploit** — the degenerate strategy of drifting laterally to escape
downwash. Closed in v2 by weighting lateral error 10×. If trained policies show
large `eval_lateral_error_m`, it is back and results are invalid.

**Episode completion** — fraction of the 720-step episode actually flown. **The
honest learning signal.** In v1 returns rose 59% while completion stayed at
3–14% — the policy was learning to crash faster, not to fly.

---

## 2. The problem

### What everyone currently does

The standard CTDE recipe: give the critic **everything** the simulator can
provide — all agents' states, all actions, privileged variables. Since the
critic is discarded anyway, more information is assumed to be free.

The buried assumption:

> **More critic information is never harmful. Give the critic the maximum
> available.**

**Nobody has ever tested this.** It is not a theorem and not an experimental
finding — it is inherited habit, copied from MADDPG (2017) because MADDPG
worked.

### What we claim instead

> Performance versus critic observability `k` is **non-monotonic**, peaking at
> an interior value. And that peak **moves predictably** with the strength of
> physical coupling between agents.

### Why an interior peak should exist

Three forces act as `k` grows.

**Pushing performance UP:**

- *Variance reduction.* A drone dips unexpectedly. A critic seeing only that
  drone concludes "bad action" — **wrong**; it was downwash. A critic that also
  sees the drone above knows better. It explains away noise.

**Pushing performance DOWN:**

- *Representational mismatch.* The critic learns "this is good" partly because
  the wind happens to be favourable. But the actor cannot sense wind. That part
  of the critic's judgment is a lesson in a language the student does not
  speak — it becomes noise in the learning signal.
- *Estimation cost.* A wider critic needs more data to fit. Under a fixed
  budget, a bigger critic is a *worse-fitted* critic.
- *CDM.* Teammates' random exploration leaks into your value estimate.

**The shape:** variance reduction **saturates** — once the critic sees the
teammate actually blowing air on you, adding a distant fourth drone adds
nothing. But mismatch and estimation cost **keep growing**.

> A saturating gain plus a steadily growing cost has a peak in the middle.

And the mechanism predicts *where*: near the **coupling horizon** — the critic
should see the teammates that aerodynamically matter, and no more.

---

## 3. The RL problem, defined

### Observation (what each drone sees) — 12 numbers

| Index | Content | Meaning |
|---|---|---|
| 0–2 | position error (x, y, z) | how far off my formation slot |
| 3–5 | roll, pitch, yaw | how I am tilted |
| 6–8 | velocity (vx, vy, vz) | how fast I am moving |
| 9–11 | angular velocity | how fast I am rotating |

Note: position **error**, not absolute position. The drone knows how far it is
from where it should be — that is what makes formation-keeping learnable.

> **Wind is NOT in the observation.** It is redrawn every episode and no drone
> can sense it. This is deliberate: wind is a *hidden common cause* pushing all
> drones simultaneously, so a teammate's velocity carries information about a
> disturbance the agent cannot see directly. Exactly the variance a centralized
> critic might explain away.
>
> **Bug found 28 Aug 2026:** wind was sampled and handed to the critic but
> **never applied as a force**. A pure-PID policy scored 598.6 ± 0.00 at 0, 2,
> and 5 m/s — identical, zero variance. With no disturbance a `k=0` critic
> predicts almost perfectly and teammate information adds nothing, so the sweep
> would have come back flat **for reasons unrelated to Claim 1** — silently
> faking kill-criterion K1. Now applied as a drag force before the PID responds.
> See [`task_v2_notes.md`](task_v2_notes.md).

### Action — 3 numbers  *(task version v4-correctness)*

A desired **position offset** from the drone's formation slot, in `[-1, 1]`,
scaled by `max_offset_m = 0.2`:

```
target_position = formation_slot + action × 0.2 m
```

`0` = "aim exactly at my slot" · `+1` = 0.2 m offset in that axis.

#### The three-layer control stack

The policy does **not** command rotors. It sits above a complete PID stack:

| Layer | Who | Job |
|---|---|---|
| **Guidance** | **RL policy** | *where* to aim, given wind and neighbours |
| Position | DSL PID | how to get there |
| Attitude | DSL PID | how to stay upright doing it |

```
policy → offset from slot (3-D)
              ↓
      slot + offset = target position
              ↓
   ┌──────────────────────────────┐
   │  DSL PID: position loop      │
   │           attitude loop      │
   └──────────────────────────────┘
              ↓
         rotor RPMs → physics
```

> **State this openly in the paper.** The action space is **guidance-level**,
> not motor-level. A reviewer expecting end-to-end motor control should find
> that stated plainly rather than buried.

**Why not direct rotor control?** Task version v1 tried exactly that and
failed: episodes ended after 20–100 of 720 steps even after 300,000 steps of
training. A quadrotor is underactuated and falls within ~1 s of imperfect
thrust, so a randomly-initialised network never got past attitude
stabilisation to reach the formation problem at all.

**Why this does not weaken the paper.** The paper is about *what the critic
sees*, not what the actor controls. Downwash still perturbs drones, wind still
pushes everyone unobserved, the actor still sees only its own 12 numbers, and
the critic still sees `k` teammates. The mechanism is untouched — and a
three-layer stack is what real swarms use, so Claim 4 (sim-to-real) becomes
*more* credible.

### Reward

```python
reward = + 1.0                                  # survival bonus
         − 10.0 × (lateral error)²              # lateral weighted 10× vertical
         − (vertical error)²
         − 0.05 × (tilt)²
         − crash penalty (scaled by time remaining)   # only on termination
```

Returns are now **positive** — a good policy scores ~600–670 out of a 720
ceiling, since the survival bonus dominates when errors are small.

**Why the survival bonus?** In v1, reward was a pure per-step penalty with
early termination, so crashing at step 30 accumulated *fewer* penalties than
flying badly for 720. Returns "improved" 59% while episode length stayed flat
— the policy was learning to **die faster**. The bonus plus a
time-scaled crash penalty makes return monotone in survival time.

**Why is lateral error weighted 10×?** This closes the **sidestep exploit**: at
1 m separation, a 10 cm lateral offset removes 86% of the downwash for a
reward cost of only 0.45% of a typical return. Without the weighting, the
optimal policy is to step aside — which would delete the very coupling this
paper studies and produce a flat curve that *looks* like "no effect."

The tilt penalty is deliberately small: tilting is *required* to translate.

### Episode

- **Start:** drones stacked in formation; wind drawn randomly (0–12 m/s)
- **Duration:** 15 seconds (720 steps at 48 Hz)
- **Ends early if:** a drone hits the floor, tips past ~70°, or drifts 3 m off
  station

**Wind magnitude is calibrated, not arbitrary.** At `wind_max = 12 m/s` a
pure-PID policy scores ~520/720, leaving **27.7% headroom** for the policy to
win, with substantial seed variance (std ~122) for the critic to explain, and
zero crashes. Below ~5 m/s the PID absorbs everything and there is nothing to
learn (kill-criterion K2); above ~16 m/s crashes appear.

### Environment

| Setting | Value | Why |
|---|---|---|
| Simulator | gym-pybullet-drones | Only option with a *validated* downwash model |
| Physics rate | 240 Hz | Simulator default, matches Crazyflie system ID |
| Control rate | 48 Hz | Physics must run faster than control for stability |
| Drone | Crazyflie 2.x (27 g) | KFUPM has these — needed for Phase 8 hardware |
| Aerodynamics | ground effect + drag + **downwash** | Downwash is the coupling channel |
| Control | 3-layer: RL guidance → PID position → PID attitude | v1's direct rotor control was unlearnable |
| Wind | 0–12 m/s, resampled per episode, **unobserved** | The hidden common cause (calibrated for headroom) |

---

## 4. The formation — who knows what

### Physical arrangement

Four drones in a **vertical stack**:

```
        🚁  drone 0  (LEADER)     altitude 3.80 m
        │
        │  1.0 m  ← formation spacing
        ▼
        🚁  drone 1               altitude 2.80 m
        │
        │  1.0 m
        ▼
        🚁  drone 2               altitude 1.80 m
        │
        │  1.0 m
        ▼
        🚁  drone 3               altitude 0.80 m  ← floor clearance
```

The leader flies a slow circle (0.5 m radius, 12 s period, gentle vertical
sinusoid). Everyone else holds position directly beneath it.

**Why vertical?** Downwash flows *downward*. Stacking puts every follower in the
wake of the drones above — the coupling channel this project studies.

**Geometry note:** the stack is built **upward from a floor clearance**, not
downward from a fixed leader altitude. Building downward puts the lowest drone
underground as soon as `num_drones × spacing` exceeds the leader's altitude.

### Who gets more information as `k` grows — **everyone**

> **There is no "leader knows more, followers know less."** `k` applies to all
> four drones equally.

Every drone has:

| | Actor | Critic |
|---|---|---|
| Role | chooses actions — **flies the drone** | judges "was that good?" during training |
| Exists at deployment? | **Yes** | **No — deleted** |
| Input | own 12 numbers, **always** | `k` teammates — **the experiment** |

At `k=2`, *every* drone's critic sees its 2 most-influential teammates.

At deployment you have four drones each running only its own 13-D actor, with
no radio link between them. That is what "decentralized execution" means.

### Which teammates? Ranked by **downwash influence**, not distance

This is a direct consequence of Phase 1:

| Configuration | Force |
|---|---|
| 0.6 m above, 0.00 m offset (near, aligned) | 0.211 N |
| 0.6 m above, 0.10 m offset (near, **offset**) | **0.000 N** |
| 1.5 m above, 0.00 m offset (far, **aligned**) | **0.034 N** |
| 0.5 m **below** | 0.000 N |

A drone 0.6 m above but 10 cm sideways exerts **zero** force, while one 1.5 m
above and aligned exerts real force. **Euclidean ranking would have handed the
critic an aerodynamically irrelevant teammate at `k=1`** and silently corrupted
every rung of the ladder.

So "nearest neighbour" means *"the teammate whose wake affects me most."*

---

## 5. The observability ladder

| Rung | Critic sees | Dimensions |
|---|---|---|
| `k0` | own observation only | **12** |
| `k1` | own + 1 teammate | **18** |
| `k2` | own + 2 teammates | **24** |
| `k3` | own + 3 teammates | **30** |
| `k3+P` | own + 3 teammates + wind vector | **33** |

### The arithmetic

```
 12  = own observation
 +6  = per teammate: relative position (3) + relative velocity (3)
 +3  = wind vector (privileged rung only)
```

So `k1` = 12 + 6 = **18** · `k2` = 12 + 12 = **24** · `k3` = 12 + 18 = **30**.

**Why 6 numbers per teammate rather than their full 12?** Relative quantities
are what determine coupling — "where are you compared to me" is what sets
downwash. And keeping it small matters for the hypothesis: if each teammate
cost 12 dimensions, estimation cost would swamp variance reduction before the
trade-off became observable.

### Both endpoints matter

`k0` is a genuinely decentralized critic; `k3+P` is maximal centralization.

**Claim 1 requires the interior optimum to beat BOTH.** Beating only `k0` would
merely re-show that centralization helps — known since MADDPG (2017).

### The three confound controls

Without these the sweep is uninterpretable and a reviewer will say so.

**1. The actor is byte-identical across all rungs.**

```
ACTOR   12 numbers  ─────────────────  never changes with k
CRITIC  12 → 18 → 24 → 30 → 33  ─────  THIS is the experiment
```

Any performance difference is therefore attributable to critic information
alone.

**2. Critic capacity is held constant.**

Naively, a wider input means more first-layer weights, so `k3` would have more
*capacity* than `k1` — and we would be measuring capacity, not information.
**This is the most likely reviewer objection to the entire paper.**

Fix: a fixed-width projection layer feeding an identical trunk.

| input dim | projection | trunk | total |
|---|---|---|---|
| 12 | 832 | **82,689** | 83,521 |
| 18 | 1,216 | **82,689** | 83,905 |
| 24 | 1,600 | **82,689** | 84,289 |
| 30 | 1,984 | **82,689** | 84,673 |
| 33 | 2,176 | **82,689** | 84,865 |

The trunk — where essentially all capacity lives — is identical to the
parameter. The residual 1,344-parameter spread **favours high `k`**, working
*against* our hypothesis. If performance still drops at high `k`, capacity
cannot be the explanation. That direction of bias is what makes the control
credible.

**3. Neighbour ordering is physical and permutation-invariant** — by predicted
downwash force, so `k` varies information content rather than encoding
difficulty.

---

## 6. Why 1.0 m spacing

Three constraints pin it down.

### Lower bound — the physics model breaks below 0.6875 m

The downwash model has a Gaussian width term:

```
beta = 0.16 × separation − 0.11      →  zero at 0.6875 m
```

Below that, `beta` goes **negative**. Since it is squared, nothing crashes —
**the failure is completely silent** — but a negative width is physically
meaningless. We would be extrapolating outside the range the model was
calibrated on.

> **Must be disclosed in the paper.** A reviewer examining Claim 2 will ask
> whether the coupling model is trustworthy at the spacings used. See
> [`downwash_model_notes.md`](downwash_model_notes.md).

### Upper bound — coupling must still matter

| Separation | Force | As multiple of drone weight |
|---|---|---|
| 0.70 m | 0.155 N | 0.58× |
| 1.00 m | 0.076 N | 0.29× |
| 1.50 m | 0.034 N | 0.13× |
| 2.50 m | 0.012 N | 0.05× |

At 2.5 m there is barely anything for the critic to explain.

### Also — drones must clear the ground

Ground effect is a *separate* aerodynamic phenomenon. 4 drones × 1.0 m from a
0.8 m floor clearance puts the leader at 3.8 m, keeping the whole stack clear.

### Verdict

**1.0 m** is safely above the validity threshold, gives coupling at 29% of body
weight, and is geometrically workable.

**For Claim 2, spacing becomes the dial** — rerun at 0.7 / 1.0 / 1.5 / 2.5 m
and check whether `k*` shifts. Verified manipulation check:

| Spacing | Total team coupling |
|---|---|
| 0.70 m | 0.5579 N |
| 1.00 m | 0.2737 N |
| 2.50 m | 0.0438 N |

**12.7× contrast, strictly monotonic** — the knob changes the physics, not just
the commanded geometry.

---

## 7. What we expect

### The two competing shapes

```
FIELD'S ASSUMPTION (monotonic)       OUR CLAIM 1 (interior optimum)

perf │        ╭────  k3              perf │      ╭──╮  k1 or k2
     │     ╭──╯                           │    ╭─╯  ╰──╮
     │  ╭──╯                              │  ╭─╯       ╰──  k3
     │╭─╯  k0                             │╭─╯  k0
     └──────────────── k                  └──────────────── k
```

### Our prediction

`k*` should sit **near the coupling horizon** — the critic should see the
teammates that aerodynamically matter and gain little from the rest.

At 1.0 m spacing, relative coupling from a drone's perspective:

| Teammate distance | Relative influence |
|---|---|
| 1.0 m (adjacent) | 1.00 |
| 2.0 m (two away) | 0.25 |
| 3.0 m (three away) | 0.11 |

So we predict **`k* ≈ 1` or `2`** — the adjacent teammate matters greatly, the
one three slots away contributes ~11% of the influence while costing the same
6 input dimensions.

### And the outcome that would refute us

Performance rising monotonically to `k3+P`. That is what most researchers would
bet on — **which is exactly why the experiment is worth running.**

---

## 8. The phases

### Phase 1 — Premise check ✅ DONE

**Pure physics measurement. No learning.**

Question: *does downwash actually couple the drones?*

Two drones, one above the other, identical hover commands. Measure how much the
lower one sinks. The upper drone is the control — it is in clean air, so any
shared error cancels in the difference.

**Result:**

| Separation | Induced drift |
|---|---|
| 0.60 m | 0.46 m |
| 1.00 m | 0.22 m |
| 2.50 m | 0.038 m |

**12× contrast, smoothly decaying. Premise confirmed.**

> **A trap worth remembering.** The first sweep (0.10–1.50 m) came back *flat* —
> which looked like "no coupling". It was the opposite: at 0.15 m the force is
> **12.7× the drone's weight**, so the drone was in free fall at every point and
> the reading saturated. A ceiling effect masquerading as a null result. Always
> verify a null before acting on it.

### Phase 2 — Apparatus ✅ DONE

Built the task, the ladder, and the confound controls. Verify with:

```bash
python scripts/inspect_ladder.py
```

### Phase 3 — Pilot ⚠️ DECISION GATE

**The cheapest possible test of whether the effect exists.**

Run it as two stages:

#### Stage 1 — Budget check (~1 h)

One rung (`k1`), 3 seeds, long budget. **This is real training.**

Question: *does training keep improving, or has it flattened?*

This exists because the learning check plateaued around 12k steps then wobbled.
If training has not converged, all rungs look similar simply because **none of
them finished** — and that would be misread as "critic information doesn't
matter."

#### Stage 2 — Pilot (~6 h)

Three rungs (`k0`, `k1`, `k3`), 5 seeds. **Also real training.**

Question: *does `k` matter?*

#### Why Stage 1 makes Stage 2 interpretable

| Stage 2 shows | Stage 1 said | Conclusion |
|---|---|---|
| `k1` wins, CIs separate | — | **Claim 1 supported** → full sweep |
| Monotonic | Converged | **K1 fires** → pivot to Claim 3 |
| Monotonic | Still improving | **Do NOT fire K1** — run longer |
| Flat | Still improving | **K2** — budget too small |

Without Stage 1, a flat or monotonic result is ambiguous between "no effect"
and "not trained enough" — and the night would be wasted either way.

**Command:**

```bash
nohup python scripts/run_overnight.py \
  --stage1-steps 300000 --stage2-steps 400000 > overnight.log 2>&1 &
```

Read `results/<experiment_id>/<date>_<task_version>/REPORT.txt` in the morning.

---

## 9. Reading the output

### A sweep line

```
  [  1/6] k3     seed1   sp1.00  return     -2.74  (4.7 min)
           │      │       │             │
           │      │       │             └─ performance (higher = better)
           │      │       └─ formation spacing 1.00 m
           │      └─ random seed #1
           └─ rung: critic saw 3 teammates
```

### The summary table

```
  rung     n    mean         std        95% CI
  k0       2    -2.799       0.146      [-3.00, -2.60]
  k1       2    -2.766       0.047      [-2.83, -2.70]
  k3       2    -2.702       0.051      [-2.77, -2.63]
```

- **mean** — average across seeds
- **std** — how much seeds disagreed
- **95% CI** — the range the true value probably lies in

> **Overlapping CIs mean the rungs cannot be distinguished.** In the example
> above all three overlap heavily — **no conclusion is possible.**

### A learning curve line

```
step   9,216   eval  -2.20   explained_var  0.743   value_loss  0.614
       │             │                    │                   │
       │             │                    │                   └─ critic error
       │             │                    └─ how much return variance the
       │             │                       critic captures (want > 0.1)
       │             └─ deterministic-policy return
       └─ environment steps so far
```

**Why returns are positive:** since v2 the reward includes a per-step survival
bonus, so a full 720-step episode with small errors scores ~600–670 out of a
720 ceiling. **Higher is better.** (In v1 returns were negative penalties —
old logs will look different.)

**Why evaluation differs from training reward:** training uses a *stochastic*
policy (actions sampled for exploration), which depresses reward. Evaluation
uses the *deterministic* policy. **The evaluation number is what we report.**

---

## 10. Kill criteria

Decided in advance, before any emotional investment in a result.

| # | Condition | Action |
|---|---|---|
| **K1** | Curve flat or monotonic, adequately powered | Claim 1 dead → pivot to Claim 3 |
| **K2** | Task solved to ceiling, or budget too short | Not a finding — fix the setup |
| **K3** | Seed variance exceeds the between-rung effect | Underpowered — never report as real |
| **K4** | Claim 1 holds but `k*` ignores coupling | Claim 2 dead — weaker paper, still alive |
| **K5** | Someone publishes this first | Pivot toward Claims 3/4 |

> **The rule.** A negative result, honestly reported with adequate power, is
> still publishable: *"the field's assumption is correct, and here is the first
> evidence for it"* is a real contribution. What is **not** publishable is an
> underpowered ambiguous result dressed up as a finding.

### And a standing reminder

The pilot is **underpowered by design**. Five seeds decide whether the 16-hour
full sweep is worth running. Whatever it shows, **it is not a finding and must
never be reported as one.**

---

## Quick reference

```bash
# Verify install
python scripts/check_simulator.py

# Phase 1 — physics premise (no learning)
python scripts/run_downwash_probe.py

# Phase 2 — inspect the apparatus
python scripts/inspect_ladder.py

# Sanity: can the agent learn at all?
python scripts/check_learning.py --steps 30000

# Phase 3 — overnight staged run
nohup python scripts/run_overnight.py \
  --stage1-steps 300000 --stage2-steps 400000 > overnight.log 2>&1 &

# Tests
pytest tests/ -q
```

| Number | Value |
|---|---|
| Task version | **v4-correctness** |
| Actor input | 13-D, **always** (byte-identical across rungs) |
| Actor output | 3-D position offset (±0.2 m) via tanh-squashed Gaussian, not rotor RPMs |
| Critic input | **34-D at every rung** — real slots up to k, zeros beyond |
| Critic capacity | **84,929 parameters at every rung**, verified on real learners |
| Agents | 4 (leader + 3 followers), **chain** topology |
| Spacing | 1.0 m (Claim 2 sweeps 0.7–2.5 m) |
| Disturbance | OU gust field: mean 1.0 m/s, σ 4.0 m/s, τ 1.0 s, spatial corr 0.7 — unobserved by actors |
| Episode | 15 s @ 48 Hz control = 720 steps |
| Reported metrics | RMS formation error (cm), min separation (m), collision rate, downstream error amplification G — **not** return |

> **Return is no longer the headline.** It is a training variable whose meaning
> changed with every reward revision (three so far). The scientific results are
> reported in metres and rates — see [`../src/ctde/metrics.py`](../src/ctde/metrics.py)
> and Charter §S.11.
| Throughput | ~46 steps/sec/process, PyBullet-bound |
| Model validity floor | 0.6875 m — below this, extrapolated |
