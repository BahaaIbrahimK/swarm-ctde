# 05 — Phase 3: training and the pilot sweep ⚠️ DECISION GATE

```bash
cd ~/projects/swarm-ctde
source .venv/bin/activate
```

---

## What Phase 3 is for

**This is the gate that decides whether the project continues as planned.**

The pilot runs three rungs with a reduced budget and asks one question: does
Claim 1's effect exist at all? If the curve is flat or monotonic,
kill-criterion K1 fires and we pivot to Claim 3 — in week 6, having spent days,
rather than in month 5 having spent months.

---

## Run the checks in this order

### Step 0 — Task health ⚠️ **do not skip** (~25 min)

```bash
python scripts/verify_task_fixes.py      # ~3 min
python scripts/check_learning.py --steps 30000   # ~20 min
```

**See [`06_task_health.md`](06_task_health.md) for what these do and how to read
them.**

Five bugs were found in the task design, four invisible without specific
measurement. The worst — wind being sampled but never applied — would have
produced a flat sweep and **silently faked kill-criterion K1**, abandoning the
project's primary claim over a missing function call.

These checks cost 25 minutes. The sweep costs 7 hours. Run them.

### Step 1 — Pipeline smoke test (~5 min)

```bash
python scripts/run_pilot_sweep.py --smoke --workers 6
```

Verifies the machinery runs end to end: environment → learner → parallel sweep
→ aggregation → artifacts.

> **This deliberately issues NO verdict.** At 2,048 steps no policy has learned
> anything, so all rungs sit at untrained-network noise. Comparing those means
> would produce a confident-looking but entirely spurious result — and firing
> K1 on noise would abandon the primary claim for no reason. The script refuses
> to give a verdict below 20,000 steps.

### Step 2 — The overnight staged run (~7 h on 6 cores) ← **recommended**

```bash
nohup python scripts/run_overnight.py \
  --stage1-steps 300000 --stage2-steps 400000 > overnight.log 2>&1 &
```

Two stages, and the first makes the second interpretable:

| Stage | What | Question |
|---|---|---|
| **1** (~1 h) | k1 only, 3 seeds, 300k | Is the budget long enough? Has training converged? |
| **2** (~6 h) | k0/k1/k3, 5 seeds, 400k | **Does `k` matter?** |

Without Stage 1, a flat Stage 2 is ambiguous between "no effect" (K1) and "not
trained long enough" (K2) — and the night is wasted either way.

Read `results/<experiment_id>/<date>_<task_version>/REPORT.txt` in the morning. Its verdict is
**qualified by what Stage 1 found**: if training had not converged, the report
explicitly says *not* to fire K1.

### Alternative — pilot only (~1.5 h)

```bash
python scripts/run_pilot_sweep.py --steps 100000 --seeds 5 --workers 6
```

Faster, but gives no budget check, so a flat result cannot be interpreted.

---

## How to read the pilot

Three rungs distinguish the two competing hypotheses:

```
MONOTONIC (the field is right):     k0  <  k1  <  k3
INTERIOR OPTIMUM (Claim 1):         k1  >  both k0 and k3
```

**The interior point must beat BOTH endpoints.** Beating only `k0` would merely
re-show that centralization helps — known since MADDPG (2017).

| Shape | Meaning | Action |
|---|---|---|
| Interior optimum, CIs separate | Consistent with Claim 1 | Proceed to Phase 4 |
| Interior optimum, CIs overlap | Suggestive only | Proceed; the full sweep resolves it |
| Monotonic increasing | Field's assumption holds | **K1 fires** — pivot to Claim 3 |
| Flat | Task saturated or budget too small | Check **K2** before concluding |

> **The pilot is underpowered by design.** Five seeds decide whether the full
> sweep is worth running. It is not a finding and must never be reported as one.

---

## Compute budget

Measured on this machine: **~46 env-steps/sec per process**, PyBullet-bound.

| Sweep | Runs | 6 cores |
|---|---|---|
| Pilot (3 rungs × 5 seeds, 100k) | 15 | **~1.5 h** |
| Full (5 rungs × 8 seeds, 100k) | 40 | **~4 h** |
| Claim 2 (5 × 8 × 4 spacings) | 160 | **~16 h** |

At a 300k budget, multiply by 3. The Claim 2 grid at 300k is ~48 h — worth
planning as an overnight-plus run, or trimming the spacing count.

Runs are fully independent, so wall-clock divides by worker count. This is the
Charter §13 argument cashing out: we are bottlenecked by **seed count**, not
steps/second, and the workload is embarrassingly parallel.

---

## What Phase 3 added

| File | Role |
|---|---|
| `src/ctde/training/buffer.py` | Rollout storage + GAE |
| `src/ctde/training/mappo.py` | The learner, parameterised by `k` |
| `src/ctde/training/trainer.py` | One run = one (rung, seed) pair |
| `src/ctde/training/sweep.py` | Parallel grid + aggregation with CIs |

### Why MAPPO rather than SAC/TD3/DDPG

The previous project's off-policy centralized critic was broken two ways:
RLlib's SAC/DDPG never called the custom `value_function()`, so it was dead
code receiving no gradients; and actions written into observations went stale
in the replay buffer.

MAPPO avoids both structurally — on-policy data is used once while fresh, and
the critic is an explicit `V(s)` we write ourselves. Charter §9: we deliberately
do **not** claim algorithm superiority. Algorithm is held fixed so `k` is the
only variable. One correct algorithm beats five broken ones.

### The test that prevents the old failure

```bash
pytest tests/test_phase3.py::test_critic_receives_gradients -v
```

Asserts that both the critic's projection and trunk receive non-zero gradients.
The previous project's dead critic produced plausible-looking, meaningless
numbers — this makes that failure mode impossible to reintroduce silently.

---

## Full test suite

```bash
pytest tests/ -q          # 40 tests, ~9 s
```
