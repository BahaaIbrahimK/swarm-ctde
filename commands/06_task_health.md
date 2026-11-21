# 06 — Task health checks ⚠️ run before any sweep

```bash
cd ~/projects/swarm-ctde
source .venv/bin/activate
```

---

## Why this file exists

**Five bugs were found in the task design, four of them invisible until
specific measurement.** Each would have produced a plausible-looking but
meaningless result:

| Bug | Symptom | What it would have caused |
|---|---|---|
| Task unlearnable | Episodes ended at 3–14% | Flat curve, misread as K1 |
| Reward paid for crashing | Return rose 59%, episodes stayed short | "Learning" that was dying faster |
| Sidestep exploit | (caught in review) | Agents delete the coupling → flat curve |
| **Wind never applied** | PID scored 598.6 ± 0.00 at *every* wind level | Nothing for the critic to explain → **fake K1** |
| Exploration too large | Return degraded 588 → 138 | Policy unlearns from the PID ceiling |

The common thread: **a design that was reasoned about but never measured.**
These scripts exist so that cannot happen silently again.

> **Never skip these to save time.** The overnight sweep costs ~7 hours. These
> checks cost ~25 minutes and have already prevented two experiments that would
> have produced garbage.

---

## Step 1 — Verify the task's incentives (~3 min)

```bash
python scripts/verify_task_fixes.py
```

Four checks, each testing a specific way the task could be silently broken:

| Check | Question | Expected |
|---|---|---|
| **PID assist** | Can drones fly with *no* policy input? | 722/720 steps, no crash |
| **Survival shaping** | Is dying ever better than flying? | ordering: good > bad > crash |
| **Sidestep exploit** | Does dodging the wake pay? | never, at any offset |
| **Coupling knob** | Does spacing change the physics? | 12.7× contrast, monotonic |

All four must PASS. If any fails, **do not train** — the results would be
uninterpretable.

### Reading Check 1

Zero action means "aim exactly at my slot." If the PID cannot hold formation
with a null action, no policy can learn anything on top of it. This is the
floor, and **v1 failed here** — drones fell within ~1 second.

### Reading Check 2

Three scenarios must order as **good flying > bad flying > crashing**. If
crashing wins, v1's fatal bug is back: a per-step penalty with early
termination makes dying cheaper than flying badly.

---

## Step 2 — Verify the agent can actually learn (~20 min)

```bash
python scripts/check_learning.py --steps 30000
```

One rung, one seed. Asks the question the pilot sweep *cannot*: **can the agent
learn this task at all?**

### The four pass criteria

| Metric | Requirement | Why |
|---|---|---|
| **Episode completion** | **> 80%** | **The honest signal.** v1 showed 3–14% while return "improved" |
| Return trend | rising | Must beat the pure-PID baseline (~520 at 12 m/s wind) |
| Explained variance | > 0.1 | Critic must predict returns, not guess the mean |
| Crash rate | ~0 | No termination effects contaminating the reward |

### Why completion is the criterion that matters

In v1, return rose 59% while episodes stayed at 20–100 of 720 steps. **The
policy was learning to crash faster** — fewer steps meant fewer accumulated
penalties.

Rising return alone is *not* evidence of learning. This script originally used
return as its pass criterion and missed the bug it was written to catch.

### What a healthy curve looks like

```
step  3,072   eval  591.93   ← starts near the PID baseline
step 15,104   eval  621.03
step 30,208   eval  664.01   ← climbing, no degradation
```

Return rising **above** pure-PID (~520 with wind) means the policy is learning
genuine formation strategy rather than merely not breaking.

### What failure looks like

| Pattern | Cause | Fix |
|---|---|---|
| Completion < 80% | Task unlearnable, or crash-rewarding reward | Re-run Step 1 |
| Return *degrades* from a high start | PID already solves it, exploration only hurts | Lower `log_std` / `max_offset` |
| Return flat, explained variance ~0 | Nothing to learn — no disturbance | Check wind is applied and calibrated |
| Explained variance wildly unstable | Critic learning rate too high | Lower `lr_critic` (currently 1e-3) |

---

## Step 3 — Only now, the sweep

```bash
nohup python scripts/run_overnight.py \
  --stage1-steps 300000 --stage2-steps 400000 > overnight.log 2>&1 &
```

See [`05_phase3_pilot.md`](05_phase3_pilot.md).

---

## Diagnostics logged during every run

Since v2, each evaluation records these so failures are visible *during*
training rather than discovered afterwards:

| Field | Catches |
|---|---|
| `eval_episode_len` | the v1 "learning to die faster" mode |
| `eval_crash_rate` | termination effects contaminating reward |
| `eval_lateral_error_m` | **the sidestep exploit in trained policies** |
| `eval_coupling_N` | whether coupling survives training |
| `explained_variance` | whether the critic is predictive and stable |

`eval_lateral_error_m` deserves particular attention. The analytic check in
Step 1 proves sidestepping *shouldn't* pay; this measures whether trained
policies actually stay on-axis. **If lateral error grows during training, the
exploit is back and results are invalid** — the coupling the paper studies
would have been engineered away.

---

## After changing anything about the task

Reward, action space, termination, or control mode:

1. Bump `TASK_VERSION` in `src/ctde/utils/run_naming.py`
2. Re-run Steps 1 and 2
3. Mark superseded results with `mark_status(dir, "superseded", reason)`

**Never delete a failed run.** It is the evidence for *why* the design changed.
The v1 results are kept and marked `invalid` for exactly this reason.
