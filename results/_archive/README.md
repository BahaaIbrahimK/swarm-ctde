# Archived results — ALL INVALID

**No number in this directory may appear in the paper.**

These runs were produced under task versions with defects that were later found
and fixed. They are kept, not deleted, because a failed run is evidence of *why*
the design changed — deleting it destroys that record and invites repeating the
mistake.

| Directory | Task version | Why invalid |
|---|---|---|
| `20260829_023030` | `v2-pidassist` | Stage 1. Analytical leader trajectory (execution not honestly decentralized); steady disturbances cancelled by the PID (seed variance 0.00); wind in the wrong frame. |
| `20260830_010651` | `v2-pidassist` | Stage 2. Same defects. |
| `20260831_202053` | `v3-chain-gust` | Stage 2. Physics correct, learning machinery not — six audit defects. **The "interior optimum" reported here is not trustworthy.** |
| `20260829_173958` | unknown | No `run_config.json`; aborted/incomplete. |
| `20260916_learnability_gust4.0_FAILED` | `v4-correctness` | Learnability check that FAILED — because the **task** was unsurvivable, not the learner. See below. |

Each directory carries its own `README.txt` banner and `MANIFEST.json` with the
full reason.

## The specific trap

`20260831_202053` is the dangerous one. It *looks* like a result — it shows an
interior optimum, which is precisely what Claim 1 predicts. It was produced by a
learner whose GAE was off by one and ran straight through episode resets, whose
crash penalty was dead code, and whose critics did not have equal capacity
across rungs. The shape may well be real. It is not *evidence* that it is real.

Re-running it under `v4-correctness` is a scheduled experiment, not a formality.

## The stale-calibration failure (16 Sep 2026)

`20260916_learnability_gust4.0_FAILED` is worth reading before designing any
new condition.

The v4 learnability check failed at 72.9% episode completion, which reads like
"the agent cannot learn". It was not. A **pure PID with zero RL action** at the
same gust strength crashes 40% of the time and completes 73% — almost exactly
the same number. Nothing could have learned that task.

The cause was a calibration measured against a bug. `gust_std_ms = 4.0` had been
chosen as "best variance, still 0% crashes" while wind reached the drones on 1
of 5 physics substeps — about 20% strength. v4 fixed the substep bug, so the
real disturbance became roughly 5× what the calibration described, and the
number was never re-measured.

Recalibrated on the pure PID under the corrected wind:

| σ | seed std | crashes | completion |
|---|---|---|---|
| 2.0 | 23.9 | 0% | 100% |
| 2.5 | 37.3 | 0% | 100% |
| **3.0** | **53.6** | **0%** | **100%** ← chosen |
| 3.5 | 73.5 | 0% | 100% |
| 4.0 | — | **40%** | 73% |

**The transferable lesson:** when a fix changes the magnitude of a physical
quantity, every number calibrated against the old magnitude is stale — including
ones that were carefully measured and correctly recorded at the time. And when a
learner fails, check whether the *task* is solvable at all before touching the
learner. A pure-PID probe costs minutes and would have caught this immediately.

## Valid results live elsewhere

```
results/<experiment_id>/<YYYY-MM-DD>_<task_version>/
```

The task version is in the path, so pooling across versions requires typing two
different directory names — the mistake is visible rather than silent. See
`src/ctde/utils/run_naming.py`.
