# Documentation

| File | What it is | When to read it |
|---|---|---|
| [`EXPERIMENT_DEFINITION.md`](EXPERIMENT_DEFINITION.md) | **Start here.** Glossary, the RL problem, the formation, the ladder, what we expect, how to read output | Before running anything; whenever a term or number is unclear |
| [`audit_2026-09-16.md`](audit_2026-09-16.md) | **Five defects that invalidated all prior results.** What broke, how it was measured, how each is now guarded | **Read before trusting any result**; before writing a test |
| [`task_v2_notes.md`](task_v2_notes.md) | **Why the task was redesigned.** Five bugs found in v1, how each was diagnosed and fixed, and the calibration tables | Before changing the task; when a result looks wrong |
| [`downwash_model_notes.md`](downwash_model_notes.md) | Phase 1 physics findings, including two limitations that must appear in the paper | When working on coupling, spacing, or Claim 2 |

## How this relates to the other docs

| Location | Purpose |
|---|---|
| [`../PROJECT_CHARTER.md`](../PROJECT_CHARTER.md) | **The research plan** — gap, claims, kill criteria, phases |
| `docs/` (here) | **Understanding** — what things mean, why choices were made |
| [`../commands/`](../commands/) | **Doing** — exact terminal commands, in order |

Rough rule: charter says *what and why*, docs say *what it means*, commands say
*how to run it*.

## Task versions

| Version | Status | Summary |
|---|---|---|
| `v1-raw4d` | ❌ **invalid** | Direct 4-D rotor control. Unlearnable — episodes ended after 20–100 of 720 steps even after 300k steps. Kept as evidence, never deleted. |
| `v2-pidassist` | ❌ **invalid** | RL guidance layer over a PID stack. Followers tracked an *analytical* leader trajectory, so execution was not honestly decentralized; steady wind and steady downwash are both cancelled by a PID (seed variance 0.00); wind used the wrong PyBullet frame and application point. |
| `v3-chain-gust` | ❌ **invalid** | Physics correct, learning machinery not. The September 2026 audit found six defects: GAE off by one and running through resets, dead crash penalty, no collision detection, privileged critic missing the gust, and unequal critic capacity. |
| `v4-correctness` | ✅ **current** | All six audit defects fixed, plus four reporting defects (Sep 16). Exact capacity control (34-D input, 84,929 params at every rung, verified on real learners); collision detection; bounded policy; value normalization; held-out evaluation seeds; physical metrics in metres. |

> **Never pool results across task versions.** A different task version is a
> different experiment. The version tag appears in every result path so the
> mistake is visible rather than silent.

---

## Findings recorded here that must reach the paper

Two items from `downwash_model_notes.md` are limitations a reviewer will
probe:

1. **The downwash model is extrapolated below 0.6875 m.** Its Gaussian width
   term goes negative there. It does not crash — the failure is silent — but
   the model is outside its calibration range. Formation spacings should stay
   above this, or the extrapolation must be disclosed.

2. **Coupling at close range is enormous.** At 0.15 m separation the downwash
   force is ~12.7× the drone's weight. This caused the first probe sweep to
   saturate and read *flat*, which looked like "no coupling" but was the
   opposite. Worth stating as a methodological note — and worth remembering as
   a general lesson: **verify a null result before acting on it.**
