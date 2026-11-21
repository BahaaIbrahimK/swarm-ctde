# Setup on a new machine

**Written for: anyone setting up this project on a fresh machine.**

Read this before running anything. The fast path is one command; the rest of
this file is the failure modes, because every one listed here actually happened
on the original machine.

---

## Fast path

```bash
cd ~/projects/swarm-ctde
bash setup_new_machine.sh
```

The script is idempotent. It ends by running 100 tests. **If they pass, the
environment matches the machine that produced every recorded result.** If they
fail, nothing in this project should be trusted until they don't.

Then read `HANDOFF.md`.

---

## Target environment

| | |
|---|---|
| OS | Windows + **WSL2 (Ubuntu)** — see below |
| Python | **3.12 exactly** (3.12.3 on the original machine) |
| Compute | **CPU only** — no GPU was ever used |
| Toolchain | `build-essential` (pybullet compiles on install) |
| Simulator | `gym-pybullet-drones` @ commit `90c97ef3d97ce5de90ca720537707ff1ee97efb8` |
| Project size | ~35 MB (excluding `.venv`, which is 5.3 GB and must be rebuilt) |

### Why WSL rather than native Windows

The code itself is portable — there are **no hardcoded paths and no platform
checks** anywhere in `src/`, `scripts/`, or `tests/` (verified by grep). So
native Windows is not impossible. WSL is still strongly preferred:

1. **The dependency lock was generated on Linux.** On Windows some pins resolve
   to different builds. "Different numpy build" is exactly the class of thing
   that makes results stop matching, and this project has already been burned
   three times by environment drift.
2. **`pybullet` and `gym-pybullet-drones` compile from source.** On WSL that is
   `apt install build-essential`. On Windows it means Visual Studio Build Tools
   — a much larger detour, and a missing compiler was one of the original
   setup's failures.
3. **Everything assumes a POSIX shell** — `commands/`, the scripts, every path
   in the docs. PowerShell would need all of it rewritten.

Install WSL from PowerShell (Administrator):

```powershell
wsl --install -d Ubuntu
```

Then work **inside** the Linux filesystem (`~/projects/...`), not across
`/mnt/c/`. Running from `/mnt/c` is several times slower for the many small
file reads this project does, and file permissions behave differently.

### Ubuntu prerequisites

```bash
sudo apt update
sudo apt install -y python3.12 python3.12-venv python3.12-dev build-essential git
```

If `apt` has no `python3.12` (Ubuntu older than 24.04):

```bash
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y python3.12 python3.12-venv python3.12-dev
```

---

## Two deliberate differences from the original lock file

`requirements.lock.txt` is kept for the historical record. **Install from
`requirements.cpu.txt` instead.** The original would fail or waste ~2.5 GB:

**1. It pinned the CUDA build of PyTorch.** `torch==2.13.0+cu130` plus 19
`nvidia-*` packages. But `torch.cuda.is_available()` was **False** on the
machine that produced every result, and the training code runs with
`device="cpu"`. On a laptop without an NVIDIA card those wheels are useless and
may refuse to install.

**2. It used an editable install at a hardcoded path.** The instructions cloned
to `~/gym-pybullet-drones` and pinned `-e git+...`. A different username breaks
it. `requirements.cpu.txt` installs the same commit non-editably.

Same versions otherwise. 38 packages.

---

## Why the commit pin is not optional

`gym-pybullet-drones` is **not on PyPI**, so pip cannot pin it normally. Two
things in it directly affect results:

- **The downwash model** — the aerodynamic coupling this entire paper studies.
- **`BaseAviary.step()` counter ordering** — it increments `step_counter`
  *after* computing observation, reward, terminated and truncated. The project
  reads simulation time in those callbacks, and getting this wrong produced
  episodes of 722 steps instead of 720, with "episode completion" reported
  above 100%. See `docs/review_fixes_2026-09-17.md` (R5).

A different commit means different numbers, silently.

---

## Known failure modes

| Symptom | Cause | Fix |
|---|---|---|
| `No matching distribution found for torch==2.13.0` | pip is hitting PyPI, not the CPU index | use the `--index-url https://download.pytorch.org/whl/cpu` line from the script |
| Wall of C/C++ errors during install | no compiler | `sudo apt install -y build-essential` |
| `error: command 'gcc' failed` on 3.13 | wheels have no 3.13 build, falls back to source | use Python **3.12** |
| `BaseAviary ... NotImplementedError` | instantiating the abstract base class | the code uses `CtrlAviary`; this means a wrong simulator version |
| `TypeError: ... unexpected keyword 'freq'` | API drift between commits | wrong commit — check the pin |
| Tests pass but numbers differ from the docs | almost always the commit | verify the pin before anything else |
| Load average huge, runs never finish | thread oversubscription | `export OMP_NUM_THREADS=1` before training (cost 13 hours once) |

### Always set this before training

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

The original machine had 8 cores. Six parallel workers × 4 torch threads each
produced a load average of 18.4 and **zero completed runs in 13 hours**.

---

## Verifying by hand

```bash
source .venv/bin/activate

python --version                      # want 3.12.x
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
                                      # want 2.13.0 False

# Config should read: v4-correctness / k0,k1,k2,k2+P / 2.5 / 8 / 24
python - <<'PY'
import sys; sys.path.insert(0,"src")
from ctde.models.observability import build_ladder
from ctde.training.trainer import RunConfig
from ctde.utils.run_naming import TASK_VERSION
c = RunConfig(level=build_ladder(4)[0])
print(TASK_VERSION, [l.label for l in build_ladder(4)],
      c.gust_std_ms, c.eval_episodes, c.final_eval_episodes)
PY

python -m pytest tests/ -q            # want 100 passed
python -m pytest tests/ -q -m "not slow"   # faster subset while iterating
```

---

## Project layout

```
src/ctde/
  envs/formation_aviary.py   the task: chain formation, gusts, downwash, reward
  envs/coupling.py           downwash model + physics-based neighbour ranking
  models/observability.py    THE LADDER -- k0..k2+P, and what each rung may see
  models/networks.py         actor + critic (exact capacity control)
  training/mappo.py          MAPPO/PPO learner
  training/trainer.py        one run; RunConfig holds every calibrated constant
  training/sweep.py          the grid; produces the reported numbers
  metrics.py                 physical metrics (cm, m, rates) -- what the paper reports
  baselines.py               reference controllers the learner must beat

tests/                       100 tests. Rule: instantiate the REAL objects.
docs/                        why every decision was made -- read before changing anything
results/_archive/            invalid runs, KEPT AS EVIDENCE. Read its README.
commands/                    terminal commands per stage
```

### Read these, in order

1. **`HANDOFF.md`** — where it stopped and the open decision
2. `docs/review_fixes_2026-09-17.md` — the most recent and most important audit
3. `docs/learnability_v4.md` — why training was halted
4. `PROJECT_CHARTER.md` §S.* — status sections, newest first
5. `results/_archive/README.md` — why old results are invalid

---

## Standing rules for working on this project

These were established over the course of the project and are not negotiable
defaults:

- **Plan before editing.** State what will change, which files, and how it will
  be tested.
- **Tests must instantiate the real objects** the experiment uses — `Trainer`,
  `MAPPO`, `FormationAviary` — never hand-built substitutes. Four separate
  defects in this project were correct fixes that never reached the running
  code, and isolated tests passed through all of them.
- **Verify a test fails before trusting it.** One guard test here passed with
  the bug deliberately reintroduced; it was matching an unrelated comment.
- **Never calibrate on a small sample.** Three calibrations were set on 12–24
  episodes and all three were wrong (see `HANDOFF.md`). Use ≥100 episodes and
  the rule of three: 0 failures in *n* trials only bounds the rate below ~3/*n*.
- **Update the docs after each fix round**, not at the end.
- **Do not pool results across task versions.** The version is in the result
  path so the mistake has to be typed deliberately.
