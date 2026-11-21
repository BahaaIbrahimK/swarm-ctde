# 02 — Verify the installation

Run these after setup, and again any time you change dependencies.

```bash
cd ~/projects/swarm-ctde
source .venv/bin/activate
```

---

## Check 1 — Infrastructure tests (fast, no simulator)

```bash
pytest tests/ -v
```

**Expected:** all tests pass in about a second.

These test the config system and the seeding harness. They need no simulator,
which is why they run first — if the *foundations* are broken there is no point
debugging the physics.

**What they actually guard against:**

| Test | Catches |
|---|---|
| `test_deep_merge_preserves_sibling_keys` | Config override silently deleting other settings |
| `test_seeding_is_reproducible` | Same seed producing different results (breaks Charter §7) |
| `test_different_seeds_differ` | All 8 seeds being identical → meaningless confidence intervals |
| `test_named_streams_are_independent` | Wind randomness perturbing exploration randomness |
| `test_real_probe_config_loads` | A typo in `configs/*.yaml` reaching a long sweep |

These are infrastructure bugs that would **not crash anything** — the numbers
would just quietly be wrong. That is exactly why they are tested.

---

## Check 2 — Simulator imports

```bash
python -c "
from gym_pybullet_drones.utils.enums import DroneModel, Physics
print('DroneModel:', [d.value for d in DroneModel])
print('Physics   :', [p.value for p in Physics])
"
```

**Expected:** `Physics` includes `pyb_gnd_drag_dw`.

That specific mode enables ground effect + blade drag + **downwash**. Downwash
is the physical coupling our whole hypothesis rests on (Charter §1.5), so if it
is absent nothing downstream is testable.

---

## Check 3 — The simulator actually runs

```bash
python scripts/check_simulator.py
```

**Expected output:**

```
hover RPM  : 14468.43
drone 0 z  : 1.5000
drone 1 z  : 1.1995
step()     : returned 5 values (gymnasium 5-tuple)

simulator OK
```

> ### WARNING: use `CtrlAviary`, never `BaseAviary`
>
> `BaseAviary` is an **abstract base class**. Instantiating it directly fails:
>
> ```
> File ".../BaseAviary.py", line 1027, in _actionSpace
>     raise NotImplementedError
> NotImplementedError
> ```
>
> This is by design, not a bug. `BaseAviary` provides shared simulation
> machinery but leaves `_actionSpace()` / `_observationSpace()` for subclasses.
>
> `CtrlAviary` is the concrete subclass whose action space is direct rotor RPM
> — what we need, and what upstream `examples/downwash.py` uses.
>
> **API note:** current versions use `pyb_freq` + `ctrl_freq`, replacing the
> older single `freq` / `aggregate_phy_steps` arguments. The probe is already
> updated for this.

> **Tip — avoid pasting multi-line `python -c "..."` commands.** Terminals
> often mangle them (lines merge, quotes break), producing errors that look
> like code bugs but are paste bugs. Use the script files instead.

## Check 4 — Determinism (the Charter §7 promise)

```bash
python -c "
import sys; sys.path.insert(0, 'src')
from ctde.utils.seeding import seed_everything, make_rng
import numpy as np

seed_everything(42); a = np.random.rand(3)
seed_everything(42); b = np.random.rand(3)
assert np.array_equal(a, b), 'NOT reproducible'
print('same seed  -> same numbers   OK')

seed_everything(1); c = np.random.rand(3)
seed_everything(2); d = np.random.rand(3)
assert not np.allclose(c, d), 'seeds not independent'
print('diff seeds -> diff numbers   OK')

w = make_rng(0, 'wind'); p = make_rng(0, 'init_pose')
assert not np.allclose(w.random(3), p.random(3))
print('streams independent          OK')
"
```

This is the property that lets us claim "every figure is regenerable with one
command." It is worth confirming directly rather than trusting it.

---

## All four passed?

Proceed to [`03_phase1_downwash.md`](03_phase1_downwash.md).

Any failed? See [`99_troubleshooting.md`](99_troubleshooting.md).
