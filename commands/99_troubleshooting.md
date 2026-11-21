# 99 — Troubleshooting

Errors you are likely to hit, and what they actually mean.

---

## `error: [Errno 2] No such file or directory: 'x86_64-linux-gnu-g++'`

**Cause:** no C++ compiler installed.

PyBullet is a C++ physics engine. There is no pre-built wheel for Python 3.12
on Linux, so pip must **compile it from source** — which needs a compiler.

```bash
sudo apt update
sudo apt install -y build-essential python3.12-dev
g++ --version          # confirm it now exists
```

`build-essential` provides `gcc`, `g++`, and `make`. `python3.12-dev` provides
the Python C headers that extensions compile against.

Then reinstall:

```bash
pip install -r requirements.txt
```

> **Expect a slow, noisy build.** Compiling PyBullet takes 3–10 minutes and
> prints hundreds of compiler lines. That is normal progress, not errors.

### Why this also broke numpy

pip installs a requirements file **atomically** — if any package fails, the
whole transaction rolls back. So one missing compiler left you with *nothing*
installed, and the confusing follow-on error:

```
ModuleNotFoundError: No module named 'numpy'
```

numpy was never the problem. Fix the compiler and the whole file installs.

---

## `pip list` shows conda packages / prompt shows `(.venv) (base)`

**Cause:** conda's `base` environment and your venv are both active, and conda
is winning.

```bash
which python        # if NOT inside .venv/, conda has priority
```

Fix for this session:

```bash
conda deactivate
source .venv/bin/activate
```

Fix permanently (recommended — removes a recurring source of confusion):

```bash
conda config --set auto_activate_base false
```

Takes effect in new terminals.

---

## `NotImplementedError` at `BaseAviary._actionSpace()`

**Cause:** `BaseAviary` is an **abstract base class**. It cannot be
instantiated directly.

```
File ".../BaseAviary.py", line 1027, in _actionSpace
    raise NotImplementedError
```

This is by design. `BaseAviary` supplies shared simulation machinery but leaves
`_actionSpace()` / `_observationSpace()` for subclasses to define.

**Fix:** use `CtrlAviary` — the concrete subclass whose action space is direct
rotor RPM (what upstream `examples/downwash.py` uses):

```python
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary

env = CtrlAviary(
    drone_model=DroneModel.CF2X,
    num_drones=2,
    initial_xyzs=np.array([[0, 0, 1.5], [0, 0, 1.2]]),
    physics=Physics.PYB_GND_DRAG_DW,
    pyb_freq=240,       # NOT `freq` -- API changed
    ctrl_freq=240,
    gui=False,
)
```

Or simply run the ready-made check:

```bash
python scripts/check_simulator.py
```

> Our probe already uses `CtrlAviary`. If you hit this error, you are running
> an **old snippet** from an earlier version of the docs.

---

## Pasted command produces garbled text like `"rint('simulator OK')env._get...`

**Cause:** the terminal mangled a multi-line paste — lines merged, quotes
broke. The resulting error looks like a code bug but is a paste bug.

**Fix:** use the script files rather than pasting multi-line
`python -c "..."` blocks:

```bash
python scripts/check_simulator.py
python scripts/run_downwash_probe.py
pytest tests/ -v
```

All are single-line commands and paste safely.

---

## `ModuleNotFoundError: No module named 'ctde'`

**Cause:** running from the wrong directory. The scripts add `src/` to the
import path relative to the project root.

```bash
cd ~/projects/swarm-ctde      # must be here
python scripts/run_downwash_probe.py
```

---

## `ModuleNotFoundError: No module named 'yaml'` (or numpy, pybullet, ...)

**Cause (99% of the time):** the virtual environment is not activated.

```bash
which python
```

If that does **not** print a path inside `.venv/`, activate it:

```bash
source .venv/bin/activate
```

**This is the most common error in the project.** Every new terminal session
needs activation. Your prompt shows `(.venv)` when it is active.

---

## `ERROR: Could not find a version that satisfies the requirement gym-pybullet-drones`

**Cause:** it is not on PyPI. It never has been.

Look at the error text — `from versions: none` means "this package does not
exist here", not "no compatible version". Install from GitHub instead:

```bash
git clone https://github.com/utiasDSL/gym-pybullet-drones.git ~/gym-pybullet-drones
pip install -e ~/gym-pybullet-drones
```

See [`01_setup.md`](01_setup.md) Step 5.

---

## `ERROR: Ignored the following versions that require a different python version`

**Cause:** your Python is newer than the scientific stack supports. If you see
many packages rejected for `<3.13`, you are on 3.13+.

```bash
python --version
```

Fix: build the venv with Python 3.12 — [`01_setup.md`](01_setup.md) Steps 2–3.

> The scientific-Python ecosystem typically lags new Python releases by about a
> year. Using the newest interpreter is not an advantage here.

---

## `python3.12 --version` works, but `python --version` says 3.13

**Cause:** the `.venv` was created *before* Python 3.12 was installed, so it is
locked to 3.13. **A venv cannot change its Python version** — it must be
rebuilt.

Diagnose:

```bash
python3.12 --version     # 3.12.x  -- 3.12 IS installed
python --version         # 3.13.x  -- but the venv is on 3.13
which python             # .../.venv/bin/python  -- venv IS active
```

All three can be true at once. The venv is active and working; it is simply
built on the wrong interpreter.

**Fix — delete and rebuild:**

```bash
cd ~/projects/swarm-ctde

deactivate                    # leave the old venv (ignore "not found")
rm -rf .venv                  # safe: contains no project code

python3.12 -m venv .venv      # rebuild on 3.12 explicitly
source .venv/bin/activate

python --version              # must now print 3.12.x
```

Then reinstall:

```bash
pip install --upgrade pip
pip install -r requirements.txt
git clone https://github.com/utiasDSL/gym-pybullet-drones.git ~/gym-pybullet-drones
pip install -e ~/gym-pybullet-drones
```

> **Why deleting `.venv` is safe:** it holds only installed dependencies, never
> your own code. Everything in `src/`, `configs/`, `scripts/`, and `tests/` is
> untouched. Rebuilding costs a few minutes of download.

> **Note on `(base)` in your prompt:** if you see both `(.venv)` and `(base)`,
> conda's base environment is also active. That is harmless as long as
> `which python` points inside `.venv` — the venv takes precedence. To silence
> conda permanently: `conda config --set auto_activate_base false`.

---

## `ValueError: physics.mode is 'pyb', which does not include downwash`

**Cause:** this is our own guard firing, and it is doing its job.

Without downwash the probe measures zeros, and you would wrongly conclude the
premise failed when in fact the physics was simply switched off. Fix in
`configs/base.yaml`:

```yaml
physics:
  mode: "pyb_gnd_drag_dw"
```

---

## `TypeError: BaseAviary.__init__() got an unexpected keyword argument 'freq'`

**Cause:** API drift. Newer gym-pybullet-drones renamed constructor arguments
(`freq` → `pyb_freq`, plus a separate `ctrl_freq`).

Check what your installed version expects:

```bash
python -c "
import inspect
from gym_pybullet_drones.envs.BaseAviary import BaseAviary
print(inspect.signature(BaseAviary.__init__))
"
```

Send me that output — it is a one-line fix in `_make_env()` inside
`src/ctde/physics/downwash_probe.py`.

---

## `ValueError: vertical_separation_m=... puts the lower drone at ... m`

**Cause:** our guard against **ground effect** contamination.

Ground effect is a *different* aerodynamic phenomenon (extra lift near the
floor). If the lower drone flies too close to the ground, we would be measuring
downwash *plus* ground effect and could not separate them.

Fix: raise `upper_altitude_m` in `_make_env()`, or drop the largest separation
from the config.

---

## Probe runs but every measurement is ~0.0

Work through these in order:

1. **Is downwash physics on?**
   ```bash
   grep mode configs/base.yaml       # expect pyb_gnd_drag_dw
   ```
2. **Are the drones close enough?** Try a very tight spacing:
   ```yaml
   vertical_separations_m: [0.08, 0.10, 0.12]
   ```
3. **Is the settle window long enough?** The wake needs time to develop.
   Raise `settle_steps` to 480 (2 s).
4. **Compare against the upstream example**, which is known to produce visible
   downwash:
   ```bash
   python ~/gym-pybullet-drones/gym_pybullet_drones/examples/downwash.py
   ```
   If that shows an effect and ours does not, the bug is in our probe — send me
   both outputs.

If after all four the effect is genuinely absent, that is a real finding about
the simulator, and we discuss the pivot options in Charter §11 rather than
forcing the result.

---

## matplotlib fails / no display

```bash
python scripts/run_downwash_probe.py --no-plot
```

The CSV is written *before* plotting is attempted, so your data is safe. The
plot can be regenerated from the CSV later.

> Our plotting code already forces the non-interactive `Agg` backend, so this
> should be rare — but the flag exists as a safety valve.

---

## PyBullet prints lots of startup text

Normal. PyBullet is chatty on initialisation. Not an error.

---

## Still stuck?

Send me:

1. The **full** error message (not just the last line)
2. `python --version` and `which python`
3. `pip list | grep -Ei "numpy|pybullet|gymnasium|drones"`

The full traceback matters — the last line usually names the symptom, while
the actual cause sits several frames above it.
