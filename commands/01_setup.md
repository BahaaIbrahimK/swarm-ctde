# 01 — Setup (one time only)

Run these once. Afterwards you only need [`02_verify.md`](02_verify.md) and
the phase-specific files.

---

## Why Python 3.12 specifically

You hit this error:

```
ERROR: Ignored the following versions that require a different python version:
  ... 1.11.3 Requires-Python >=3.9,<3.13 ...
ERROR: Could not find a version that satisfies the requirement gym-pybullet-drones
```

Two separate problems were hiding in there:

**Problem 1 — your Python is too new.** Every scipy build was rejected for
requiring `<3.13`, which means your interpreter is 3.13 or newer. The
scientific-Python stack lags new releases by roughly a year.

**Problem 2 — `gym-pybullet-drones` is not on PyPI at all.** Note
`from versions: none` — that is different from a version conflict. The package
has never been published to PyPI; it installs from GitHub source only.

The official repository targets **Python 3.12**, so that is what we use.

---

## Step 1 — Check what Python you have

```bash
python3 --version
```

If it already says `Python 3.12.x`, skip to **Step 3**.

---

## Step 2 — Install Python 3.12

```bash
sudo apt update
sudo apt install -y software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y python3.12 python3.12-venv python3.12-dev
```

Confirm it worked:

```bash
python3.12 --version      # expect: Python 3.12.x
```

> **What `deadsnakes` is:** a well-known community PPA (package archive) that
> provides Python versions Ubuntu does not ship by default. Installing 3.12
> this way does **not** replace or break your system Python.

---

## Step 3 — Create the virtual environment

```bash
cd ~/projects/swarm-ctde
python3.12 -m venv .venv
source .venv/bin/activate
```

Your prompt should now start with `(.venv)`.

> **What a venv is and why it matters:** an isolated folder holding its own
> Python and its own packages. Without it, installing our dependencies could
> break other projects on your machine, and vice versa. It also means the exact
> package set is reproducible — which Charter §7 requires.

Confirm the venv is really the active Python:

```bash
python --version        # expect: Python 3.12.x
which python            # expect: .../swarm-ctde/.venv/bin/python
```

If `which python` does **not** point inside `.venv`, activation failed — do not
continue, see [`99_troubleshooting.md`](99_troubleshooting.md).

---

## Step 4 — Install the standard dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs numpy, scipy, PyYAML, pybullet, gymnasium, matplotlib, pandas,
and pytest.

> **Note:** `torch` and `ray` are deliberately commented out in
> `requirements.txt`. The Phase 1 probe contains no RL, and they are a large
> download. We enable them at the start of Phase 2.

---

## Step 5 — Install gym-pybullet-drones from source

This is the step that failed before. It is **not** a pip package.

```bash
# Clone next to the project (not inside it, so it stays out of our git repo)
git clone https://github.com/utiasDSL/gym-pybullet-drones.git ~/gym-pybullet-drones

# Install it in editable mode into our venv
pip install -e ~/gym-pybullet-drones
```

> **Why `-e` (editable):** the package is installed as a *link* to the source
> folder rather than a copy. If we ever need to inspect or patch the downwash
> model, we can edit the source directly and our code picks it up immediately.
> Given that our paper's central claim depends on that specific physics model,
> being able to read and verify it is valuable.

> **Why clone to `~` and not inside the project:** keeping a second git
> repository inside ours would confuse version control and pollute our commits.

---

## Step 6 — Confirm everything imports

```bash
python -c "
import numpy, scipy, yaml, matplotlib, pandas, pybullet, gymnasium
from gym_pybullet_drones.utils.enums import DroneModel, Physics
print('all imports OK')
print('downwash physics modes:', [p.value for p in Physics if 'dw' in p.value])
"
```

Expected output:

```
all imports OK
downwash physics modes: ['pyb_dw', 'pyb_gnd_drag_dw']
```

Seeing `pyb_gnd_drag_dw` in that list is the important part — it is the physics
mode our probe requires. If it is missing, the installed version is too old and
the probe cannot measure anything.

---

## Done

Setup is complete. Next: [`02_verify.md`](02_verify.md).

**Remember for every future session:**

```bash
cd ~/projects/swarm-ctde
source .venv/bin/activate
```
