# Commands — terminal reference

Every command you need, in the order you need it. Each file is a numbered step.

**All commands assume you are in the project root:**

```bash
cd ~/projects/swarm-ctde
```

---

## Files in this folder

| File | What it covers | When you use it |
|---|---|---|
| [`01_setup.md`](01_setup.md) | Python 3.12, venv, dependencies, gym-pybullet-drones | **Once**, at the start |
| [`02_verify.md`](02_verify.md) | Check the install actually works | After setup, and after any dependency change |
| [`03_phase1_downwash.md`](03_phase1_downwash.md) | Run the premise check | Phase 1 |
| [`04_phase2_apparatus.md`](04_phase2_apparatus.md) | Inspect the ladder + confound controls | Phase 2 |
| [`05_phase3_pilot.md`](05_phase3_pilot.md) | Training + the pilot sweep **decision gate** | Phase 3 |
| [`06_task_health.md`](06_task_health.md) | **Verify the task is sound before spending compute** | Before any sweep, and after any task change |
| [`99_troubleshooting.md`](99_troubleshooting.md) | Fixes for errors you will hit | When something breaks |

New files get added as we build later phases.

---

## The 60-second version

If you just want to get running and already have Python 3.12:

```bash
cd ~/projects/swarm-ctde

# One-time setup
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
git clone https://github.com/utiasDSL/gym-pybullet-drones.git ~/gym-pybullet-drones
pip install -e ~/gym-pybullet-drones

# Verify
pytest tests/ -v

# Run Phase 1
python scripts/run_downwash_probe.py
```

If any of that fails, work through [`01_setup.md`](01_setup.md) properly — it
explains each step and the failure modes.

---

## Daily workflow reminder

**Every new terminal session needs the environment activated:**

```bash
cd ~/projects/swarm-ctde
source .venv/bin/activate
```

Your prompt shows `(.venv)` when it is active. Forgetting this is the single
most common cause of "but it worked yesterday" errors — see
[`99_troubleshooting.md`](99_troubleshooting.md).
