#!/usr/bin/env python3
"""
================================================================================
 SIMULATOR SMOKE TEST
================================================================================

Verifies that gym-pybullet-drones is installed and usable, and that the
downwash physics our project depends on is available.

USAGE
-----
    python scripts/check_simulator.py

WHY THIS IS A SCRIPT AND NOT A `python -c "..."` ONE-LINER
----------------------------------------------------------
Multi-line commands pasted into a terminal frequently get mangled -- lines
merge, quotes break, and the resulting error looks like a code bug rather than
a paste bug. A script file always runs the same way.

IMPORTANT: CtrlAviary, NOT BaseAviary
-------------------------------------
`BaseAviary` is an ABSTRACT base class. Instantiating it directly raises:

    File ".../BaseAviary.py", line 1027, in _actionSpace
        raise NotImplementedError

That is by design, not a bug. BaseAviary provides shared simulation machinery
but leaves `_actionSpace()` / `_observationSpace()` for subclasses to define.

`CtrlAviary` is the concrete subclass whose action space is direct rotor RPM --
exactly what we need (the probe commands both drones to hold hover RPM), and
what the upstream `examples/downwash.py` uses.
================================================================================
"""

from __future__ import annotations

import numpy as np


def main() -> int:
    # ---- 1. Enums -----------------------------------------------------------
    from gym_pybullet_drones.utils.enums import DroneModel, Physics

    print("DroneModel :", [d.value for d in DroneModel])
    print("Physics    :", [p.value for p in Physics])

    downwash_modes = [p.value for p in Physics if "dw" in p.value]
    print("downwash   :", downwash_modes)

    if "pyb_gnd_drag_dw" not in downwash_modes:
        print("\n[FAIL] 'pyb_gnd_drag_dw' not available.")
        print("       Downwash is the physical coupling our hypothesis needs")
        print("       (Charter §1.5). Without it nothing downstream is testable.")
        return 1

    # ---- 2. Build a 2-drone environment -------------------------------------
    from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary

    env = CtrlAviary(
        drone_model=DroneModel.CF2X,
        num_drones=2,
        initial_xyzs=np.array([[0, 0, 1.5], [0, 0, 1.2]]),
        physics=Physics.PYB_GND_DRAG_DW,
        # Current API: `freq` was split into pyb_freq (physics rate) and
        # ctrl_freq (action rate). Equal values => one step == one physics step.
        pyb_freq=240,
        ctrl_freq=240,
        gui=False,
    )

    try:
        # Gymnasium API: reset() returns (obs, info)
        env.reset()

        print()
        print(f"hover RPM  : {env.HOVER_RPM:.2f}")
        print(f"drone 0 z  : {env._getDroneStateVector(0)[2]:.4f}")
        print(f"drone 1 z  : {env._getDroneStateVector(1)[2]:.4f}")

        # ---- 3. Confirm stepping works --------------------------------------
        action = np.array([[env.HOVER_RPM] * 4] * 2, dtype=np.float32)
        out = env.step(action)
        print(f"step()     : returned {len(out)} values (gymnasium 5-tuple)")

        print()
        print("simulator OK")
        return 0

    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
