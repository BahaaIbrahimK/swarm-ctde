"""
================================================================================
 CONFIGURATION SYSTEM
================================================================================

WHY THIS FILE EXISTS
--------------------
The previous project configured experiments through argparse flags:

    python multiagent.py --num_drones 2 --env leaderfollower --act one_d_rpm ...

That approach breaks down badly for THIS project. We will run roughly

    8 seeds x 6 observability levels x 4 formation spacings ~= 200 runs

If settings live only in shell commands, then six months from now nobody can
answer "what exactly produced figure 3?". Charter §7 demands every figure be
regenerable with one command -- that requires settings to live in FILES that
are versioned in git, not in shell history.

THE DESIGN
----------
  * A config is a YAML file (human-readable text).
  * Configs can INHERIT from a base config via `_base_`, so shared settings are
    written once. Only the differences are stated in each experiment file.
  * Every loaded config is validated and then SAVED alongside its results, so
    each result carries the exact settings that produced it.

WHY YAML RATHER THAN PYTHON FILES?
----------------------------------
YAML is data, not code. A Python config file can contain arbitrary logic, which
means "what were the settings?" may have no fixed answer. YAML always has one.
================================================================================
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


# ==============================================================================
# SECTION 1 : LOADING WITH INHERITANCE
# ==============================================================================

def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config, resolving `_base_` inheritance.

    Inheritance lets us write shared settings once. Example:

        # configs/base.yaml
        physics:
          sim_freq_hz: 240
        drone:
          model: cf2x

        # configs/probe_downwash.yaml
        _base_: base.yaml          # <-- inherit everything above
        probe:
          vertical_separations_m: [0.1, 0.2, 0.4]

    The result is the base config with the child's values merged on top.

    Parameters
    ----------
    path:
        Path to the YAML file.

    Returns
    -------
    dict
        The fully resolved configuration.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    # ---- Resolve inheritance ------------------------------------------------
    base_ref = cfg.pop("_base_", None)
    if base_ref is not None:
        # `_base_` is interpreted relative to the child config's directory.
        base_path = (path.parent / base_ref).resolve()
        base_cfg = load_config(base_path)          # recursive: bases may chain
        cfg = deep_merge(base_cfg, cfg)

    return cfg


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge `override` into `base`; `override` wins on conflict.

    "Deep" means nested dictionaries are merged key-by-key rather than being
    replaced wholesale. This matters -- a shallow merge would silently discard
    sibling keys.

    Example
    -------
    base     = {"physics": {"sim_freq_hz": 240, "gravity": 9.81}}
    override = {"physics": {"sim_freq_hz": 500}}

    deep merge -> {"physics": {"sim_freq_hz": 500, "gravity": 9.81}}   correct
    shallow    -> {"physics": {"sim_freq_hz": 500}}                    gravity lost!
    """
    result = copy.deepcopy(base)

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)   # recurse into nested dict
        else:
            result[key] = copy.deepcopy(value)             # scalar / list -> replace

    return result


# ==============================================================================
# SECTION 2 : ACCESS HELPERS
# ==============================================================================

def get_nested(cfg: dict, dotted_key: str, default: Any = None) -> Any:
    """Read a nested config value using dotted notation.

    Turns this:
        cfg["probe"]["vertical_separations_m"]
    into this:
        get_nested(cfg, "probe.vertical_separations_m")

    Returns `default` if any level of the path is missing, so callers do not
    need nested `if key in cfg` checks.
    """
    node: Any = cfg
    for part in dotted_key.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def require(cfg: dict, dotted_key: str) -> Any:
    """Read a nested config value, raising if it is absent.

    Use this for settings that have no sensible default. Failing loudly at
    startup is far better than silently substituting a wrong value and
    discovering it after a 200-run sweep.
    """
    sentinel = object()
    value = get_nested(cfg, dotted_key, sentinel)
    if value is sentinel:
        raise KeyError(f"Required config key missing: '{dotted_key}'")
    return value


# ==============================================================================
# SECTION 3 : SAVING (provenance)
# ==============================================================================

def save_config(cfg: dict, out_dir: str | Path, filename: str = "config.yaml") -> Path:
    """Write the fully-resolved config next to its results.

    This is what makes a result self-describing: the output directory contains
    both the numbers AND the exact settings that produced them, with all
    inheritance already resolved (so it can be read without chasing `_base_`).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / filename
    with open(out_path, "w", encoding="utf-8") as f:
        # sort_keys=False preserves our authored ordering, which is easier to read.
        yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)

    return out_path
