"""
================================================================================
 TESTS: configuration and seeding utilities
================================================================================

WHY TEST THESE FIRST
--------------------
Config loading and seeding are INFRASTRUCTURE: every later experiment depends
on them. A silent bug here (a config value not overriding, a seed not actually
applied) would corrupt every result downstream and be very hard to detect,
because nothing would crash -- the numbers would just quietly be wrong.

These tests need no simulator, so they run in under a second.

RUN WITH
--------
    pytest tests/ -v
================================================================================
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make src/ importable without requiring an editable install.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pytest
import yaml

from ctde.utils.config import deep_merge, get_nested, load_config, require
from ctde.utils.seeding import make_rng, seed_everything


# ==============================================================================
# SECTION 1 : CONFIG MERGING
# ==============================================================================

def test_deep_merge_preserves_sibling_keys():
    """Nested merge must not discard keys the override did not mention.

    This is THE bug that a shallow merge would introduce: overriding one
    physics setting would silently delete all the others.
    """
    base = {"physics": {"sim_freq_hz": 240, "gravity": 9.81}}
    override = {"physics": {"sim_freq_hz": 500}}

    result = deep_merge(base, override)

    assert result["physics"]["sim_freq_hz"] == 500     # overridden
    assert result["physics"]["gravity"] == 9.81        # preserved


def test_deep_merge_does_not_mutate_inputs():
    """Merging must not modify the caller's dictionaries.

    Configs get merged repeatedly during inheritance resolution; in-place
    mutation would make later loads depend on earlier ones.
    """
    base = {"a": {"b": 1}}
    override = {"a": {"c": 2}}

    deep_merge(base, override)

    assert base == {"a": {"b": 1}}
    assert override == {"a": {"c": 2}}


# ==============================================================================
# SECTION 2 : CONFIG INHERITANCE (end-to-end, via real files)
# ==============================================================================

def test_config_inheritance(tmp_path):
    """A child config with `_base_` inherits and overrides correctly."""
    (tmp_path / "base.yaml").write_text(
        yaml.safe_dump({
            "physics": {"sim_freq_hz": 240, "mode": "pyb_gnd_drag_dw"},
            "drone": {"model": "cf2x"},
        }),
        encoding="utf-8",
    )
    (tmp_path / "child.yaml").write_text(
        yaml.safe_dump({
            "_base_": "base.yaml",
            "physics": {"sim_freq_hz": 500},
            "probe": {"settle_steps": 240},
        }),
        encoding="utf-8",
    )

    cfg = load_config(tmp_path / "child.yaml")

    assert cfg["physics"]["sim_freq_hz"] == 500              # child wins
    assert cfg["physics"]["mode"] == "pyb_gnd_drag_dw"       # inherited
    assert cfg["drone"]["model"] == "cf2x"                   # inherited
    assert cfg["probe"]["settle_steps"] == 240               # child-only key
    assert "_base_" not in cfg                               # marker consumed


def test_real_probe_config_loads():
    """The actual probe config in this repo must load and be well-formed.

    Guards against a typo in configs/*.yaml reaching a long sweep.
    """
    cfg = load_config(PROJECT_ROOT / "configs" / "probe_downwash.yaml")

    # Inherited from base.yaml
    assert cfg["physics"]["mode"] == "pyb_gnd_drag_dw", (
        "Probe REQUIRES downwash physics, else it measures nothing."
    )
    # Probe-specific
    assert len(cfg["probe"]["vertical_separations_m"]) >= 3
    assert 0.0 in cfg["probe"]["horizontal_offsets_m"], (
        "Offset 0.0 (directly beneath) is required: it is the aligned case "
        "used for the coupling-horizon estimate."
    )


# ==============================================================================
# SECTION 3 : CONFIG ACCESS
# ==============================================================================

def test_get_nested_and_require():
    cfg = {"a": {"b": {"c": 42}}}

    assert get_nested(cfg, "a.b.c") == 42
    assert get_nested(cfg, "a.b.missing") is None
    assert get_nested(cfg, "a.b.missing", default=7) == 7

    assert require(cfg, "a.b.c") == 42
    with pytest.raises(KeyError):
        require(cfg, "a.b.missing")     # must fail loudly, not return None


# ==============================================================================
# SECTION 4 : SEEDING
# ==============================================================================

def test_seeding_is_reproducible():
    """Same seed must produce the same numbers -- the basis of Charter §7."""
    seed_everything(123)
    first = np.random.rand(5)

    seed_everything(123)
    second = np.random.rand(5)

    np.testing.assert_array_equal(first, second)


def test_different_seeds_differ():
    """Different seeds must produce different numbers.

    Without this, an 8-seed sweep would be 8 identical runs and every
    confidence interval would be meaninglessly narrow.
    """
    seed_everything(1)
    a = np.random.rand(5)

    seed_everything(2)
    b = np.random.rand(5)

    assert not np.allclose(a, b)


def test_named_streams_are_independent():
    """Separate streams must not interfere with one another.

    This is why we prefer make_rng() over the global np.random: drawing wind
    samples must never shift the numbers the exploration stream sees.
    """
    wind = make_rng(seed=0, stream="wind")
    pose = make_rng(seed=0, stream="init_pose")

    assert not np.allclose(wind.random(5), pose.random(5))


def test_stream_reproducible_regardless_of_draw_order():
    """A stream's output must not depend on what other streams did.

    Concretely: draining the wind stream first must not change what the pose
    stream produces. This is the property the global RNG lacks.
    """
    pose_a = make_rng(seed=0, stream="init_pose").random(5)

    wind = make_rng(seed=0, stream="wind")
    wind.random(100)                                     # heavy use of another stream
    pose_b = make_rng(seed=0, stream="init_pose").random(5)

    np.testing.assert_array_equal(pose_a, pose_b)
