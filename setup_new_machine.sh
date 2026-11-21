#!/usr/bin/env bash
# =============================================================================
#  SETUP ON A NEW MACHINE
# =============================================================================
#  Run from the project root, inside WSL/Ubuntu (or any Linux/macOS shell):
#
#      bash setup_new_machine.sh
#
#  It is SAFE TO RE-RUN. Each step checks whether it is already done.
#
#  The script ends by running the full test suite. That is the real
#  verification: 100 tests pass only if the physics, the learner and the
#  evaluation protocol all match the machine that produced the recorded
#  results. Do not trust any new number until they pass.
# =============================================================================

set -euo pipefail

GPBD_COMMIT="90c97ef3d97ce5de90ca720537707ff1ee97efb8"
step() { printf '\n\033[1m==> %s\033[0m\n' "$1"; }
fail() { printf '\n\033[31mFAILED: %s\033[0m\n' "$1" >&2; exit 1; }

# ---------------------------------------------------------------------------
step "1/6  Checking Python 3.12"
# ---------------------------------------------------------------------------
# 3.12 SPECIFICALLY. 3.13 was tried during the original setup and fails: some
# pinned wheels have no 3.13 build and fall back to building from source, which
# then breaks. 3.11 and earlier lack syntax used in the codebase.
if ! command -v python3.12 >/dev/null 2>&1; then
    cat <<'MSG'
python3.12 not found. On Ubuntu/WSL:

    sudo apt update
    sudo apt install -y python3.12 python3.12-venv python3.12-dev

If apt has no python3.12 (Ubuntu < 24.04):

    sudo add-apt-repository ppa:deadsnakes/ppa
    sudo apt update
    sudo apt install -y python3.12 python3.12-venv python3.12-dev
MSG
    fail "install Python 3.12, then re-run this script"
fi
echo "    found: $(python3.12 --version)"

# ---------------------------------------------------------------------------
step "2/6  Checking the C/C++ toolchain"
# ---------------------------------------------------------------------------
# gym-pybullet-drones and pybullet COMPILE during install. Without a compiler
# the install fails partway through with a wall of C errors -- this bit the
# original setup, so it is checked up front.
if ! command -v cc >/dev/null 2>&1 || ! command -v g++ >/dev/null 2>&1; then
    echo "    missing. On Ubuntu/WSL:  sudo apt install -y build-essential"
    fail "install build-essential, then re-run this script"
fi
echo "    found: $(cc --version | head -1)"

if ! command -v git >/dev/null 2>&1; then
    fail "git not found -- sudo apt install -y git"
fi

# ---------------------------------------------------------------------------
step "3/6  Creating the virtual environment"
# ---------------------------------------------------------------------------
# NEVER copy .venv between machines: it contains absolute paths and compiled
# binaries for the machine that built it. It is also 5.3 GB, which is why it is
# never committed to the repository. Always rebuild it here.
if [ -d .venv ]; then
    echo "    .venv already exists -- reusing it"
else
    python3.12 -m venv .venv
    echo "    created"
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --quiet --upgrade pip setuptools wheel

# ---------------------------------------------------------------------------
step "4/6  Installing PyTorch (CPU build)"
# ---------------------------------------------------------------------------
# CPU ON PURPOSE. This project never used a GPU: every recorded result came from
# a machine where torch.cuda.is_available() was False, and the training code
# runs with device="cpu". The CUDA build is ~2.5 GB of wheels that would do
# nothing, and on a laptop without an NVIDIA card may not install at all.
if python -c "import torch" 2>/dev/null; then
    echo "    already installed: $(python -c 'import torch; print(torch.__version__)')"
else
    pip install --index-url https://download.pytorch.org/whl/cpu torch==2.13.0
fi

# ---------------------------------------------------------------------------
step "5/6  Installing the remaining pinned dependencies"
# ---------------------------------------------------------------------------
# Includes gym-pybullet-drones at the exact commit. This is the slow step
# (it compiles); several minutes is normal.
pip install -r requirements.cpu.txt

step "    Verifying the simulator commit"
# A DIFFERENT COMMIT MEANS DIFFERENT PHYSICS. The downwash model and the
# BaseAviary.step() counter ordering both changed between commits, and the
# second of those caused a real off-by-two bug in episode length. If this check
# fails, the numbers this project recorded are not reproducible here.
ACTUAL=$(pip show gym_pybullet_drones 2>/dev/null | grep -i '^Version' || true)
python - <<'PYEOF'
import sys
try:
    import gym_pybullet_drones as g
    print(f"    gym-pybullet-drones imported from {g.__file__}")
except Exception as exc:
    sys.exit(f"    could not import gym_pybullet_drones: {exc}")
PYEOF
echo "    expected commit: ${GPBD_COMMIT}"

# ---------------------------------------------------------------------------
step "6/6  Running the test suite (this is the real check)"
# ---------------------------------------------------------------------------
# 100 tests. Several are deliberately slow because they run the REAL
# environment -- a test that builds its own objects instead of the ones the
# experiment uses has repeatedly missed defects in this project.
#
# Expect roughly 4-6 minutes.
if python -m pytest tests/ -q; then
    cat <<'MSG'

=============================================================================
  SETUP COMPLETE -- all tests pass.

  The environment matches the one that produced the recorded results.

  NEXT: read HANDOFF.md. It says where the work stopped, what the open
  decision is, and the exact command to run next. There IS an open decision
  that needs a human answer before more compute is spent.

  Activate the environment in future sessions with:
      source .venv/bin/activate
=============================================================================
MSG
else
    cat <<'MSG'

=============================================================================
  TESTS FAILED. Do not run experiments yet.

  Most likely causes, in order:
    1. Wrong gym-pybullet-drones commit  -> physics differs
    2. Wrong Python version              -> check: python --version (want 3.12)
    3. A dependency resolved differently -> compare: pip freeze

  Check the failing test names against the failure modes in
  SETUP_NEW_MACHINE.md; the failure usually identifies the cause directly.
=============================================================================
MSG
    exit 1
fi
