#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
MODE="${1:-bootstrap}"
[[ "$MODE" == bootstrap || "$MODE" == fast || "$MODE" == full ]] || { echo "Usage: $0 [bootstrap|fast|full]" >&2; exit 2; }
mkdir -p runs/setup .cache
export PIP_CACHE_DIR="$ROOT/.cache/pip"
export HF_HOME="$ROOT/.cache/huggingface"
export CUDA_VISIBLE_DEVICES=0
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
if [[ ! -x .venv/bin/python ]]; then python3 -m venv --without-pip .venv; fi
if ! .venv/bin/python -m pip --version >/dev/null 2>&1; then
  # Ubuntu may lack ensurepip; bootstrap only this project's venv, without sudo.
  python3 -m venv --without-pip .venv
  curl --fail --location --retry 3 https://bootstrap.pypa.io/get-pip.py -o runs/setup/get-pip.py
  .venv/bin/python runs/setup/get-pip.py
fi
.venv/bin/python -m pip install --upgrade 'pip==25.3' 'setuptools==80.9.0' 'wheel==0.45.1'
.venv/bin/python -m pip install -r requirements-sim.txt
.venv/bin/python -m pip install -e . --no-deps
# CUDA12.4 wheels match the NVIDIA 550 driver on this host.
.venv/bin/python -m pip install 'torch==2.6.0' --index-url https://download.pytorch.org/whl/cu124
if [[ "$MODE" == full ]]; then
  .venv/bin/python -m pip install 'torchvision==0.21.0' --index-url https://download.pytorch.org/whl/cu124
  .venv/bin/python -m pip install -r requirements-perception.txt
fi
.venv/bin/python -m pip check
.venv/bin/python -m pip freeze | sed -e '/^## !! Could not determine repository location/d' -e 's|^-e .*Cross_Robot_Real2Sim$|-e .|' > "requirements-working-${MODE}.lock"
if [[ -f src/cross_robot_real2sim/adapters/source_hdf5.py ]]; then
  .venv/bin/python -m cross_robot_real2sim.adapters.source_hdf5 --download --inspect
else
  .venv/bin/python - <<'PY'
from huggingface_hub import hf_hub_download
p = hf_hub_download('robomimic/robomimic_datasets', 'v1.5/can/ph/low_dim_v15.hdf5', repo_type='dataset', local_dir='data/source')
import h5py
with h5py.File(p) as f:
    print('Dataset:', p, 'keys:', list(f.keys()), 'episodes:', len(f['data']))
PY
fi
printf 'Setup complete. Reproduce: bash scripts/setup.sh %s\nNext: bash scripts/preflight.sh && bash scripts/launch.sh %s\n' "$MODE" "$MODE"
