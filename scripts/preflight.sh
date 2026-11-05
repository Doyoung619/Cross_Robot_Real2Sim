#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_runtime.sh"
cd "$ROOT"
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 HF_HOME="$ROOT/.cache/huggingface"
export MUJOCO_GL="${MUJOCO_GL:-egl}" PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
mkdir -p runs/preflight
nvidia-smi
python --version
df -h "$ROOT"
python -m pip check
python -m pip freeze > runs/preflight/packages.txt
python -m cross_robot_real2sim.adapters.source_hdf5 --inspect | tee runs/preflight/dataset.txt
python -m cross_robot_real2sim.cli check 2>&1 | tee runs/preflight/check.log
