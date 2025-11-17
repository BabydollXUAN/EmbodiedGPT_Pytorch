#!/usr/bin/env bash
set -e
source .venv/bin/activate
python smoke_test.py
# tiny 权重放好后再跑
if [ -d "./checkpoints/Embodied_family_7btiny" ]; then
  python run_infer_min.py
fi
