#!/usr/bin/env bash
# 77개 전체의 IK 목표 추적 오차를 재서 csv로 남긴다.
set -e
source /home/sehoon/miniconda3/etc/profile.d/conda.sh
conda activate gmr
cd /home/sehoon/Documents/GitHub/g1-motion-tracking
MUJOCO_GL=egl python src/quality_metrics.py "$@"
