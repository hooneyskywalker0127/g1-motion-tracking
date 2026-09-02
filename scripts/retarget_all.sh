#!/usr/bin/env bash
# LAFAN1 77개 전체 리타게팅. 이미 있는 결과는 건너뛴다.
set -e
source /home/sehoon/miniconda3/etc/profile.d/conda.sh
conda activate gmr
cd /home/sehoon/Documents/GitHub/g1-motion-tracking/src
python retarget_all.py "$@"
