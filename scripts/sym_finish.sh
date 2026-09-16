#!/usr/bin/env bash
# sym_all.sh 가 끝나면 표 본문과 표 이미지를 만들어 오늘 폴더에 놓는다.
# 세훈님이 자리에 없어도 산출물이 나와 있게 하려는 것이다.
set -u
R=/home/sehoon/Documents/GitHub/g1-motion-tracking
PY=/home/sehoon/miniconda3/envs/env_isaaclab/bin/python
DAY="/home/sehoon/Desktop/참고/할일/26/09/260915"
L=$R/outputs/sym/finish.log

echo "=== 대기 시작 $(date +%F\ %H:%M:%S)" >> "$L"
while pgrep -f "[s]ym_all.sh" >/dev/null; do sleep 60; done
echo "=== sym_all 종료 확인 $(date +%F\ %H:%M:%S)" >> "$L"

cd "$R"
OMP_NUM_THREADS=1 "$PY" src/sym_table.py --out "$DAY/sim2sim_결과.md" >> "$L" 2>&1
echo "    본문 종료코드 $?" >> "$L"
OMP_NUM_THREADS=1 "$PY" src/sym_table_png.py --out "$DAY/sim2sim_표.png" >> "$L" 2>&1
echo "    이미지 종료코드 $?" >> "$L"
ls -la "$DAY" >> "$L" 2>&1
echo "=== 끝 $(date +%F\ %H:%M:%S)" >> "$L"
