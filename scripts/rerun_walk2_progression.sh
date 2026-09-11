#!/usr/bin/env bash
# walk2_subject4 발전과정은 깨진 iter_10000.mp4 때문에 실패했다.
# 지금 도는 렌더 세트가 끝나기를 기다렸다가 그 하나만 다시 만든다.
set -u
while pgrep -f render_all_sets.sh > /dev/null; do sleep 60; done
bash /home/sehoon/Documents/GitHub/g1-motion-tracking/scripts/render_progression.sh walk2_subject4 \
  && echo "SET progression walk2_subject4 OK" \
  || echo "SET progression walk2_subject4 FAIL"
