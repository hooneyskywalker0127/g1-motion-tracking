#!/usr/bin/env bash
# 학습이 끝난 시퀀스를 감지해 평가 -> 3티어 렌더 -> 구글 드라이브 업로드까지
# 사람 없이 이어서 한다. 주말처럼 컴퓨터에 접근할 수 없을 때 쓴다.
#
# 업로드 대상은 시퀀스당 병합 영상 세 개다.
#   <시퀀스>_compare.mp4  <시퀀스>_progression.mp4  <시퀀스>_randomization.mp4
#
# 멈추려면 /tmp/stop_auto_pipeline 파일을 만든다. 진행 중인 시퀀스는 끝까지 간다.
set -u
REPO=/home/sehoon/Documents/GitHub/g1-motion-tracking
WBT=/home/sehoon/Projects/whole_body_tracking
RCLONE=/home/sehoon/.local/bin/rclone
PY=/home/sehoon/miniconda3/envs/env_isaaclab/bin/python
REMOTE=${REMOTE:-gdrive}
RDIR=${RDIR:-g1-motion-tracking}
STATE=$REPO/outputs/uploaded.txt
LOG=$REPO/outputs/auto_pipeline.log

mkdir -p "$REPO/outputs"
touch "$STATE"

say() { echo "[$(date +%F\ %H:%M:%S)] $*"; }

# 시퀀스 하나를 처리한다. 이미 있는 산출물은 각 렌더 스크립트가 알아서 건너뛴다.
process() {
  local seq=$1
  # 날짜 폴더를 두지 않는다. 26/09/g1_motion_tracking/<시퀀스>/ 아래에 쌓는다.
  local base=/home/sehoon/Desktop/참고/할일/26/09/g1_motion_tracking

  say "$seq 시작"

  # 자막에 들어갈 수치. 평가가 없으면 자막이 'not evaluated yet'으로 나온다.
  if [ ! -f "$REPO/outputs/eval/$seq.json" ]; then
    say "$seq 평가"
    bash "$REPO/scripts/eval_all.sh" 100 || say "$seq 평가 실패, 렌더는 계속"
  fi

  bash "$REPO/scripts/render_tracking.sh"      "$seq" "$base" || say "$seq compare 실패"
  bash "$REPO/scripts/render_progression.sh"   "$seq" "$base" || say "$seq progression 실패"
  bash "$REPO/scripts/render_randomization.sh" "$seq" "$base" || say "$seq randomization 실패"

  # 영상마다 유튜브 제목·설명을 영어로 써 둔다. 영상과 같이 올라간다.
  "$PY" "$REPO/scripts/write_youtube_notes.py" "$seq" "$base" || say "$seq 노트 실패"

  # 병합본 세 개와 그에 딸린 노트를 올린다.
  local ok=0
  for f in "$base/$seq/compare/${seq}_compare.mp4" \
           "$base/$seq/compare/${seq}_compare_youtube.txt" \
           "$base/$seq/progression/${seq}_progression.mp4" \
           "$base/$seq/progression/${seq}_progression_youtube.txt" \
           "$base/$seq/randomization/${seq}_randomization.mp4" \
           "$base/$seq/randomization/${seq}_randomization_youtube.txt"; do
    if [ ! -f "$f" ]; then
      say "없음: $(basename "$f")"
      continue
    fi
    if grep -qxF "$seq/$(basename "$f")" "$STATE"; then
      ok=$((ok+1))
      continue
    fi
    say "업로드 $(basename "$f") ($(du -h "$f" | cut -f1))"
    if "$RCLONE" copy "$f" "$REMOTE:$RDIR/$seq/" \
         --retries 10 --low-level-retries 20 --transfers 1; then
      echo "$seq/$(basename "$f")" >> "$STATE"
      ok=$((ok+1))
      say "업로드 완료 $(basename "$f")"
    else
      say "업로드 실패 $(basename "$f")"
    fi
  done
  say "$seq 종료 · 업로드 $ok/6 (영상 3 + 노트 3)"
}

# 학습이 30,000까지 끝났는지 본다. 재개한 학습은 폴더 이름 끝에 _resume, _resume2 가 붙는다.
trained() {
  ls "$WBT"/logs/rsl_rl/g1_flat/*_"$1"*/model_29999.pt >/dev/null 2>&1
}

say "감시 시작 · 원격 $REMOTE:$RDIR"
while true; do
  [ -f /tmp/stop_auto_pipeline ] && { say "중지 요청 확인"; break; }

  # 손으로 띄운 렌더가 돌고 있으면 GPU를 두 겹으로 쓰지 않도록 기다린다.
  # 세트 전체뿐 아니라 개별 렌더와 재렌더도 함께 본다.
  while pgrep -f "render_all_sets.sh|rerun_walk2_progression.sh|render_progression.sh|render_randomization.sh|render_tracking.sh" >/dev/null; do
    sleep 120
  done

  # 학습 순서 목록에 더해, 목록 밖에서 이미 학습이 끝난 시퀀스도 대상에 넣는다.
  # walk2_subject4처럼 따로 돌린 것이 빠지지 않게 하려는 것이다.
  done_seqs=$(ls -1d "$WBT"/logs/rsl_rl/g1_flat/*/ 2>/dev/null |
    while read -r d; do
      [ -f "$d/model_29999.pt" ] || continue
      # 재개 폴더는 _resume, _resume2 처럼 숫자가 붙기도 한다. ${n%_resume} 만으로는
      # _resume2 가 안 잘려 walk4_subject1_resume2 를 별개 시퀀스로 만들어 버린다.
      n=$(basename "$d"); n=${n#*_*_}; echo "${n%%_resume*}"
    done)
  targets=$(printf '%s\n%s\n' "$(cat "$REPO/configs/train_order.txt")" "$done_seqs" |
    awk 'NF && !seen[$0]++')

  left=0
  for seq in $targets; do
    [ -f /tmp/stop_auto_pipeline ] && break
    if ! trained "$seq"; then
      left=$((left+1))
      continue
    fi
    # 세 개 다 올렸으면 건너뛴다.
    n=$(grep -c "^$seq/" "$STATE")
    [ "$n" -ge 6 ] && continue
    process "$seq"
  done

  if [ "$left" -eq 0 ]; then
    say "남은 학습 없음, 감시 종료"
    break
  fi
  sleep 600
done
say "감시 끝"
