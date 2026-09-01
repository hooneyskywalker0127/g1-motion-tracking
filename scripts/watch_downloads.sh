#!/usr/bin/env bash
# 다운로드 감시기: SEED 사람 원본 + LAFAN1 두 종만 담당.
# 진행이 멈추면 이어받기로 재시작한다. 그 외 어떤 것도 건드리지 않는다.

LOG=/home/sehoon/data/dl_supervisor.log
HF=/home/sehoon/miniconda3/envs/env_isaaclab/bin/hf
SEED=/home/sehoon/data/bones_seed
LA=/home/sehoon/data/lafan1
REF=/home/sehoon/data/lafan1_g1_ref
ZIP_SIZE=144051503
INTERVAL=120        # 확인 주기 (초)
STALL=6             # 이만큼 연속으로 안 늘면 죽은 것으로 본다 (12분)

log() { echo "$(date '+%m-%d %H:%M:%S') $*" >> "$LOG"; }

# 대상 경로 아래에서 가장 최근에 바뀐 시각 이후 흐른 초
idle_secs() {
  local newest
  newest=$(find "$1" -type f -newermt '-1 day' -printf '%T@\n' 2>/dev/null | sort -rn | head -1)
  [ -z "$newest" ] && { echo 999999; return; }
  echo $(( $(date +%s) - ${newest%.*} ))
}

seed_stall=0; la_stall=0; ref_stall=0
seed_prev=0; la_prev=0
log "=== supervisor start ==="

while true; do
  # ---------- 1. SEED 사람 원본 ----------
  if [ -f "$SEED/soma_proportional.tar.gz" ]; then
    if [ ! -d "$SEED/soma_proportional" ] && [ "$(idle_secs "$SEED")" -gt 900 ]; then
      log "SEED: 다운로드 완료, 압축 해제 시작"
      tar -xzf "$SEED/soma_proportional.tar.gz" -C "$SEED" >> "$LOG" 2>&1 \
        && log "SEED: 압축 해제 완료" || log "SEED: 압축 해제 실패"
    fi
  else
    cur=$(find "$SEED/.cache" -name '*.incomplete' -printf '%s\n' 2>/dev/null | sort -rn | head -1)
    cur=${cur:-0}
    if [ "$cur" -gt "$seed_prev" ]; then
      seed_stall=0
      log "SEED: $(( cur / 1000000 )) MB / 45470 MB"
    else
      seed_stall=$((seed_stall+1))
      log "SEED: 진행 없음 ($seed_stall/$STALL)"
    fi
    seed_prev=$cur
    if [ "$seed_stall" -ge "$STALL" ]; then
      log "SEED: 재시작"
      "$HF" download bones-studio/seed soma_proportional.tar.gz \
        --repo-type dataset --local-dir "$SEED" >> "$LOG" 2>&1
      seed_stall=0
    fi
  fi

  # ---------- 2. LAFAN1 원본 zip ----------
  cur=$(stat -c %s "$LA/lafan1.zip" 2>/dev/null || echo 0)
  if [ "$cur" -ge "$ZIP_SIZE" ]; then
    if [ -z "$(find "$LA" -name '*.bvh' -print -quit 2>/dev/null)" ]; then
      if unzip -tq "$LA/lafan1.zip" >/dev/null 2>&1; then
        log "LAFAN1: 압축 해제"
        unzip -o -q "$LA/lafan1.zip" -d "$LA" >> "$LOG" 2>&1
        log "LAFAN1: bvh $(find "$LA" -name '*.bvh' | wc -l) 개"
      else
        log "LAFAN1: zip 손상, 다시 받음"
        rm -f "$LA/lafan1.zip"
      fi
    fi
  else
    if [ "$cur" -gt "$la_prev" ]; then la_stall=0; else la_stall=$((la_stall+1)); fi
    la_prev=$cur
    log "LAFAN1 zip: $(( cur / 1000000 )) MB / 144 MB (stall $la_stall)"
    if [ "$la_stall" -ge "$STALL" ]; then
      log "LAFAN1: 재시작"
      curl -sL -C - -o "$LA/lafan1.zip" \
        https://media.githubusercontent.com/media/ubisoft/ubisoft-laforge-animation-dataset/master/lafan1/lafan1.zip \
        >> "$LOG" 2>&1
      la_stall=0
    fi
  fi

  # ---------- 3. LAFAN1 G1 정답지 ----------
  n=$(find "$REF" -name '*.csv' 2>/dev/null | wc -l)
  if [ "$n" -lt 40 ]; then
    if [ "$(idle_secs "$REF")" -gt 720 ]; then
      log "G1 ref: 진행 없음, 재시작 (현재 csv $n 개)"
      "$HF" download lvhaidong/LAFAN1_Retargeting_Dataset \
        --repo-type dataset --local-dir "$REF" >> "$LOG" 2>&1
    fi
  fi

  # ---------- 종료 조건 ----------
  if [ -d "$SEED/soma_proportional" ] \
     && [ -n "$(find "$LA" -name '*.bvh' -print -quit 2>/dev/null)" ] \
     && [ "$n" -ge 40 ]; then
    log "=== 세 가지 모두 완료 ==="
    break
  fi

  sleep $INTERVAL
done
