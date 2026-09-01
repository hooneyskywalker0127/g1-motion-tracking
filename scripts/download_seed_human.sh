#!/usr/bin/env bash
# SEED 사람 원본(soma_proportional) 다운로드 + 압축 해제
set -e

HF=/home/sehoon/miniconda3/envs/env_isaaclab/bin/hf
DST=/home/sehoon/data/bones_seed

echo "[1/2] downloading soma_proportional.tar.gz (45.5 GB)"
"$HF" download bones-studio/seed soma_proportional.tar.gz \
  --repo-type dataset --local-dir "$DST"

echo "[2/2] extracting"
tar -xzf "$DST/soma_proportional.tar.gz" -C "$DST"

echo "done. contents:"
ls "$DST"
