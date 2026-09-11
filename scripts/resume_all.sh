#!/usr/bin/env bash
# walk4 재개(8,000) 후 나머지 시퀀스 체인 계속. 완료분은 체인이 알아서 건너뛴다.
set -u
S=/home/sehoon/Documents/GitHub/g1-motion-tracking/scripts
"$S/resume_walk4.sh"
"$S/train_chain.sh"
