# g1-motion-tracking

Human mocap retargeting and whole-body motion tracking pipeline for the Unitree G1.

사람의 모션 캡처 데이터를 Unitree G1 휴머노이드로 옮기고, 그 동작을 따라가도록
전신 제어 정책을 학습시키는 파이프라인입니다.

Simulation only. Real-robot deployment is out of scope for now.
현재 범위는 시뮬레이션까지이며, 실제 로봇 배포는 포함하지 않습니다.

---

## Pipeline

```
  human mocap        retargeting        reference motion       RL training
 (사람 모캡 데이터)  →  (리타게팅)     →   (레퍼런스 모션)    →   (강화학습)
   joint positions     G1 joint angles     target trajectory     tracking policy
```

### 1. Human mocap — 사람 모션 캡처 데이터

Motion capture records where every joint of a human body was at each instant,
as numbers rather than video.

사람 몸의 관절이 매 순간 어디에 있었는지를 숫자로 기록한 데이터입니다.
영상이 아니라 "0.1초 시점에 왼쪽 무릎은 여기, 오른쪽 팔꿈치는 여기" 같은
좌표의 나열입니다.

### 2. Retargeting — 리타게팅

Human joint data cannot be applied to the G1 directly: link lengths differ,
joint axes differ, and some human joints have no robot counterpart. Retargeting
solves for the robot joint angles that best reproduce the original motion while
respecting the robot's joint limits and keeping the feet on the ground.

사람 기준의 관절값을 G1에 그대로 쓸 수는 없습니다. 팔다리 길이가 다르고,
관절이 꺾이는 축도 다르며, 사람에게 있는 관절이 로봇에는 없기도 합니다.
그래서 "어깨 각도 몇 도"를 복사하는 대신, 손끝과 발끝 위치처럼 지켜야 할
것을 정해 두고 그것을 최대한 만족하는 로봇 관절값을 최적화로 찾습니다.
로봇의 관절 한계를 넘지 않아야 하고, 발이 바닥을 뚫거나 뜨지 않아야 합니다.

### 3. Reference motion — 레퍼런스 모션

The retargeted trajectory becomes the target the robot is asked to follow.
At this point nothing is physically simulated yet — only the goal exists.

리타게팅 결과가 로봇이 따라가야 할 목표 궤적이 됩니다. 이 단계까지는
아직 로봇이 실제로 움직인 것이 아니라, 따라야 할 정답 동작만 만들어진
상태입니다.

### 4. Motion tracking policy — 모션 트래킹 정책 학습

The robot is trained with reinforcement learning: it is rewarded for staying
close to the reference motion and penalised for drifting away or falling. The
control rule it converges on is the policy. Success is measured by tracking
accuracy, not by whether the walk merely looks plausible.

강화학습으로 로봇이 그 목표를 따라가게 만듭니다. 레퍼런스 모션에 가까우면
점수를 주고, 벗어나거나 넘어지면 점수를 깎습니다. 이 과정을 반복해 로봇이
찾아낸 조종 방법이 정책(policy)이고, 점수 규칙이 보상(reward)입니다.
평가 기준은 "그럴듯하게 걷는가"가 아니라 "레퍼런스를 얼마나 정확히
따라갔는가"입니다.

---

## Layout

```
src/       retargeting and training code   리타게팅·학습 코드
configs/   robot / dataset / training      로봇·데이터셋·학습 설정
outputs/   motions, logs, videos           생성 결과 (gitignored)
data ->    symlink to local dataset root   데이터 심볼릭 링크 (gitignored)
```

## Data

Motion datasets run to tens of gigabytes and are not tracked in this repo.
They live under `/home/sehoon/data`, reached through the `data` symlink.

모션 데이터는 수십 기가바이트라 저장소에 포함하지 않습니다.
로컬의 `/home/sehoon/data` 아래에 두고 `data` 심볼릭 링크로 접근합니다.

### Primary source — LAFAN1

[LAFAN1](https://github.com/ubisoft/ubisoft-laforge-animation-dataset) is used
as the input to retargeting. 4.6 hours, 77 sequences, 5 subjects, BVH at 30 fps.
It was picked over the larger alternatives for three reasons:

1. Format. BVH is consumed directly by the retargeting tools in use; no
   intermediate body-model fitting step is required.
2. Comparability. Prior work has published per-sequence tracking success rates
   for four retargeting methods on a 21-sequence subset of LAFAN1, so results
   here can be placed against known numbers rather than argued for.
3. Motion range. Sequences run 5 s to 2 min and span walking and turning through
   martial arts and dance, which separates the cases retargeting handles from
   the ones it breaks on.

리타게팅의 입력으로 LAFAN1을 사용합니다. 4.6시간, 77개 시퀀스, 5명, 30 fps
BVH 형식입니다. 규모가 더 큰 후보들 대신 선택한 이유는 세 가지입니다.

1. 형식. BVH는 사용할 리타게팅 도구가 그대로 받으므로, 중간에 몸 모델을
   맞추는 변환 단계가 필요 없습니다.
2. 비교 가능성. 선행 연구가 LAFAN1의 21개 시퀀스에 대해 네 가지 리타게팅
   방법의 추적 성공률을 공개했습니다. 따라서 결과를 주장하는 대신 알려진
   수치와 대조할 수 있습니다.
3. 동작의 폭. 5초에서 2분까지 이어지며 걷기·회전부터 격투·춤까지 포함해,
   리타게팅이 감당하는 구간과 깨지는 구간을 나누어 볼 수 있습니다.

Download `lafan1.zip` from the [official repo](https://github.com/ubisoft/ubisoft-laforge-animation-dataset/blob/master/lafan1/lafan1.zip)
and unzip it into `data/lafan1`. All 77 `.bvh` files sit flat in that directory.

[공식 저장소](https://github.com/ubisoft/ubisoft-laforge-animation-dataset/blob/master/lafan1/lafan1.zip)에서
`lafan1.zip`을 받아 `data/lafan1`에 풉니다. bvh 77개가 그 아래 평평하게 놓입니다.

### Reference — already-retargeted G1 motions

These ship motions that have already been retargeted to the G1. They are not
inputs, and they are not ground truth either — they are another method's output.
Comparing joint angles against them measures how far two methods diverge, not
how much error either one carries. They are kept for qualitative reference only.

이들은 G1으로 리타게팅이 이미 끝난 결과물입니다. 입력도 아니고 정답지도
아닙니다. 다른 방법의 출력이므로, 관절각을 맞대어 보면 두 방법이 얼마나
갈리는지가 나올 뿐 어느 쪽의 오차인지는 알 수 없습니다. 눈으로 참고하는
용도로만 둡니다.

| dataset | contents | local path |
| --- | --- | --- |
| lvhaidong/LAFAN1_Retargeting_Dataset | LAFAN1 retargeted to G1 | `data/lafan1_g1_ref` |
| bones-studio/seed | 142,220 Vicon motions, human and G1 side | `data/bones_seed` |

SEED is held for later. It is larger than LAFAN1 and pairs each motion with the
capture subject's measured body dimensions, but its human-side data is in a
custom format whose compatibility with the retargeting tools is unverified, and
no published baseline exists for it. It becomes useful once the pipeline runs.

SEED는 나중을 위해 보류합니다. LAFAN1보다 크고 배우의 실측 신체 치수가 함께
제공되지만, 사람 쪽 데이터가 자체 형식이라 리타게팅 도구와 호환되는지
확인되지 않았고 공개된 비교 기준도 없습니다. 파이프라인이 돌아간 뒤에
쓸모가 생깁니다.

AMASS is deferred to the training stage, where scale matters more than
comparability.

AMASS는 규모가 비교 가능성보다 중요해지는 학습 단계에서 사용합니다.

## Tools

| tool | role |
| --- | --- |
| [GMR](https://github.com/YanjieZe/GMR) | retargeting tool in use; MIT, takes LAFAN1 BVH and outputs G1 joint angles |
| Isaac Sim 5.1 / Isaac Lab 2.3.2 | reference motion playback, and RL training in stage 4 |

Retargeting itself needs no physics simulation — it is a kinematics problem.
The simulator enters at playback and at policy training.

리타게팅 자체는 물리 시뮬레이션이 필요 없는 기구학 문제입니다.
시뮬레이터는 레퍼런스 모션 재생과 정책 학습 단계에서 사용합니다.

## Background

The premise that retargeting quality is worth treating as its own problem comes
from *Retargeting Matters: General Motion Retargeting for Humanoid Motion
Tracking* ([arXiv:2510.02252](https://arxiv.org/abs/2510.02252)), which showed that artifacts left in retargeted
trajectories — foot sliding, self-penetration, physically infeasible poses —
measurably reduce the robustness of the tracking policy trained on them.

리타게팅 품질을 별도의 문제로 다루는 근거는 Retargeting Matters: General
Motion Retargeting for Humanoid Motion Tracking(arXiv:2510.02252)입니다. 리타게팅
결과에 남은 결함, 즉 발 미끄러짐·자기 충돌·물리적으로 불가능한 자세가
그것으로 학습한 추적 정책의 안정성을 떨어뜨린다는 것을 보였습니다.

## Status

Stages 1 and 2 are done. All 77 LAFAN1 sequences are retargeted to the G1 and
verified: no NaNs, no joint-limit violations, frame counts matching the source
BVH exactly. Stage 3 is in progress — per-sequence IK target error has been
measured across all 496,672 frames, and the sequences to carry into policy
training are being selected from it. Stage 4 has not started.

Foot tracking error by motion type: walk 1.00 cm, dance 1.25, run 1.33,
obstacles 1.65, fallAndGetUp 2.11, ground 2.76. Hand error sits at 5-9 cm
regardless of motion type, which is the arm-length gap rather than a per-motion
failure.

1·2단계는 끝났습니다. LAFAN1 77개 시퀀스를 모두 G1으로 리타게팅했고, NaN 없음,
관절 한계 위반 없음, 원본 BVH와 프레임 수 일치를 전수 확인했습니다. 3단계가
진행 중입니다. 496,672 프레임 전체에 대해 시퀀스별 IK 목표 추적 오차를 측정했고,
그 결과로 정책 학습에 쓸 시퀀스를 고르고 있습니다. 4단계는 시작 전입니다.

동작 종류별 발 추적 오차는 걷기 1.00 cm, 춤 1.25, 달리기 1.33, 장애물 1.65,
넘어졌다 일어나기 2.11, 바닥 동작 2.76입니다. 손 오차는 동작 종류와 무관하게
5~9 cm인데, 이는 개별 동작의 실패가 아니라 팔 길이 차이에서 오는 값입니다.
