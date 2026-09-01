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

| dataset | contents | status |
| --- | --- | --- |
| bones-studio/seed | motions already retargeted to G1 | local, `data/bones_seed` |
| LAFAN1_Retargeting_Dataset | LAFAN1 motions retargeted to G1 | not downloaded yet |

Both datasets ship motions that are already retargeted. They are used here as a
reference to check this pipeline's own retargeting output against, not as its
input.

두 데이터셋은 리타게팅이 이미 끝난 결과물입니다. 이 저장소에서는 입력이
아니라, 직접 만든 리타게팅 결과가 맞는지 대조할 기준으로 사용합니다.

## Status

Setting up. Nothing runs yet.
구성 중이며, 아직 실행 가능한 코드는 없습니다.
