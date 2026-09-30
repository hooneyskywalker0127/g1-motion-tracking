# 배경과 kobe

[← README](../README.md)

## 배경

리타게팅 품질을 별도의 문제로 다루는 근거는 Retargeting Matters: General
Motion Retargeting for Humanoid Motion Tracking([arXiv:2510.02252](https://arxiv.org/abs/2510.02252))입니다.
리타게팅 결과에 남은 결함, 즉 발 미끄러짐·자기 충돌·물리적으로 불가능한
자세가 그것으로 학습한 추적 정책의 안정성을 떨어뜨린다는 것을 보였습니다.

### kobe — LAFAN1 밖의 동작에서도 되는가

위 17개는 전부 LAFAN1의 보행·댄스입니다. 고난도 단발 동작이 없어, ASAP
데이터셋의 kobe 모션 하나를 `src/asap_to_csv.py`로 변환해 같은 파이프라인에
태웠습니다. 206프레임, 4.1초입니다.

![kobe](kobe.gif)

4.1초 전체입니다. 왼쪽이 Isaac Lab, 오른쪽이 MuJoCo입니다. MuJoCo 100시행에서
BeyondMimic 성공률 1.000, PolySim 성공률 0.990, 전역 바디 오차 128.0 mm입니다.
여기서도 전이 손실은 없습니다.

오해를 막기 위해 적습니다. 이 클립은 kobe 전용 정책으로 돌렸습니다. 17개를
학습한 정책이 kobe를 따라간 것이 아닙니다. 이 파이프라인은 모션 하나당 정책
하나라 kobe도 따로 30,000 iteration을 학습했습니다. 즉 여기서 보인 것은 전이이지
일반화가 아닙니다.

그래서 이 클립이 뒷받침하는 것은 하나뿐입니다. 전이가 잘 되는 것이 LAFAN1이라는
데이터셋의 특성 때문은 아니라는 것입니다. 다른 데이터셋의 4초짜리 빠른 동작에서도
같은 결과가 나옵니다. 한 개는 표본이 아니므로 그 이상은 주장하지 않습니다.

이 클립이 값을 갖는 자리는 따로 있습니다. 여러 모션을 하나의 정책으로 합치고
나면, kobe를 학습에 넣지 않은 채로 돌려 일반화를 잴 수 있습니다. 다른 데이터셋,
학습에 없는 동작 종류, 짧고 빠른 구간이라 학습 분포 밖이라는 것이 분명합니다.

세 번 학습한 끝에 나온 값입니다. 앞의 두 번은 csv 관절 순서를 URDF가 아닌
Isaac 순서로 쓴 것(`error_joint_pos` 2.44 rad)과 레퍼런스가 지면에서 떠 있던
것 때문에 수렴하지 않았습니다.

이 클립을 PolySim 논문 표와 맞대지 않는 이유는 세 가지입니다. 첫째, 그 논문
표 III에서 `IsaacSimDR → MuJoCo`의 0.100은 PolySim의 성적이 아니라 대조군인
단일 시뮬레이터 DR 베이스라인입니다. PolySim 자신의 값은 마지막 행
`IsaacSim+IsaacGym+Genesis`의 1.000입니다. 둘째, 논문이 쓴 14개 모션도 5개
모션도 이름을 밝히지 않아 같은 집합을 맞출 수 없습니다. 전문에 나오는 모션
이름은 Kobe 하나뿐입니다. 셋째, 학습기가 다릅니다. PolySim은 HumanoidVerse에
ASAP 보상과 teacher-student이고 이쪽은 BeyondMimic입니다. 같은 LAFAN1·G1·
BeyondMimic으로 돌린 Retargeting Matters가 sim2sim 성공률 대부분 100%를 받으므로,
여기의 1.000은 PolySim을 이긴 값이 아니라 BeyondMimic 계보의 통상값입니다.
