# humanoid-motion-tracking

사람의 모션 캡처를 휴머노이드로 옮기고, 그 동작을 따라가는 전신 제어 정책을
학습하는 파이프라인입니다. Unitree G1으로 시작했고 두 번째 로봇 IGRIS-C로 넓혔습니다.

[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.1-silver.svg)](https://docs.isaacsim.omniverse.nvidia.com/)
[![IsaacLab](https://img.shields.io/badge/IsaacLab-2.3.2-silver.svg)](https://isaac-sim.github.io/IsaacLab/)
[![MuJoCo](https://img.shields.io/badge/MuJoCo-3.x-blue.svg)](https://mujoco.org/)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)

[**English**](README.en.md) · [**Hugging Face**](https://huggingface.co/hooneyskywalker/humanoid-motion-tracking-policies) · [**W&B**](https://wandb.ai/hooneyskywalker-humanoid) · [**YouTube**](https://www.youtube.com/playlist?list=PLdtcYiDg1nhI)

![tracking](docs/tracking.gif)

> **범위는 시뮬레이션까지입니다.** 하드웨어가 없어 실제 로봇 배포는 하지 않았고,
> sim-to-real 성능을 주장하지 않습니다. Isaac Lab에서 학습하고 MuJoCo로 옮겨 검증합니다.

군집 휴머노이드 시뮬레이션에서 NVIDIA의
[GEAR-SONIC](https://github.com/NVlabs/GR00T-WholeBodyControl)으로 G1을 제어해 봤는데,
쓰는 것과 만드는 것은 아는 것이 달랐습니다. 같은 계열(레퍼런스 모션 추종)의 전신
제어를 처음부터 끝까지 직접 만들어 보려고 시작했습니다.

## News

- **[2026-10-02]** [IGRIS-C 리타게팅에서 고친 두 곳](docs/igris.md): G1 설정을 그대로 쓰면 팔꿈치가 쭉 펴진 채 굳고, 다리를 발목까지 재면 걷는 내내 발이 바닥 아래 1.8cm에 박힙니다. 로봇을 바꿀 때 손봐야 하는 곳이 리타게팅 표입니다.
- **[2026-10-02]** [IGRIS-C 정책](docs/igris.md#처음부터-학습-vs-g1-정책에서-옮기기): 같은 달리기 클립을 처음부터 학습한 것(완주 96%)과 G1 정책을 Any2Any 방식으로 옮긴 것(16%)을 비교했습니다. 옮긴 쪽이 초반엔 10배 빨리 배우지만 낮게 멈춥니다. 처음엔 자기 충돌 때문에 전혀 학습되지 않았습니다.
- **[2026-10-01]** [보상 ablation](docs/reward_ablation.md): 추종 보상 세 묶음 모두 필요합니다. 앵커 항을 빼면 표류(전역 오차 1.25m), 몸체 자세 항을 빼면 2초 만에 쓰러집니다. 전역 위치를 붙잡는 몫은 앵커 항보다 속도 항이 더 컸습니다.
- **[2026-09-30]** 두 번째 로봇 [IGRIS-C 리타게팅](docs/igris.md). C++ 실시간 추론 루프가 20ms 제어 예산에 10배 가까운 여유로 듭니다([견고성](docs/robustness.md)).
- **[2026-09-29]** [모델 불일치 민감도](docs/robustness.md): 지연 한 스텝(20ms)에 완주율 99.9% → 29.6%. 평가 코드가 랜덤화를 켠 채 돌던 버그를 고쳐 비교를 바로잡았습니다. 지표 이름을 정의한 논문의 것으로 바꿨습니다.
- **[2026-09-27]** 일반화 한계 측정: LAFAN1의 학습에 없던 63개 클립 중 완주 0개.
- **[2026-09-24]** 단일 동작 전문가 14개를 정책 하나로 [증류](docs/pipeline.md#5단계-정책-통합). 여섯 지표 전부 같거나 낫습니다.
- **[2026-09-14]** 교사 정책을 MuJoCo로 옮겨 검증([sim-to-sim](docs/sim2sim.md)). LAFAN1 밖의 동작([kobe](docs/kobe.md))도 학습.
- **[2026-09-10]** LAFAN1 교사 정책 17개 학습 완료.
- **[2026-09-02]** 첫 교사 정책. RTX 5080에서 30,000회에 8시간 38분.

## 목차

- [개요](#개요)
- [결과](#결과)
- [데모](#데모)
- [지원 로봇](#지원-로봇)
- [체크포인트](#체크포인트)
- [설치](#설치)
- [사용법](#사용법)
- [폴더 구조](#폴더-구조)
- [TODO](#todo)
- [감사의 말](#감사의-말)
- [라이선스](#라이선스)

## 개요

![pipeline](docs/pipeline.png)

| 단계 | 하는 일 | 도구 |
|---|---|---|
| 1. 모션 캡처 | 사람 동작 데이터 [LAFAN1](https://github.com/ubisoft/ubisoft-laforge-animation-dataset) 77개 시퀀스, 4.6시간 | — |
| 2. 리타게팅 | 사람 뼈대의 자세를 로봇 관절각으로 옮깁니다. 물리 없음 | [GMR](https://github.com/YanjieZe/GMR) |
| 3. 레퍼런스 선별 | 따라갈 수 있는 모션을 고르고 50 fps npz로 만듭니다 | Isaac Sim |
| 4. 교사 정책 | 모션 하나당 정책 하나를 강화학습으로 학습합니다 | [BeyondMimic](https://github.com/HybridRobotics/whole_body_tracking), PPO |
| 5. 정책 통합 | 교사 14개를 학생 정책 하나로 증류합니다(DAgger) | Isaac Lab |
| 검증 | 다른 시뮬레이터로 옮기고, 모델 불일치를 흔들고, 추론 지연을 잽니다 | MuJoCo, C++ |

단계별 설명과 평가 기준은 [docs/pipeline.md](docs/pipeline.md)에 있습니다.

## 결과

**전문가 14개를 하나로 합쳤고, 여섯 지표 전부 같거나 낫습니다.**

| | 완주율 | E_g-mpbpe | E_mpbpe | E_mpjpe | E_mpbve | E_mpbae |
|---|---|---|---|---|---|---|
| 전문가 14개 (각자 자기 클립) | 99.0% | 102mm | 42mm | 0.084 | 4.78 | 2.09 |
| **통합 정책 하나** | **99.7%** | **90mm** | **41mm** | **0.082** | **4.32** | **1.91** |

100 롤아웃, 도메인 랜덤화 끔, 클립 전체 길이, 같은 평가 코드입니다. 오차는 완주한
롤아웃만 평균낸 값입니다. 지표 정의는 [GMR 논문](https://arxiv.org/abs/2510.02252)
(E_g-mpbpe, E_mpbpe, E_mpjpe)과 [PBHC](https://arxiv.org/abs/2506.12851)(E_mpbve, E_mpbae)를 따릅니다.

되는 것과 안 되는 것을 함께 쟀습니다.

| 무엇 | 결과 | 자세히 |
|---|---|---|
| 학습한 14클립 | 완주 99.7% | [pipeline](docs/pipeline.md#5단계-정책-통합) |
| 학습에 없던 LAFAN1 63클립 | **완주 0개** — 일반화는 안 됩니다 | [pipeline](docs/pipeline.md#되지-않는-것도-쟀습니다) |
| 밀치기(학습과 같은 세기) | 전문가 91.0%, 통합 84.3% — 복구력은 전문가가 낫습니다 | [pipeline](docs/pipeline.md#교란-아래에서는-전문가가-낫습니다) |
| 도메인 랜덤화 켠 채 | 76.1% | [pipeline](docs/pipeline.md#5단계-정책-통합) |
| MuJoCo로 전이 | 14개 중 12개 완주 | [sim2sim](docs/sim2sim.md) |
| 제어 지연 1스텝(20ms) | **29.6%** — 가장 치명적 | [robustness](docs/robustness.md) |
| 토크 ×0.7 / 질량 ×1.2 / 마찰 ×0.5 | 56.4% / 25.0% / 70.3% | [robustness](docs/robustness.md) |
| 추론 지연 p99.9 (C++) | 1.15ms, 예산 20ms | [robustness](docs/robustness.md#실시간-추론-루프--20ms-예산-안에-드는가) |
| IGRIS-C 처음부터 vs G1 정책 옮기기 (run2, 1000프레임부터) | 96% vs 16% — 이 설정에선 처음부터가 이김 | [igris](docs/igris.md#처음부터-학습-vs-g1-정책에서-옮기기) |
| 보상 한 묶음씩 빼기 (1만 회) | 앵커 3%, 몸체 자세 0%, 속도 55% (전부 쓴 쪽 63%) | [reward_ablation](docs/reward_ablation.md) |

## 데모

<table>
  <tr>
    <td align="center"><img src="docs/demo/retargeting.gif" width="360"/><br/>리타게팅 (사람 → G1)</td>
    <td align="center"><img src="docs/demo/tracking.gif" width="360"/><br/>추종 정책 (Isaac Lab)</td>
  </tr>
  <tr>
    <td align="center"><img src="docs/demo/sim2sim.gif" width="360"/><br/>sim-to-sim (Isaac Lab | MuJoCo)</td>
    <td align="center"><img src="docs/demo/randomization.gif" width="360"/><br/>도메인 랜덤화</td>
  </tr>
  <tr>
    <td align="center"><img src="docs/demo/kobe.gif" width="360"/><br/>LAFAN1 밖의 동작 (ASAP kobe)</td>
    <td align="center"><img src="docs/demo/igris_scratch_vs_transfer.gif" width="360"/><br/>IGRIS-C: 처음부터 학습 | G1 정책에서 옮기기</td>
  </tr>
</table>

전체 영상: [전체 클립](https://youtu.be/l1M4y_Nl7oc) · [학습 경과](https://youtu.be/qQw8PtmXV9s) · [도메인 랜덤화](https://youtu.be/d61rKk675qY) · [재생목록](https://www.youtube.com/playlist?list=PLdtcYiDg1nhI)

## 지원 로봇

| 로봇 | 자유도 | 키 / 무게 | 리타게팅 | 교사 학습 | 통합 | sim-to-sim |
|---|---|---|---|---|---|---|
| Unitree G1 | 29 | 1.32m / 35kg | ✅ LAFAN1 77개 | ✅ 14개 | ✅ 14 → 1 | ✅ MuJoCo |
| [IGRIS-C](https://github.com/robrosinc/igris_c_description_public) | 31 | 1.5m / 58kg | ✅ 14개 | ✅ 1개 (run2) | — | — |

![igris_transfer](docs/igris_transfer.png)

IGRIS-C에서 G1 정책을 옮겨 쓰는 것([cross-embodiment transfer](https://arxiv.org/abs/2605.23733))과
처음부터 학습하는 것을 비교했습니다. 모델 파일에 라이선스가 없어 저장소에 복사하지 않고
원 저장소를 링크합니다. 자세한 것은 [docs/igris.md](docs/igris.md).

## 체크포인트

[Hugging Face](https://huggingface.co/hooneyskywalker/humanoid-motion-tracking-policies)에 있습니다.

| 폴더 | 내용 |
|---|---|
| `student/` | 통합 정책 하나(`final_model.pt`, `policy.onnx`)와 평가 결과. 은닉 [2048, 2048, 1024, 1024, 512], 관측 260, 행동 29 |
| `policies/<시퀀스>/` | 클립별 교사 정책 17개(`model_29999.pt`, `policy.onnx`) |
| `eval/`, `eval_polysim/`, `sym/` | 교사 평가, MuJoCo 전이 평가 |
| `igris_c/` | IGRIS-C 정책 두 개(처음부터, Any2Any)와 체크포인트별 평가, 관절 대응표 |
| `reward_ablation/` | 보상 ablation 네 조건의 체크포인트와 평가 |
| `tables/`, `media/` | 결과 표, 그림과 GIF |

학습 곡선은 [W&B](https://wandb.ai/hooneyskywalker-humanoid)에 있습니다. 교사는
`humanoid-motion-tracking` 프로젝트의 `stage4_teachers` 그룹, 증류는 `humanoid-motion-tracking-distill` 프로젝트의
`final` 그룹, 보상 ablation은 `reward_ablation_obstacles3`, IGRIS-C는 `igris_c_transfer` 그룹입니다.

## 설치

환경이 셋입니다.

| 용도 | 환경 | 설치 |
|---|---|---|
| 리타게팅 (1-2단계) | conda `gmr`, Python 3.10 | `bash scripts/setup_gmr.sh` |
| 학습·평가 (3-5단계) | Isaac Sim 5.1, Isaac Lab 2.3.2, rsl-rl 3.1.2 | [Isaac Lab 설치 안내](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html) 후 BeyondMimic 설치 |
| sim-to-sim, C++ 루프 | MuJoCo 3.x, ONNX Runtime | `pip install mujoco onnxruntime`, C++ 빌드는 [cpp/README.md](cpp/README.md) |

학습은 BeyondMimic을 로컬에서 고쳐 씁니다. 원 저장소는 Isaac Sim 4.5 / Isaac Lab 2.1
기준이라 인터페이스 다섯 곳을 고쳤고, 고친 곳은 [docs/pipeline.md](docs/pipeline.md#4단계-모션-트래킹-정책-학습)에
적었습니다. 증류와 평가 코드를 포함한 그 포크는 아직 공개하지 않았습니다.

스크립트에 로컬 경로(`/home/sehoon/...`)가 박혀 있어 그대로 복제해 돌리기는 아직
어렵습니다([TODO](#todo)).

## 사용법

**1. 리타게팅** — LAFAN1 bvh를 G1 관절각으로 바꿉니다.
```bash
bash scripts/retarget_all.sh                      # data/lafan1/*.bvh -> outputs/retarget/*.pkl
python src/pkl_to_csv.py --src outputs/retarget --dst outputs/csv
```

**2. 레퍼런스 모션** — csv를 물리 시뮬레이터에서 재생해 50 fps npz로 만듭니다.
```bash
bash scripts/npz_all.sh                           # BeyondMimic scripts/csv_to_npz.py
```

**3. 교사 정책 학습** — 클립 하나에 정책 하나.
```bash
python scripts/rsl_rl/train.py --headless --task=Tracking-Flat-G1-v0 \
    --motion_file <시퀀스>.npz --run_name <시퀀스>                    # 4096 env, 30,000회
```

**4. 증류** — 교사 14개를 학생 하나로.
```bash
python scripts/distill/train_student.py --headless --seqs configs/distill_seqs.txt \
    --student_hidden 2048 2048 1024 1024 512 --max_iteration 50000
```

**5. 평가** — 100 롤아웃, 랜덤화 끔, 클립 전체 길이.
```bash
python scripts/distill/eval_student.py --headless --checkpoint <학생.pt> \
    --seqs configs/distill_seqs.txt --num_envs 100 --out eval.json
```
불일치 조건은 `--action_delay 1`, `--torque_scale 0.7`, `--mass_scale 1.2` 같은 인자를 하나씩 켭니다.

**6. sim-to-sim** — 학생 정책을 MuJoCo에서 돌립니다.
```bash
python src/sim2sim_student.py walk4_subject1 --onnx student/policy.onnx --video out.mp4
```

**7. 실시간 루프** — C++ 추론 루프의 지연을 잽니다. [cpp/README.md](cpp/README.md)

3-5단계의 스크립트는 BeyondMimic 포크 쪽에 있습니다.

## 폴더 구조

```
src/        리타게팅, 지표, 렌더, 표 생성, sim-to-sim (MuJoCo)
scripts/    배치 실행 스크립트
  igris/    두 번째 로봇: 모델 준비, IK 표, 관절 대응
cpp/        실시간 추론 루프(C++)와 파이썬 대조군
configs/    선별 목록, 학습 순서
docs/       단계별 상세 문서와 이 README의 그림
outputs/    생성 결과 (gitignored)
data ->     데이터 심볼릭 링크 (gitignored)
```

| 문서 | 내용 |
|---|---|
| [docs/pipeline.md](docs/pipeline.md) | 1-5단계 상세, 평가 기준, 완주하지 못한 클립의 원인 |
| [docs/sim2sim.md](docs/sim2sim.md) | MuJoCo 전이, 두 가지 합격 기준, 전이 손실 |
| [docs/robustness.md](docs/robustness.md) | 모델 불일치 민감도, 실시간 추론 루프 |
| [docs/igris.md](docs/igris.md) | 두 번째 로봇 리타게팅과 관절 대응 |
| [docs/kobe.md](docs/kobe.md) | 배경, LAFAN1 밖의 동작 |
| [docs/reward_ablation.md](docs/reward_ablation.md) | 추종 보상 세 묶음을 하나씩 뺀 비교 |
| [docs/data.md](docs/data.md) | 데이터 출처와 선택 이유 |

## TODO

- [x] LAFAN1 77개 리타게팅, 교사 정책 14개
- [x] 교사 14개 → 통합 정책 하나 (증류)
- [x] MuJoCo sim-to-sim
- [x] 일반화·교란·모델 불일치 한계 측정
- [x] C++ 실시간 추론 루프
- [x] 두 번째 로봇 IGRIS-C 리타게팅
- [x] 보상 항 ablation — 추종 보상 세 묶음을 하나씩 빼고 비교
- [x] IGRIS-C 정책 — 처음부터 학습 vs G1 정책에서 옮기기
- [ ] 학습에 지연 넣기 — 도메인 랜덤화에 지연, 또는 명령 이력을 관측에
- [ ] 학습 모션 14 → 60개, 나머지를 held-out으로
- [ ] DAgger 롤아웃에 교란 넣기 (복구력)
- [ ] 리타게팅 선별 기준 다시 잡기 — 지면 관통, 발 미끄럼, 관절속도 위반
- [ ] 로컬 경로를 떼어 내 복제해 돌릴 수 있게
- [ ] 실제 로봇

## 감사의 말

- [GMR](https://github.com/YanjieZe/GMR) (MIT) — 리타게팅. 두 번째 로봇의 IK 표도 G1 표에서 만들었습니다.
- [BeyondMimic](https://github.com/HybridRobotics/whole_body_tracking) (MIT) — 교사 정책 학습 코드와 보상 설계.
- [Isaac Lab](https://github.com/isaac-sim/IsaacLab), [rsl_rl](https://github.com/leggedrobotics/rsl_rl), [MuJoCo](https://github.com/google-deepmind/mujoco).
- [LAFAN1](https://github.com/ubisoft/ubisoft-laforge-animation-dataset) — 모션 데이터. [ASAP](https://github.com/LeCAR-Lab/ASAP) — kobe 모션.
- Unitree G1 모델은 BeyondMimic이 쓰는 `unitree_description`, IGRIS-C 모델은 [robrosinc/igris_c_description_public](https://github.com/robrosinc/igris_c_description_public)(라이선스 표기 없음, 복사하지 않음).
- 평가와 비교의 기준: [Retargeting Matters (GMR)](https://arxiv.org/abs/2510.02252), [PBHC](https://arxiv.org/abs/2506.12851), [PolySim](https://arxiv.org/abs/2510.01708), [Any2Any](https://arxiv.org/abs/2605.23733), [GEAR-SONIC](https://arxiv.org/abs/2511.07820).

## 라이선스

이 저장소의 코드는 [MIT](LICENSE)입니다. 데이터셋과 로봇 모델은 각 원 저장소의 조건을 따릅니다.
