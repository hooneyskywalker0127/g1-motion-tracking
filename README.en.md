# humanoid-motion-tracking

A pipeline that moves human motion capture onto a humanoid and trains a whole-body
control policy to track it. It started on the Unitree G1; a second robot, IGRIS-C, is being added.

[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.1-silver.svg)](https://docs.isaacsim.omniverse.nvidia.com/)
[![IsaacLab](https://img.shields.io/badge/IsaacLab-2.3.2-silver.svg)](https://isaac-sim.github.io/IsaacLab/)
[![MuJoCo](https://img.shields.io/badge/MuJoCo-3.x-blue.svg)](https://mujoco.org/)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)

[**한국어**](README.md) · [**Hugging Face**](https://huggingface.co/hooneyskywalker/humanoid-motion-tracking-policies) · [**W&B**](https://wandb.ai/hooneyskywalker-humanoid) · [**YouTube**](https://www.youtube.com/playlist?list=PLdtcYiDg1nhI)

![tracking](docs/tracking.gif)

> **Simulation only.** There is no hardware, so nothing here was deployed on a real robot
> and no sim-to-real performance is claimed. Policies are trained in Isaac Lab and checked
> by moving them to MuJoCo.

I drove a G1 with NVIDIA's [GEAR-SONIC](https://github.com/NVlabs/GR00T-WholeBodyControl)
in a multi-humanoid simulation, and using a controller taught me different things from
building one. This project builds the same kind of whole-body control (reference motion
tracking) end to end.

## News

- **[2026-10-01]** [Reward ablation](docs/reward_ablation.en.md): all three groups of tracking rewards are needed. Without the anchor terms the robot drifts (1.25 m global error); without the body-pose terms it falls within two seconds. The velocity terms did more to hold global position than the anchor terms.
- **[2026-09-30]** [IGRIS-C retargeting](docs/igris.en.md), a second robot. A C++ real-time inference loop fits the 20 ms control budget with close to a tenfold margin ([robustness](docs/robustness.en.md)).
- **[2026-09-29]** [Model-mismatch sweep](docs/robustness.en.md): one step (20 ms) of latency drops completion from 99.9% to 29.6%. Fixed an evaluation bug that ran with randomization on and had inverted a comparison. Metrics renamed to the names used by the papers that define them.
- **[2026-09-27]** Measured the generalization limit: 0 of the 63 held-out LAFAN1 clips complete.
- **[2026-09-24]** [Distilled](docs/pipeline.en.md#5-policy-distillation) 14 single-motion experts into one policy. It matches or beats them on all six metrics.
- **[2026-09-14]** Teacher policies checked in MuJoCo ([sim-to-sim](docs/sim2sim.en.md)). A motion from outside LAFAN1 ([kobe](docs/kobe.en.md)) trained as well.
- **[2026-09-10]** All 17 LAFAN1 teacher policies trained.
- **[2026-09-02]** First teacher policy: 30,000 iterations in 8 h 38 min on an RTX 5080.

## Contents

- [Overview](#overview)
- [Results](#results)
- [Demos](#demos)
- [Supported robots](#supported-robots)
- [Checkpoints](#checkpoints)
- [Installation](#installation)
- [Usage](#usage)
- [Repository structure](#repository-structure)
- [TODO](#todo)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## Overview

![pipeline](docs/pipeline.png)

| Stage | What it does | Tool |
|---|---|---|
| 1. Motion capture | Human motion from [LAFAN1](https://github.com/ubisoft/ubisoft-laforge-animation-dataset): 77 sequences, 4.6 hours | — |
| 2. Retargeting | Human skeleton poses become robot joint angles. No physics | [GMR](https://github.com/YanjieZe/GMR) |
| 3. Reference selection | Pick motions the robot can follow and write 50 fps npz files | Isaac Sim |
| 4. Teacher policies | One RL policy per motion | [BeyondMimic](https://github.com/HybridRobotics/whole_body_tracking), PPO |
| 5. Consolidation | Distill 14 teachers into one student (DAgger) | Isaac Lab |
| Checks | Move to a second simulator, perturb the model, time inference | MuJoCo, C++ |

Stage-by-stage details and evaluation criteria: [docs/pipeline.en.md](docs/pipeline.en.md).

## Results

**Fourteen experts become one policy, which matches or beats them on all six metrics.**

| | Completion | E_g-mpbpe | E_mpbpe | E_mpjpe | E_mpbve | E_mpbae |
|---|---|---|---|---|---|---|
| 14 experts, each on its own clip | 99.0% | 102mm | 42mm | 0.084 | 4.78 | 2.09 |
| **one unified policy** | **99.7%** | **90mm** | **41mm** | **0.082** | **4.32** | **1.91** |

100 rollouts, domain randomization off, full clip length, identical evaluation code. Errors
are averaged over completed rollouts. Metrics follow the [GMR paper](https://arxiv.org/abs/2510.02252)
(E_g-mpbpe, E_mpbpe, E_mpjpe) and [PBHC](https://arxiv.org/abs/2506.12851) (E_mpbve, E_mpbae).

What works and what does not were both measured.

| What | Result | Details |
|---|---|---|
| The 14 training clips | 99.7% completion | [pipeline](docs/pipeline.en.md#5-policy-distillation) |
| 63 held-out LAFAN1 clips | **0 complete** — it does not generalize | [pipeline](docs/pipeline.en.md#what-it-cannot-do-was-measured-too) |
| Pushes at training strength | experts 91.0%, unified 84.3% — experts recover better | [pipeline](docs/pipeline.en.md#under-perturbation-the-experts-win) |
| Domain randomization on | 76.1% | [pipeline](docs/pipeline.en.md#5-policy-distillation) |
| Moved to MuJoCo | 12 of 14 complete | [sim2sim](docs/sim2sim.en.md) |
| One step (20 ms) of control latency | **29.6%** — the most damaging axis | [robustness](docs/robustness.en.md) |
| Torque ×0.7 / mass ×1.2 / friction ×0.5 | 56.4% / 25.0% / 70.3% | [robustness](docs/robustness.en.md) |
| Inference latency p99.9 (C++) | 1.15 ms against a 20 ms budget | [robustness](docs/robustness.en.md#real-time-inference-loop--does-it-fit-the-20-ms-budget) |
| Removing one reward group (10k iterations) | anchor 3%, body pose 0%, velocity 55% (all rewards 63%) | [reward_ablation](docs/reward_ablation.en.md) |

## Demos

<table>
  <tr>
    <td align="center"><img src="docs/retargeting.gif" width="360"/><br/>Retargeting (human → G1)</td>
    <td align="center"><img src="docs/tracking.gif" width="360"/><br/>Tracking policy (Isaac Lab)</td>
  </tr>
  <tr>
    <td align="center"><img src="docs/sim2sim.gif" width="360"/><br/>sim-to-sim (Isaac Lab | MuJoCo)</td>
    <td align="center"><img src="docs/randomization.gif" width="360"/><br/>Domain randomization</td>
  </tr>
  <tr>
    <td align="center"><img src="docs/kobe.gif" width="360"/><br/>Outside LAFAN1 (ASAP kobe)</td>
    <td align="center"><img src="docs/retarget_igris.gif" width="360"/><br/>One clip, two robots (G1 | human | IGRIS-C)</td>
  </tr>
</table>

Full videos: [full clip](https://youtu.be/l1M4y_Nl7oc) · [training progression](https://youtu.be/qQw8PtmXV9s) · [domain randomization](https://youtu.be/d61rKk675qY) · [playlist](https://www.youtube.com/playlist?list=PLdtcYiDg1nhI)

## Supported robots

| Robot | DoF | Height / mass | Retargeting | Teachers | Consolidation | sim-to-sim |
|---|---|---|---|---|---|---|
| Unitree G1 | 29 | 1.32 m / 35 kg | ✅ 77 LAFAN1 clips | ✅ 17 | ✅ 14 → 1 | ✅ MuJoCo |
| [IGRIS-C](https://github.com/robrosinc/igris_c_description_public) | 31 | 1.5 m / 58 kg | ✅ 14 clips | in progress | — | — |

For IGRIS-C, reusing the G1 policy ([cross-embodiment transfer](https://arxiv.org/abs/2605.23733))
is being compared with training from scratch. The model files have no license, so they are
linked rather than copied. Details: [docs/igris.en.md](docs/igris.en.md).

## Checkpoints

On [Hugging Face](https://huggingface.co/hooneyskywalker/humanoid-motion-tracking-policies).

| Folder | Contents |
|---|---|
| `student/` | The unified policy (`final_model.pt`, `policy.onnx`) and its evaluations. Hidden [2048, 2048, 1024, 1024, 512], 260 observations, 29 actions |
| `policies/<sequence>/` | The 17 per-clip teachers (`model_29999.pt`, `policy.onnx`) |
| `eval/`, `eval_polysim/`, `sym/` | Teacher evaluations and MuJoCo transfer evaluations |

Training curves are on [W&B](https://wandb.ai/hooneyskywalker-humanoid): teachers in the
`stage4_teachers` group of the `humanoid-motion-tracking` project, distillation in the `final`
group of `humanoid-motion-tracking-distill`, the reward ablation in group `reward_ablation_obstacles3`.

## Installation

Three environments.

| For | Environment | Install |
|---|---|---|
| Retargeting (stages 1-2) | conda `gmr`, Python 3.10 | `bash scripts/setup_gmr.sh` |
| Training and evaluation (stages 3-5) | Isaac Sim 5.1, Isaac Lab 2.3.2, rsl-rl 3.1.2 | [Isaac Lab install guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html), then BeyondMimic |
| sim-to-sim, C++ loop | MuJoCo 3.x, ONNX Runtime | `pip install mujoco onnxruntime`; C++ build in [cpp/README.md](cpp/README.md) |

Training uses a locally modified BeyondMimic. The upstream targets Isaac Sim 4.5 / Isaac Lab
2.1, so five interface points were changed; they are listed in
[docs/pipeline.en.md](docs/pipeline.en.md#4-motion-tracking-policy). That fork, which also holds
the distillation and evaluation code, is not public yet.

Scripts still carry local paths (`/home/sehoon/...`), so cloning and running as is will not
work yet ([TODO](#todo)).

## Usage

**1. Retargeting** — LAFAN1 BVH to G1 joint angles.
```bash
bash scripts/retarget_all.sh                      # data/lafan1/*.bvh -> outputs/retarget/*.pkl
python src/pkl_to_csv.py --src outputs/retarget --dst outputs/csv
```

**2. Reference motion** — replay the csv in the simulator and write 50 fps npz.
```bash
bash scripts/npz_all.sh                           # BeyondMimic scripts/csv_to_npz.py
```

**3. Teacher training** — one policy per clip.
```bash
python scripts/rsl_rl/train.py --headless --task=Tracking-Flat-G1-v0 \
    --motion_file <sequence>.npz --run_name <sequence>              # 4096 envs, 30,000 iterations
```

**4. Distillation** — 14 teachers into one student.
```bash
python scripts/distill/train_student.py --headless --seqs configs/distill_seqs.txt \
    --student_hidden 2048 2048 1024 1024 512 --max_iteration 50000
```

**5. Evaluation** — 100 rollouts, randomization off, full clip length.
```bash
python scripts/distill/eval_student.py --headless --checkpoint <student.pt> \
    --seqs configs/distill_seqs.txt --num_envs 100 --out eval.json
```
Mismatch conditions are single flags such as `--action_delay 1`, `--torque_scale 0.7`, `--mass_scale 1.2`.

**6. sim-to-sim** — run the student in MuJoCo.
```bash
python src/sim2sim_student.py walk4_subject1 --onnx student/policy.onnx --video out.mp4
```

**7. Real-time loop** — time the C++ inference loop. See [cpp/README.md](cpp/README.md).

The scripts for stages 3-5 live in the BeyondMimic fork.

## Repository structure

```
src/        retargeting, metrics, rendering, tables, sim-to-sim (MuJoCo)
scripts/    batch drivers
  igris/    second robot: model preparation, IK table, joint mapping
cpp/        real-time inference loop (C++) and its Python control
configs/    selection lists, training order
docs/       stage-by-stage documents and the figures in this README
outputs/    generated results (gitignored)
data ->     dataset symlink (gitignored)
```

| Document | Contents |
|---|---|
| [docs/pipeline.en.md](docs/pipeline.en.md) | stages 1-5, evaluation criteria, why three clips never finish |
| [docs/sim2sim.en.md](docs/sim2sim.en.md) | MuJoCo transfer, the two pass criteria, transfer loss |
| [docs/robustness.en.md](docs/robustness.en.md) | model-mismatch sweep, real-time inference loop |
| [docs/igris.en.md](docs/igris.en.md) | second-robot retargeting and joint mapping |
| [docs/kobe.en.md](docs/kobe.en.md) | background, a motion from outside LAFAN1 |
| [docs/reward_ablation.en.md](docs/reward_ablation.en.md) | removing each group of tracking rewards in turn |
| [docs/data.en.md](docs/data.en.md) | data sources and why they were chosen |

## TODO

- [x] Retarget all 77 LAFAN1 clips, train 17 teachers
- [x] Distill 14 teachers into one policy
- [x] MuJoCo sim-to-sim
- [x] Measure the limits: generalization, perturbation, model mismatch
- [x] C++ real-time inference loop
- [x] Retarget onto a second robot, IGRIS-C
- [x] Reward ablation — remove each group of tracking rewards in turn
- [ ] IGRIS-C policy — from scratch vs. transferred from the G1 policy (in progress)
- [ ] Latency in training — randomize it, or put command history in the observation
- [ ] 14 → 60 training motions with the rest held out
- [ ] Perturbation in the DAgger rollouts (recovery)
- [ ] Redo retargeting selection on ground penetration, foot slip and joint-velocity violations
- [ ] Remove local paths so the repository runs after cloning
- [ ] Real robot

## Acknowledgements

- [GMR](https://github.com/YanjieZe/GMR) (MIT) — retargeting. The second robot's IK table is derived from its G1 table.
- [BeyondMimic](https://github.com/HybridRobotics/whole_body_tracking) (MIT) — teacher training code and reward design.
- [Isaac Lab](https://github.com/isaac-sim/IsaacLab), [rsl_rl](https://github.com/leggedrobotics/rsl_rl), [MuJoCo](https://github.com/google-deepmind/mujoco).
- [LAFAN1](https://github.com/ubisoft/ubisoft-laforge-animation-dataset) — motion data. [ASAP](https://github.com/LeCAR-Lab/ASAP) — the kobe motion.
- The Unitree G1 model is the `unitree_description` BeyondMimic uses. The IGRIS-C model is [robrosinc/igris_c_description_public](https://github.com/robrosinc/igris_c_description_public) (no license stated; not copied).
- Evaluation and comparison follow [Retargeting Matters (GMR)](https://arxiv.org/abs/2510.02252), [PBHC](https://arxiv.org/abs/2506.12851), [PolySim](https://arxiv.org/abs/2510.01708), [Any2Any](https://arxiv.org/abs/2605.23733) and [GEAR-SONIC](https://arxiv.org/abs/2511.07820).

## License

The code in this repository is [MIT](LICENSE). Datasets and robot models follow their own terms.
