# Reward ablation — what each tracking reward holds up

[← README](../README.en.md)

The rewards are [BeyondMimic](https://github.com/HybridRobotics/whole_body_tracking)'s, never touched until
now. The six tracking terms were split into three pairs; each pair was zeroed in turn and the same clip
trained from scratch. Position and orientation measure the same thing in different coordinates, so removing
only one lets the other compensate — hence pairs.

| Condition | Zeroed |
|---|---|
| base | none |
| no_anchor | `motion_global_anchor_pos`, `motion_global_anchor_ori` — global pose of the pelvis (anchor) |
| no_body | `motion_body_pos`, `motion_body_ori` — body pose relative to the anchor |
| no_vel | `motion_body_lin_vel`, `motion_body_ang_vel` — global body linear and angular velocity |

`obstacles3_subject3` (131 s), seed 42, 4096 environments, 10,000 PPO iterations, same code. The
regularization terms (action rate, joint limits, undesired contacts) were left as they were.

## Results

100 rollouts, domain randomization off, from frame 0 to the end of the clip.

| Condition | Completion | E_g-mpbpe | E_mpbpe | E_mpjpe | Mean frames survived (/6554) |
|---|---|---|---|---|---|
| base | 63% | 337mm | 60mm | 0.122 | 4387 |
| no_anchor | **3%** | **1255mm** | 61mm | 0.127 | **369** |
| no_body | **0%** | — | — | — | **91** |
| no_vel | 55% | **956mm** | 60mm | 0.122 | 4419 |

Errors are averaged over completed rollouts, so read them with completion.

**All three pairs are needed**, and each fails differently.

- **Without the anchor terms it drifts.** Rollouts that finish keep base-level posture error (E_mpbpe
  61 mm) but are 1.25 m off globally: the right pose in the wrong place.
- **Without the body-pose terms it falls within two seconds.** The end-effector height termination fires
  first (0.42 of training episodes, base 0.18).
- **Without the velocity terms completion holds, but global position error triples.**

## Two predictions were wrong

**1. I expected completion to survive without the anchor terms**, since the fall check looks only at pelvis
height, not horizontal position. It dropped to 3%. In training it lasts the full 10 s window 73% of the
time; run from frame 0 for 131 s it falls after 7 s on average. The gap between training (10 s windows,
random starts) and evaluation (start to end) looks like the cause, but I did not verify it.

**2. I thought the anchor terms were what held global position.** The training-time anchor error without
the velocity terms (2.10 m) is larger than without the anchor terms (0.95 m). The velocity rewards measure
error in the world frame, so over time they pin position too. On this clip they do more against drift than
the anchor terms.

| Condition | Training episode length (/500) | Timeout fraction | Anchor position error |
|---|---|---|---|
| base | 408 | 0.80 | 0.28 m |
| no_anchor | 376 | 0.73 | 0.95 m |
| no_body | 326 | 0.53 | 0.42 m |
| no_vel | 392 | 0.76 | **2.10 m** |

## Limits

- **The base policy is not converged either**: 63% completion at 10,000 iterations (the 30,000-iteration
  expert, trained on a different npz, reaches 98%). So this compares **what each variant learned in the same
  10,000-iteration budget**, not converged policies. Whether the ranking holds after convergence is unknown.
- One clip, one seed.

## Data

- Training curves: [W&B](https://wandb.ai/hooneyskywalker-humanoid/g1-motion-tracking), group `reward_ablation_obstacles3`
- Checkpoints and evaluations: [Hugging Face](https://huggingface.co/hooneyskywalker/humanoid-motion-tracking-policies/tree/main/reward_ablation) `reward_ablation/`
