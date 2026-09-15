# g1-motion-tracking

Human mocap retargeting and whole-body motion tracking pipeline for the Unitree G1.

Simulation only. Real-robot deployment is out of scope for now.

한국어 문서는 [README.md](README.md)를 참고하십시오.

Trained policies and evaluation results: [Hugging Face](https://huggingface.co/hooneyskywalker/g1-motion-tracking-policies)

All videos: [YouTube playlist](https://www.youtube.com/playlist?list=PLLNhmCfT2kPI)

---

## Pipeline

![pipeline](docs/pipeline.png)

Stages 1 to 3 carry no physics: they compute poses and decide which ones are
worth keeping. Physics enters at stage 4, where the robot has to hold itself up.

### 1. Human mocap

Motion capture records where every joint of a human body was at each instant,
as numbers rather than video.

Done: LAFAN1, 77 BVH sequences at 30 fps, five subjects, 4.6 hours.

### 2. Retargeting

Human joint data cannot be applied to the G1 directly: link lengths differ,
joint axes differ, and some human joints have no robot counterpart. Retargeting
solves for the robot joint angles that best reproduce the original motion while
respecting the robot's joint limits and keeping the feet on the ground.

![retargeting](docs/retargeting.gif)

Orange is the source human skeleton straight out of the mocap file; grey is the
retargeted G1. The bands carry the segment lengths that make this hard and the
distance between each tracked body and the target the IK was solving for. The
robot is not walking here — each frame's joint angles are written straight into
the model and rendered. No simulation step runs.

Done: all 77 sequences retargeted with
[GMR](https://github.com/YanjieZe/GMR), 496,672 frames. Verified across every
frame: no NaNs or infinities, no joint-limit violations on any of the 29 joints,
frame counts matching the source BVH exactly.

### 3. Reference motion

The retargeted trajectory becomes the target the robot is asked to follow.
At this point nothing is physically simulated yet — only the goal exists.

Not every retargeted sequence is worth training on. The measure used here is how
far each tracked body ends up from the target GMR's IK was solving for. Only
pelvis, ankles and wrists are read: torso, shoulder and hip carry position
weights of 0 to 5, and their MuJoCo body origins do not coincide with the human
joint centres, so the constant offset there is not error.

Foot error by motion type, in cm:

```
walk         12 seqs  1.00      obstacles      17 seqs  1.65
dance         8 seqs  1.25      sprint          2 seqs  1.67
aiming        5 seqs  1.27      fallAndGetUp    6 seqs  2.11
run           4 seqs  1.33      ground          5 seqs  2.76
```

Walking is cleanest and floor work is worst, by a factor of three. Falling and
lying down put contact on parts other than the feet, which is not what an
ankle-weighted IK is set up for. Hand error stays at 5-9 cm regardless of motion
type — that is the arm-length gap, not a per-motion failure, so it is not used
to filter.

Done: 19 sequences clear all three foot thresholds — mean under 1.2 cm, p95
under 3.5 cm, max under 10 cm. The list is in
[`configs/selected_motions.txt`](configs/selected_motions.txt). Each was
converted to the 50 fps npz BeyondMimic reads, which adds the link velocities
the policy needs, and uploaded to a W&B registry.

The thresholds are not principled. They were picked because they leave 19
sequences, close to the 21 the GMR paper trained on. Once training shows which
sequences fail and where their error sits, the cut can be argued for.

### 4. Motion tracking policy

The robot is trained with reinforcement learning: it is rewarded for staying
close to the reference motion and penalised for drifting away or falling. The
control rule it converges on is the policy. Success is measured by tracking
accuracy, not by whether the walk merely looks plausible.

Training uses [BeyondMimic](https://github.com/HybridRobotics/whole_body_tracking),
the same framework the GMR paper used. It trains one policy per motion, so the
19 sequences mean 19 training runs. 4096 environments, 30000 iterations, PPO.

That repository targets IsaacSim 4.5 and IsaacLab 2.1.0. Running it against the
local IsaacSim 5.1 / IsaacLab 2.3.2 / rsl-rl 3.1.2 needed three interface
fixes, none of which touch the learning itself: `csv_to_npz.py` never leaves its
`while simulation_app.is_running()` loop after saving, which does not end under
`--headless`; `isaaclab.utils.io` no longer exports `dump_pickle`; and rsl-rl 3.x
moved the observation normalizer off the runner and into the policy.

Playing a trained policy back needs two more fixes in the same file,
`scripts/rsl_rl/play.py`. `--motion_file` is only read on the W&B branch, so a
local checkpoint starts with no reference motion, and `get_observations()` now
returns a TensorDict that the existing tuple unpacking splits along the batch
dimension, leaving a one-dimensional observation.

Done: the first policy, walk2_subject4, trained to 30000 iterations in 8 h 38 m
on an RTX 5080.

![tracking](docs/tracking.gif)

The gif above is six seconds of it. The whole sequence runs 3 minutes 58
seconds.

### ▶ Full clip (YouTube)

[![Play the full clip](docs/youtube_thumb.jpg)](https://youtu.be/l1M4y_Nl7oc)

Click the image above to play it on YouTube — https://youtu.be/l1M4y_Nl7oc

Left is the trained policy stepping through physics, right is the reference it
was asked to follow. Both panels are Isaac Sim under the same lighting and
camera, start from the same motion frame, and run the full 11,909 frame
sequence. They never match pixel for pixel: the initial pose is randomised at
every reset, and the left robot has to hold itself up while the right one is
posed frame by frame.

### ▶ Training progression (YouTube)

[![Play the training progression](docs/youtube_thumb_progression.jpg)](https://youtu.be/qQw8PtmXV9s)

Click the image above to play it on YouTube — https://youtu.be/qQw8PtmXV9s

Five checkpoints of the same run and the reference, played at once on the same
motion, the same start frame and the same camera. Mean reward runs 14.67 at
1,000 iterations, 30.80 at 5,000, 33.49 at 10,000, 36.63 at 20,000 and 36.80 at
30,000, so most of the gain lands early and the later checkpoints separate on
how long they hold rather than on the number.

### ▶ Domain randomization (YouTube)

[![Play the domain randomization clip](docs/youtube_thumb_randomization.jpg)](https://youtu.be/d61rKk675qY)

Click the image above to play it on YouTube — https://youtu.be/d61rKk675qY

![randomization](docs/randomization.gif)

Six seconds out of it. On the left is the policy running with domain
randomization on, on the right is the reference. A random push lands every 1-3
seconds, and friction, torso centre of mass, joint offsets and the reset pose
are randomized as well.

A push only adds to the torso velocity, so nothing about it is visible on
screen. The moment the value goes in, the contact point is marked, an arrow is
drawn along the push direction, and the added speed is written next to it. The
arrow stays for one second.

With randomization on, the policy can drift off the reference or the episode can
end early. It then restarts from frame 0, so the two sides fall out of phase.
That is part of the result too.

### Evaluation criteria

Policies are measured the way the GMR paper
([arXiv:2510.02252](https://arxiv.org/abs/2510.02252)) measures them. It uses
the same dataset and the same training framework, so the numbers can be placed
side by side.

| metric | meaning |
| --- | --- |
| completion rate | fraction of rollouts that reach the end of the clip without the anchor body's height or orientation passing its threshold |
| E_g-mpbpe | mean body position error in global coordinates (mm) |
| E_mpbpe | mean body position error after aligning on the anchor (mm) |
| E_mpjpe | mean joint angle error (rad) |

`scripts/eval_all.sh` measures, `src/eval_table.py` builds the table. Domain
randomization is off by default and `--randomize` turns it on, matching the way
the paper separates its sim and sim-dr conditions.

17 of the 19 selected sequences were trained to 30,000 iterations and
evaluated over 100 rollouts. Completion rate sits next to foot error, to see
whether retargeting quality predicts whether the policy holds.

| sequence | foot error (cm) | completion | E_g-mpbpe (mm) | E_mpbpe (mm) | E_mpjpe (rad) |
| --- | --- | --- | --- | --- | --- |
| walk4_subject1 | 0.88 | 100% | 60 | 37 | 0.062 |
| walk3_subject2 | 0.99 | 100% | 79 | 34 | 0.071 |
| walk1_subject2 | 1.05 | 100% | 79 | 34 | 0.066 |
| walk3_subject5 | 1.11 | 100% | 85 | 36 | 0.082 |
| aiming1_subject1 | 0.74 | 100% | 89 | 35 | 0.080 |
| walk1_subject5 | 1.12 | 100% | 90 | 34 | 0.069 |
| dance2_subject3 | 0.94 | 100% | 103 | 45 | 0.104 |
| walk2_subject1 | 0.91 | 100% | 114 | 43 | 0.101 |
| walk1_subject1 | 0.80 | 99% | 80 | 34 | 0.066 |
| walk2_subject4 | 0.70 | 99% | 90 | 40 | 0.085 |
| run2_subject4 | 1.12 | 99% | 178 | 47 | 0.111 |
| jumps1_subject1 | 1.10 | 98% | 151 | 42 | 0.104 |
| obstacles3_subject3 | 1.19 | 98% | 162 | 54 | 0.101 |
| walk2_subject3 | 1.15 | 96% | 129 | 50 | 0.104 |
| walk3_subject4 | 0.88 | 0% | - | - | - |
| obstacles2_subject1 | 0.93 | 0% | - | - | - |
| walk3_subject1 | 1.01 | 0% | - | - | - |
| obstacles1_subject1 | 1.07 | not trained |  |  |  |
| obstacles4_subject2 | 1.19 | not trained |  |  |  |

The three sequences at 0% have no completed rollout, so the three metrics are
undefined. Mean tracked length and E_g-mpbpe over all rollouts instead:

| sequence | mean tracked length | E_g-mpbpe over all rollouts (mm) |
| --- | --- | --- |
| walk3_subject4 | 88% (10,862 / 12,330 frames) | 116 |
| walk3_subject1 | 78% (9,606 / 12,330 frames) | 112 |
| obstacles2_subject1 | 18% (2,144 / 12,204 frames) | 368 |

Foot error ranking did not predict policy performance. The three sequences at
0% sit mid-range at 0.88, 0.93 and 1.01 cm, while obstacles3_subject3 at the
high end (1.19 cm) completes 98% of rollouts. walk4_subject1, which has the
lowest E_g-mpbpe at 60 mm, is at 0.88 cm rather than the top of the list.

It is clearer on video. Once a termination condition fires the episode ends and
restarts from frame 0, so on screen the robot snaps back to its initial pose.

![walk3_subject1 reset](docs/reset_walk3_subject1.gif)

walk3_subject1 at 195 s, where the anchor height crosses its threshold.

![walk3_subject4 reset](docs/reset_walk3_subject4.gif)

walk3_subject4 at 218 s, where an ankle or wrist height crosses its threshold.

A 0% completion rate does not mean the policy never follows the reference. It
follows for over three minutes and then catches on one moment. That is what the
78% and 88% mean tracked lengths are describing.

### The three that never finish are a reference problem, not a training one

Re-measuring the references with `src/motion_defect_census.py` separates them
at a glance.

| sequence | ground penetration | max penetration | peak foot slip | airborne |
| --- | --- | --- | --- | --- |
| obstacles2_subject1 | 0.9% | 4.2 cm | 0.35 m/s | 26.8% |
| walk3_subject1 | 6.0% | 7.6 cm | 1.59 m/s | 0.2% |
| walk3_subject4 | 4.3% | 4.8 cm | 1.08 m/s | 0.2% |
| walk1_subject1 (completes) | 0.0% | 0.4 cm | 0.90 m/s | 0.0% |

`obstacles2_subject1` spends 26.8% of its frames airborne: the actor is
climbing stairs and the training ground is flat, so there is nothing for the
feet to land on. The other two push their feet into the floor, up to 7.6 cm.
The sequence that completes penetrates 0% of the time.

The policy is not failing to follow the reference. The reference is not
reachable. Selection looked at foot error alone, and on that ranking these
three sit mid-range, so they passed. Ground penetration and airborne fraction
were never checked.

### The work that reports 96-100% on the same material does two more things

Retargeting Matters([arXiv:2510.02252](https://arxiv.org/abs/2510.02252))
reports 96-100% over 100 sim rollouts on the same LAFAN1, the same G1 and the
same BeyondMimic. The paper states both differences directly.

It leaves the problem motions out to begin with:

> We do not include motions with complex interaction with the environment,
> such as crawling or getting up from the floor

The three sequences that never finish here are exactly that category.

And it measures penetration and corrects for it:

> We fix this by running forward kinematics on the retargeted sequences,
> storing the minimum body height at each frame, and then offsetting the
> entire motion by the mean minimum body height

Neither was done here. The gap sits in stage 3, not in the policy.

---

## sim-to-sim — does it survive a second simulator

A policy that only works in the simulator it was trained in is not evidence of
anything. The same onnx actor was loaded into MuJoCo with no retraining and no
fine-tuning, and all seventeen sequences were replayed for the full clip length.

![sim2sim](docs/sim2sim.gif)

Six seconds of walk2_subject4. Isaac Lab on the left, the same policy in MuJoCo
on the right. Both panels start and end on the same instant.

### There is no single pass criterion

Criteria differ by lineage and they measure different things, so the same
rollouts were scored under both. `src/score_standard.py` does this.

BeyondMimic scores by termination (`tracking_env_cfg.py` 255-275). The episode
ends the moment any of the three fires.

| condition | quantity | threshold |
| --- | --- | --- |
| anchor_pos | \|ref_anchor_z − rob_anchor_z\| | 0.25 |
| anchor_ori | \|ref_gravity_z − rob_gravity_z\| | 0.8 |
| ee_body_pos | ankles and wrists, \|ref_rel_z − rob_z\| | 0.25 |

PolySim([arXiv:2510.01708](https://arxiv.org/abs/2510.01708)) counts a rollout
as failed once the mean global body position error crosses 0.5 m.

All three termination conditions look at z alone. None of them sees horizontal
drift, and PolySim's threshold is built to catch exactly that. The two criteria
are not measuring the same thing.

| criterion | passes |
| --- | --- |
| BeyondMimic termination | 14 / 17 |
| PolySim 0.5 m | 11 / 17 |

`jumps1_subject1`, `run2_subject4` and `walk2_subject3` separate the two. They
never fall over, but each leaves the reference by more than 0.5 m at some point.

One caveat. PolySim's text says mean body position error over 0.5 m, but the
released code tests whether any single body exceeds a curriculum threshold
(1.5 m by default) and has that check disabled in the default configuration.
The numbers above implement the text.

### Posture crosses over, position does not

Over all 13,064 frames of `walk1_subject1`, mean joint error is 0.064 rad in
Isaac and 0.065 rad in MuJoCo, and the two references agree frame by frame to
0.000000.

What does not cross over is position. About 72% of the global error is root
horizontal drift, and the drift is a heading error rather than a step-length
deficit: the robot walks the right distance in a slightly wrong direction and
the gap opens with time. The same drift appears in Isaac, so it is not a MuJoCo
artefact. Re-anchoring the reference to the robot's torso and removing yaw
drops the mean from 382.8 mm to 87.3 mm.

### The six that do not complete

Eleven of seventeen complete the full clip in MuJoCo. Three of the six failures
are the reference defects described above. Two leave the threshold for 0.7% and
2.5% of their frames and return. One was never trained to convergence.

A single deterministic rollout per sequence is thin, so each was rerun with ten
seeds of initial joint and root noise, giving 170 rollouts. Fourteen agree with
the single-seed result; three do not. `jumps1_subject1` reads 0% on one seed
and 30% over ten.

### Under matched perturbation

Pushes drawn from the same distribution at the same 1 to 3 s interval give
0.182 in MuJoCo and 0.202 in Isaac, correlated at 0.965. Scoring the same
MuJoCo rollouts under Isaac's own termination criterion instead gives 0.769.
Which criterion you pick moves the number about four times as much as which
simulator you run.

Matching the perturbation took care. Isaac adds the push to `root_vel_w`, a
world-frame 6D vector, while MuJoCo's free joint keeps linear velocity in world
and angular velocity body-local. Adding the same vector to both compares
nothing; the angular part has to be rotated into the body frame first.

### Definitions had to be matched too

The relative-error columns were not the same quantity at first. Isaac
re-anchors the reference to `torso_link` and removes yaw
(`commands.py` 284-294); the MuJoCo side was subtracting each pelvis and
leaving rotation alone. On the same rollout that reads 22.1 mm one way and
30.5 mm the other.

### Code

| file | what it does |
| --- | --- |
| `src/sim2sim.py` | runs the onnx actor in MuJoCo |
| `src/sim2sim_polysim.py` | the five PolySim metrics |
| `src/score_standard.py` | scores one rollout set under both criteria |
| `src/sim2sim_trials.py` | N-seed trials per sequence |
| `src/sim2sim_push.py` | matched perturbation on both sides |
| `src/horizon_trials.py` | keeps the full error time series per seed |
| `src/sim2sim_full_table.py` | builds the table |
| `scripts/make_video_pair.sh` | renders both panels on the same instant |

---

## Layout

```
src/       retargeting, metrics, rendering, table building
scripts/   batch drivers
configs/   selection and training order
docs/      figures used in this README
outputs/   motions, metrics, logs, videos (gitignored)
data ->    symlink to local dataset root (gitignored)
```

| script | what it does |
| --- | --- |
| `scripts/retarget_all.sh` | retarget all 77 LAFAN1 sequences to the G1 |
| `scripts/quality_all.sh` | measure IK target tracking error per sequence |
| `scripts/render_compare.sh` | render the human skeleton and the robot into one video |
| `scripts/npz_all.sh` | convert selected sequences to npz and upload to the registry |
| `scripts/train_chain.sh` | train the sequences one after another |
| `scripts/eval_all.sh` | measure completion rate and tracking error of trained policies |
| `scripts/render_tracking.sh` | render policy and reference side by side |
| `scripts/make_pipeline_figure.py` | generate the pipeline figure in this README |
| `src/eval_table.py` | put evaluation results next to the foot error as a table |

## Data

Motion datasets run to tens of gigabytes and are not tracked in this repo.
They live under `/home/sehoon/data`, reached through the `data` symlink.

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

Download `lafan1.zip` from the [official repo](https://github.com/ubisoft/ubisoft-laforge-animation-dataset/blob/master/lafan1/lafan1.zip)
and unzip it into `data/lafan1`. All 77 `.bvh` files sit flat in that directory.

### Reference — already-retargeted G1 motions

These ship motions that have already been retargeted to the G1. They are not
inputs, and they are not ground truth either — they are another method's output.
Comparing joint angles against them measures how far two methods diverge, not
how much error either one carries. They are kept for qualitative reference only.

| dataset | contents | local path |
| --- | --- | --- |
| lvhaidong/LAFAN1_Retargeting_Dataset | LAFAN1 retargeted to G1 | `data/lafan1_g1_ref` |
| bones-studio/seed | 142,220 Vicon motions, human and G1 side | `data/bones_seed` |

SEED is held for later. It is larger than LAFAN1 and pairs each motion with the
capture subject's measured body dimensions, but its human-side data is in a
custom format whose compatibility with the retargeting tools is unverified, and
no published baseline exists for it. It becomes useful once the pipeline runs.

AMASS is deferred to the training stage, where scale matters more than
comparability.

## Tools

| tool | role |
| --- | --- |
| [GMR](https://github.com/YanjieZe/GMR) | retargeting tool in use; MIT, takes LAFAN1 BVH and outputs G1 joint angles |
| Isaac Sim 5.1 / Isaac Lab 2.3.2 | reference motion playback, and RL training in stage 4 |

Retargeting itself needs no physics simulation — it is a kinematics problem.
The simulator enters at playback and at policy training.

## Background

The premise that retargeting quality is worth treating as its own problem comes
from *Retargeting Matters: General Motion Retargeting for Humanoid Motion
Tracking* ([arXiv:2510.02252](https://arxiv.org/abs/2510.02252)), which showed
that artifacts left in retargeted trajectories — foot sliding, self-penetration,
physically infeasible poses — measurably reduce the robustness of the tracking
policy trained on them.

### kobe — putting a number beside PolySim's table

The 17 above come from LAFAN1 and cannot be placed against PolySim's table
directly. The same motion under the same metrics is what makes them
comparable. One motion from the ASAP dataset, kobe, was converted with
`src/asap_to_csv.py` and run through the same pipeline: 206 frames, 4.1 s.

![kobe](docs/kobe.gif)

The full 4.1 s. Isaac Lab on the left, MuJoCo on the right.

| | PolySim Table III | here |
| --- | --- | --- |
| setting | IsaacSim_DR → MuJoCo | Isaac Lab → MuJoCo |
| MuJoCo success | 0.100 (10 trials) | 1.000 (10 trials) |
| E_g-mpjpe | 272.6 mm | 100.4 mm |

Scored under both criteria used here:

| | Isaac Lab | MuJoCo |
| --- | --- | --- |
| global body error | 94.9 mm | 94.5 mm |
| local pose, re-anchored | 44.5 mm | 43.8 mm |
| joint angle | 0.070 rad | 0.069 rad |
| survives termination | pass | pass |
| stays within 0.5 m | pass | pass |

Global error goes from 94.9 mm to 94.5 mm. There is essentially no transfer
loss.

It took three training runs. The first two never converged: the csv was
written in Isaac joint order rather than URDF order, which put
`error_joint_pos` at 2.44 rad, and the reference floated above the ground.

The comparison carries conditions. The trainer, the window length and the
sample size all differ from the paper, and PolySim's success test differs
between its text and its released code. What is matched is the motion and the
metric definitions.

## What is left

The order follows the dependencies: 1 and 2 change what 3 operates on.

1. Redo selection. The current 19 were picked on foot error, which turned out
   not to predict policy performance. Run `src/motion_defect_census.py` over all
   77 retargeted clips and select on ground penetration, airborne fraction, foot
   slip and joint velocity violation instead.
2. Correct penetration. Run forward kinematics over each retarget, store the
   minimum body height per frame, and offset the whole motion by it — the same
   step Retargeting Matters describes. Two of the sequences that currently never
   finish may survive this.
3. Train the remaining two, obstacles1_subject1 and obstacles4_subject2. Steps 1
   and 2 change which sequences these are.
4. Score the Isaac side under both criteria. The two-criteria table is filled in
   for MuJoCo only; `scripts/gt_dump_all.sh` dumps the per-frame global error for
   all 17 and completes the other two cells.

## Status

Stages 1 to 3 are done. Stage 4 has 17 of the 19 selected sequences trained to
30,000 iterations and evaluated over 100 rollouts. The two left are
obstacles1_subject1 and obstacles4_subject2.

The evaluation code exists now. The BeyondMimic repository ships `train.py` and
`play.py` but nothing that counts whether a policy carries a reference to the
end, so completion rate and tracking error are measured here against the GMR
paper's definitions.

sim-to-sim is finished for all 17 at full clip length, with a comparison video
for each. Results are above.

Currently running is one motion from the ASAP dataset, kobe. The 17 above come
from LAFAN1, which cannot be placed next to PolySim's table directly. The same
motion under the same metrics is what puts a number beside their Table III
(success 0.100, E_g-mpjpe 272.6 mm going from IsaacSim_DR to MuJoCo).
