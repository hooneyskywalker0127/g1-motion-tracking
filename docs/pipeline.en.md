# Pipeline

[← README](../README.en.md)

![pipeline](pipeline.png)

Stages 1 to 3 carry no physics: they compute poses and decide which ones are
worth keeping. Physics enters at stage 4, where the robot has to hold itself up.
Stage 5 merges the per-motion policies into one.

## 1. Human mocap

Motion capture records where every joint of a human body was at each instant,
as numbers rather than video.

Done: LAFAN1, 77 BVH sequences at 30 fps, five subjects, 4.6 hours.

## 2. Retargeting

Human joint data cannot be applied to the G1 directly: link lengths differ,
joint axes differ, and some human joints have no robot counterpart. Retargeting
solves for the robot joint angles that best reproduce the original motion while
respecting the robot's joint limits and keeping the feet on the ground.

![retargeting](retargeting.gif)

Orange is the source human skeleton straight out of the mocap file; grey is the
retargeted G1. The bands carry the segment lengths that make this hard and the
distance between each tracked body and the target the IK was solving for. The
robot is not walking here — each frame's joint angles are written straight into
the model and rendered. No simulation step runs.

Done: all 77 sequences retargeted with
[GMR](https://github.com/YanjieZe/GMR), 496,672 frames. Verified across every
frame: no NaNs or infinities, no joint-limit violations on any of the 29 joints,
frame counts matching the source BVH exactly.

## 3. Reference motion

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
[`configs/selected_motions.txt`](../configs/selected_motions.txt). Each was
converted to the 50 fps npz BeyondMimic reads, which adds the link velocities
the policy needs, and uploaded to a W&B registry.

The thresholds are not principled. They were picked because they leave 19
sequences, close to the 21 the GMR paper trained on. Once training shows which
sequences fail and where their error sits, the cut can be argued for.

## 4. Motion tracking policy

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

![tracking](tracking.gif)

The gif above is six seconds of it. The whole sequence runs 3 minutes 58
seconds.

## 5. Policy distillation

![stage 5](stage_5.png)

Stage 4 leaves one policy per motion. Driving 14 motions means holding 14
policies and picking one. Stage 5 merges them into a single policy.

The method is distillation. The 14 trained policies act as teachers while a
single student policy drives the simulator itself, asking the teachers at every
step what they would have done and copying the answer. Copying only the states
the teachers visited leaves the student unable to recover once it drifts, so the
student rolls out first and is labelled where it actually ends up. That is
DAgger ([Ross et al., 2011](https://arxiv.org/abs/1011.0686)).

The distillation code is [HOVER](https://github.com/NVlabs/HOVER)'s
`neural_wbc/student_policy`, used as is. The trainer, the student network, the
buffer and the loss are untouched; only the package-internal imports changed.
HOVER runs on IsaacLab and rsl-rl, the same stack as this repository.

HOVER assumes a single teacher, so one place had to change. The trainer hands
the teacher nothing but an observation tensor, which leaves the observation as
the only channel for saying which teacher to use. A motion index is appended to
the end of the teacher observation, and the teacher implementation strips it off
and routes each environment to its own teacher. Carrying the selector inside the
observation is what [parkour](https://github.com/ZiwenZhuang/parkour)'s
`ActorCriticFieldMutex` does.

The student is not given the motion index. With it, the student would memorise
the clip number and stop reading the reference. Tracking a motion it never
trained on requires judging from the reference alone.

The teacher observation is 161-dimensional (58 reference joint values, 9 anchor
error, 93 proprioception, 1 motion index); the student sees 260. The student
gets 100 terms the teacher does not — the reference-minus-current joint position
difference (29), the joint velocity difference (29), and the body position
difference in the anchor frame (42). This follows OpenTrack, whose student
receives the reference as a difference rather than raw: given only the raw
reference, the network has to learn the subtraction first.

Every teacher is [512, 256, 128]; the student is
[2048, 2048, 1024, 1024, 512]. **That capacity is what decided the result.**
It started at [1024, 512, 256] and one clip out of fourteen stayed stuck at 9%
completion. Sixteen hypotheses were falsified before the cause turned out to be
capacity: raising the training episode from 10 s to 40 s and widening the
network fourfold took the worst clip to 98.4%. This is the point the BumbleBee
paper makes when it reports that a three-layer MLP could not hold several
experts and was replaced with a transformer.

14 policies are used as teachers. Of the 17 that were trained, the three with a
0% completion rate have no finished rollout to imitate. kobe is held out as the
control for generalisation.

### Multiple clips in one environment

The stage 4 environment takes a single npz. Distillation needs the student to
experience all 14 clips, so three places changed.

`MotionLoader` now accepts a list of npz files, concatenates them along time and
keeps the clip boundaries separately. Nothing is padded. The time index stays a
flat index into the concatenated array, so the existing indexing and anchor
transforms keep working. PHC, ProtoMotions and SONIC all use the same layout.

The clip a given environment is following is not stored: it is a binary search
of the time index against the clip boundaries. Anything stored separately can
drift out of sync.

Per-clip sampling probability is capped. With failure-driven adaptive sampling
alone, one hard clip takes over the distribution and the easy ones are
forgotten. SONIC caps it for the same reason with `max_prob_per_motion`, and
notes that roughly twice the fair share is the conservative choice when
diversity matters. The same factor of two is the default here.

There is a second reason for the cap. Walking accounts for 67% of the frames
across the 14 clips (107,777 of 162,049). Sampling uniformly without a cap tilts
the merged policy toward walking.

### Result — fourteen into one, with no loss of accuracy

64 rollouts, domain randomization off, full clip length.

| | Completion | E_g-mpbpe | E_mpbpe | E_mpjpe |
|---|---|---|---|---|
| 14 experts, each on its own clip | 99.0% | 102mm | 42mm | 0.084 |
| **one unified policy** | **99.7%** | **90mm** | **41mm** | **0.082** |

Both sides: 100 rollouts, the same condition, identical evaluation code. **The
unified policy matches or beats the experts on all six metrics.** Eleven of the
fourteen complete at 100% and the other three (walk1_subject1 and walk2_subject3
at 99%, jumps1_subject1 at 98%) stay above 98%, and global position error is lower on all fourteen, by 12 mm on
average.

Three things come with it.

- **sim-to-sim** — the same onnx file, with no retraining and no gain retuning,
  completes 12 of 14 in MuJoCo. Global error grows 5-20% but the anchor-aligned
  error is actually lower in MuJoCo, meaning the posture tracks and what
  accumulates is global drift.
- **domain randomization** — 76.1% mean with a random push every 1-3 s and
  randomized friction, torso CoM, joint offsets and reset pose. The more
  dynamic the motion, the more it costs (running 43.8%, walk4 98.4%).
- **motion transition** — switching the reference to the next clip without
  resetting the robot holds for 6 of 13 boundaries. Fourteen separate experts
  structurally cannot do this: the instant you swap networks, the robot is in a
  state the incoming policy has never seen.

### What it cannot do was measured too

Every remaining LAFAN1 sequence — 63 clips — was retargeted and run through the
same policy. Nothing was retrained. The split follows SONIC. This table is over
**64 rollouts**; the teacher comparison above is over 100. Completion turns out
to be insensitive to both - the same clip measured at 32, 64, 128 and 256
environments stays within 2 points.

| | Clips | Completion | Survived |
|---|---|---|---|
| the 14 training clips | 14 | 99.89% | 100% |
| test-repetition — a motion type **in** training, a take that is not | 35 | 0.0% | 13.4% |
| test-content — a motion type **not** in training | 28 | 0.0% | 8.6% |

**Not one of the 63 runs to the end.** The ordering tracks distance from the
training distribution: walk 48.9% > aiming 22.1% > dance 15.7% > run 13.8% >
obstacles 5.1% > jumps 1.4%, and fight, ground and fallAndGetUp — lying down and
heavy contact — bottom out at 3-5%.

Fourteen clips is three orders of magnitude below what general trackers train on
(GMT 8,925, SONIC 317,189). Generalization was never on the table at this scale.
Being conditioned on the reference rather than a clip index is a **necessary
condition for generalization, not a sufficient one.**

### Under perturbation the experts win

The "no loss" above is measured in clean conditions. Putting teachers and
student under identical pushes reverses it.

| Push magnitude | 14 experts | unified policy |
|---|---|---|
| same as training | 91.0% | 84.3% |
| twice that | 31.7% | 23.3% |

E_g-mpbpe is still lower for the unified policy on 11 of 14 clips. Tracking accuracy
holds; what degrades is **recovery after being pushed**. Distillation learning
the teachers' mean behaviour, with recovery from perturbed states rare in the
data, is a plausible explanation but was not verified. Adding perturbation to
the DAgger rollouts may change it.

## ▶ Full clip (YouTube)

[![Play the full clip](youtube_thumb.jpg)](https://youtu.be/l1M4y_Nl7oc)

Click the image above to play it on YouTube — https://youtu.be/l1M4y_Nl7oc

Left is the trained policy stepping through physics, right is the reference it
was asked to follow. Both panels are Isaac Sim under the same lighting and
camera, start from the same motion frame, and run the full 11,909 frame
sequence. They never match pixel for pixel: the initial pose is randomised at
every reset, and the left robot has to hold itself up while the right one is
posed frame by frame.

## ▶ Training progression (YouTube)

[![Play the training progression](youtube_thumb_progression.jpg)](https://youtu.be/qQw8PtmXV9s)

Click the image above to play it on YouTube — https://youtu.be/qQw8PtmXV9s

Five checkpoints of the same run and the reference, played at once on the same
motion, the same start frame and the same camera. Mean reward runs 14.67 at
1,000 iterations, 30.80 at 5,000, 33.49 at 10,000, 36.63 at 20,000 and 36.80 at
30,000, so most of the gain lands early and the later checkpoints separate on
how long they hold rather than on the number.

## ▶ Domain randomization (YouTube)

[![Play the domain randomization clip](youtube_thumb_randomization.jpg)](https://youtu.be/d61rKk675qY)

Click the image above to play it on YouTube — https://youtu.be/d61rKk675qY

![randomization](randomization.gif)

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

## Evaluation criteria

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
| E_mpbve | mean body velocity error (mm/frame) |
| E_mpbae | mean body acceleration error (mm/frame²) |

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

![walk3_subject1 reset](reset_walk3_subject1.gif)

walk3_subject1 at 195 s, where the anchor height crosses its threshold.

![walk3_subject4 reset](reset_walk3_subject4.gif)

walk3_subject4 at 218 s, where an ankle or wrist height crosses its threshold.

A 0% completion rate does not mean the policy never follows the reference. It
follows for over three minutes and then catches on one moment. That is what the
78% and 88% mean tracked lengths are describing.

## The three that never finish are a reference problem, not a training one

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

## The work that reports 96-100% on the same material does two more things

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
