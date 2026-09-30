# Background and kobe

[← README](../README.en.md)

## Background

The premise that retargeting quality is worth treating as its own problem comes
from *Retargeting Matters: General Motion Retargeting for Humanoid Motion
Tracking* ([arXiv:2510.02252](https://arxiv.org/abs/2510.02252)), which showed
that artifacts left in retargeted trajectories — foot sliding, self-penetration,
physically infeasible poses — measurably reduce the robustness of the tracking
policy trained on them.

### kobe — does it hold outside LAFAN1

All 17 above are LAFAN1 walking and dance. There is no high-difficulty one-shot
motion among them, so one motion from the ASAP dataset, kobe, was converted with
`src/asap_to_csv.py` and run through the same pipeline: 206 frames, 4.1 s.

![kobe](kobe.gif)

The full 4.1 s. Isaac Lab on the left, MuJoCo on the right. Over 100 MuJoCo
trials: BeyondMimic success 1.000, PolySim success 0.990, global body error
128.0 mm. No transfer loss here either.

Stated plainly to avoid a misreading: this clip runs on a policy trained for
kobe alone. The policies trained on the seventeen did not track it. This
pipeline is one policy per motion, so kobe got its own 30,000 iterations. What
this shows is transfer, not generalisation.

So the clip supports exactly one claim: that transfer working is not a property
of LAFAN1. A four-second, fast motion from a different dataset behaves the same
way. One clip is not a sample, and nothing beyond that is claimed.

Where this clip becomes worth more is later. Once several motions are merged
into a single policy, kobe can be held out of training and used to measure
generalisation — a different dataset, a motion type absent from training, and a
short fast segment all put it clearly outside the training distribution.

It took three training runs. The first two never converged: the csv was
written in Isaac joint order rather than URDF order, which put
`error_joint_pos` at 2.44 rad, and the reference floated above the ground.

This clip is not placed against PolySim's table, for three reasons. First, the
0.100 in their Table III for `IsaacSimDR → MuJoCo` is not PolySim's own result;
it is the single-simulator DR baseline the table exists to argue against.
PolySim's own row is the last one, `IsaacSim+IsaacGym+Genesis`, at 1.000.
Second, the paper names neither the 14 motions nor the 5 motions it evaluates on
— Kobe is the only motion named anywhere in the text — so the same set cannot be
assembled. Third, the trainer differs: PolySim uses HumanoidVerse with ASAP
rewards and teacher-student, this uses BeyondMimic. Retargeting Matters reports
sim2sim success mostly at 100 % on the same LAFAN1, G1 and BeyondMimic, so the
1.000 here is the ordinary value for this lineage, not a win over PolySim.
