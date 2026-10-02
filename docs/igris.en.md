# A second robot — IGRIS-C

[← README](../README.en.md)

To check that the pipeline is not tied to the G1, a second humanoid is being added:
[IGRIS-C](https://github.com/robrosinc/igris_c_description_public) (1.5 m, 58 kg,
31 DoF). Retargeting is done so far.

![retarget_igris](retarget_igris.gif)

The same LAFAN1 clip retargeted onto the G1 (left) and IGRIS-C (right), in one scene with
the human skeleton in the middle. GMR also scales the root trajectory, so the G1 walks
0.88 times the human's path. To keep them side by side, each robot's hip is moved onto
the human's horizontal hip position every frame; poses are untouched.

Adding a robot to GMR takes four things: the robot model, which human bone drives which
link, a length ratio per body part, and a rotation offset per link. The IGRIS-C table was
derived from the G1 one, and two things were wrong.

- **Forearms.** At zero pose the G1 forearm points forward and the IGRIS-C forearm points
  down. With the G1 offsets the elbows sat on their joint limit in 80-88% of frames.
  Turning the elbow and wrist offsets by 90° fixed it.
- **Leg ratio.** The IGRIS-C sole is 7.1 cm below the ankle joint, twice the G1's
  3.5 cm. Scaling the legs by ankle height sank the feet into the floor in all 14 clips.
  Scaling by sole height cut the clips with more than 3 cm of penetration from 14 to 4.
  The remaining four have crouching or crawling.

![igris_fix_arm](igris_fix_arm.png)

![igris_fix_leg](igris_fix_leg.png)

Each fix undone on its own and the same clip retargeted again (first 15 s of each clip).
With the G1 forearm offsets, the elbows in `aiming1_subject1` sit on the straight-arm limit
in every frame and the aiming pose never appears. Measuring the legs to the ankle puts the
soles of `walk1_subject1` below the floor in every frame, 1.8 cm on average and 3.9 cm at
worst. A physics simulator cannot reproduce a foot below the floor, so however well the policy
learns, that much stays as tracking error.

The model has no license file, so no IGRIS-C mesh or XML is in this repository. The
scripts in `scripts/igris/` read a local clone and write locally.

| File | What it does |
| --- | --- |
| `scripts/igris/prepare_mjcf.py` | MJCF for retargeting; drops 29 backlash joints and 22 finger joints |
| `scripts/igris/prepare_urdf.py` | URDF for training; fills the placeholder torque limits from the MJCF actuators |
| `scripts/igris/make_ik_config.py` | derives the IGRIS-C IK table from the G1 one |
| `scripts/igris/compare_retarget.py` | compares both robots' retargets on foot penetration, joint limits and velocity spikes |
| `scripts/igris/fit_slot_map.py` | maps IGRIS-C joints onto the G1 policy's 29 slots (below) |
| `src/render_retarget_two.py` | renders the clip above |

**The G1 policy does not run on IGRIS-C as is.** Its inputs and outputs are laid out
for the G1's 29 joints and it learned the G1's mass and motors. Prior work,
[Any2Any](https://arxiv.org/abs/2605.23733), maps the joints onto the original
policy's slots and fine-tunes part of it on the new robot, beating training from
scratch at a fraction of the compute. The same comparison is set up here. Mapping by
joint name is wrong: the waist axes have opposite signs, the elbows zero 90° apart, and
because the forearms point along different axes the wrist roll and yaw swap. Signs and
offsets were fitted on the same 14 clips retargeted onto both robots; every one of the
29 pairs correlates at |r| ≥ 0.72.

## Training from scratch vs. transferring the G1 policy

![igris_transfer](igris_transfer.png)

The same clip was learned on IGRIS-C two ways. The clip is the most dynamic of the fourteen,
`run2_subject4` (running, 2.0 m/s root speed, 245 s).

- **A (from scratch):** BeyondMimic PPO on IGRIS-C from a random policy.
- **B (transfer):** the G1 expert for the same clip (PPO), moved the
  [Any2Any](https://arxiv.org/abs/2605.23733) way. Joints are mapped onto the G1 slots (no training), the
  expert's weights are frozen, and LoRA on every actor layer and the critic hidden layers is the only thing
  trained (rank 9, 5.0% of parameters, paper 5.26%). PPO settings are the expert's own.

Both 30,000 iterations. Evaluation: 100 rollouts, domain randomization off, final checkpoint.

| Start frame | A completion | B completion | A posture error E_mpbpe | B posture error E_mpbpe |
|---|---|---|---|---|
| 0 | 21% | 0% | 59 mm | — |
| 1000 | 96% | 16% | 59 mm | 69 mm |
| 4000 | 98% | 35% | 59 mm | 71 mm |
| 8000 | 98% | 57% | 60 mm | 71 mm |

**On this setup, training from scratch wins.** B learns far faster at first: training episode length at
iteration 100 is 12.5 for A and 172 for B, and B passes in 100 iterations what A reaches at 1,000. Then it
stops lower. A frozen 35 kg G1 policy with a 5% low-rank correction does not seem to cover a 58 kg body with
different leg length and motors; I have not verified this. The paper transferred a large policy trained on
all of AMASS (SONIC); here the source is a single-clip expert. The starting policies differ in scale.

A also manages only 21% from frame 0. Most rollouts fall in the first 20 seconds, going from standing into
a run; skip 1000 frames and it is 96-98%. Global position error is about 1.7 m for both, accumulated drift
over 245 s of running.

The comparison is per iteration. The ~9 hours spent training the G1 expert that B starts from are not counted.

### Getting IGRIS-C to learn at all

The first two attempts did not learn: episode length 5 after 8,500 iterations (G1: 397 at the same point).

1. **PD gains.** The G1 rule (kp = armature·ω²) gave IGRIS a hip kp of 790 because of its large armature,
   shrinking one action unit to 0.047 rad (G1: 0.55). Switching to G1's kp/torque ratio alone did not help.
2. **Self-collision (the real cause).** Logging physics right after reset showed 31,000 N on the pelvis. The
   vendor meshes overlap by design at pelvis/thigh, torso/upper arm and hand/forearm, and PhysX was resolving
   collisions inside the body. With self-collision off, as in Isaac Lab's own G1 and H1 configs, the force
   went to 0 N and learning followed a G1-like curve.

Checkpoints and evaluations: [Hugging Face `igris_c/`](https://huggingface.co/hooneyskywalker/humanoid-motion-tracking-policies/tree/main/igris_c) ·
training curves: [W&B `igris_c_transfer`](https://wandb.ai/hooneyskywalker-humanoid/humanoid-motion-tracking)
