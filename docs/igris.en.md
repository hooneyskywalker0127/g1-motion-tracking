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
