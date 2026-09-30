# sim-to-sim — does it survive a second simulator

[← README](../README.en.md)

A policy that only works in the simulator it was trained in is not evidence of
anything. The same onnx actor was loaded into MuJoCo with no retraining and no
fine-tuning, and all seventeen sequences were replayed for the full clip length.

![sim2sim](sim2sim.gif)

Six seconds of walk2_subject4. Isaac Lab on the left, the same policy in MuJoCo
on the right. Both panels start and end on the same instant.

## There is no single pass criterion

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

One caveat. PolySim's text says mean body position error over 0.5 m, but the
released code tests whether any single body exceeds a curriculum threshold
(1.5 m by default) and has that check disabled in the default configuration.
The numbers below implement the text.

## Is anything lost in transfer

The same policy was run in Isaac, where it was trained, and in MuJoCo, which it
had never seen. The numbers do not drop.

| metric | Isaac | MuJoCo | over |
| --- | --- | --- | --- |
| Success rate | 0.765 | 0.775 | 17 sequences |
| Success rate (excluding the three at zero) | 0.929 | 0.941 | 14 sequences |
| E_g-mpbpe | 108.3 mm | 101.3 mm | 14 sequences |
| E_mpjpe | 0.594 rad | 0.593 rad | 14 sequences |

100 runs each. Error is averaged over completing runs only, so the three that
complete none (`obstacles2_subject1`, `walk3_subject1`, `walk3_subject4`) drop
out of the last three rows.

![sim2sim dance](sim2sim_dance.gif)

The most dynamic five seconds of `dance2_subject3`. Isaac Lab on the left, the
same policy dropped into MuJoCo on the right. This sequence is where the two
simulators agree most closely — 0.99 against 1.00 completion, 104.4 against
104.3 mm global error.

The rest of this section is how those numbers were produced.

Putting two simulators side by side requires the two columns to be the same
quantity. Four things were matched.

| item | what was matched |
| --- | --- |
| scorer | `src/score_standard.py` alone; `scripts/beyondmimic/eval_sym.py` uses the same expressions on the Isaac side |
| perturbation | the `MotionCommandCfg` values from `tracking_env_cfg.py` on both sides: root position, orientation, velocity and joints, uniform |
| alignment | reference and robot are read at the same instant |
| termination | both sides roll to the end without early termination; the two criteria are computed afterwards |

The last row is the one that matters. With Isaac's termination enabled, an
episode ends the moment the robot falls, so global error after the fall is never
recorded and a failed rollout scores as a PolySim success. MuJoCo has no
termination, rolls to the end, and a fallen robot always crosses 0.5 m. The two
numbers would carry the same name while measuring different things.

Three conditions: sim is Isaac without perturbation, sim-dr is Isaac with it over
100 environments, sim2sim is MuJoCo with the same perturbation over 100 trials.
The window is full clip length.

| | sim | sim-dr | sim2sim | retention |
| --- | --- | --- | --- | --- |
| BeyondMimic success | 0.779 | 0.765 | 0.775 | 101.4 % |
| PolySim success | 0.668 | 0.633 | 0.609 | 96.3 % |
| global body error | 105.6 mm | 108.3 mm | 101.3 mm | 106.9 % |
| local pose, re-anchored | 40.5 mm | 40.6 mm | 38.0 mm | 106.7 % |
| joint angle | 0.594 rad | 0.594 rad | 0.593 rad | 100.2 % |

Retention is MuJoCo/Isaac for success rates and Isaac/MuJoCo for errors, so 100 %
means nothing was lost either way. Nothing is lost.

Above 100 % does not mean MuJoCo is the better engine. Contact handling and the
solver differ. The sentence this supports is that there is no transfer loss, and
no more than that. The comparable published figure is PHUMA appendix D.3, which
reports 90.5 % and 93.2 % retention going from Isaac Gym to MuJoCo.

Per sequence, ordered by completion rate to match the training results table
above.

| column | meaning |
| --- | --- |
| S_bm | share of rollouts that never trip any of the three BeyondMimic termination conditions, at the thresholds in the table above |
| S_poly | share of rollouts whose mean global body error never crosses 0.5 m, judged independently of S_bm |
| global | mean body position error in world coordinates (mm); grows with root drift |
| local | the same error after re-anchoring the reference to the robot anchor (mm), which removes root drift and leaves posture |

Isaac columns are sim-dr (100 environments), MuJoCo columns are sim2sim (100
trials). Errors are averaged over completing trials only, so the three with none
are undefined.

| sequence | S_bm Isaac | S_bm MuJoCo | S_poly Isaac | S_poly MuJoCo | global Isaac | global MuJoCo | local Isaac | local MuJoCo |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `walk4_subject1` | 1.00 | 1.00 | 0.99 | 0.96 | 61.1 | 55.0 | 36.8 | 33.9 |
| `walk3_subject2` | 1.00 | 1.00 | 1.00 | 0.99 | 83.2 | 70.3 | 34.5 | 31.1 |
| `walk1_subject2` | 1.00 | 1.00 | 0.99 | 1.00 | 79.8 | 73.7 | 34.6 | 31.1 |
| `walk3_subject5` | 1.00 | 0.99 | 0.88 | 0.88 | 86.5 | 88.8 | 35.7 | 37.0 |
| `aiming1_subject1` | 0.99 | 1.00 | 0.93 | 0.86 | 89.7 | 74.2 | 35.6 | 32.4 |
| `walk1_subject5` | 1.00 | 0.94 | 0.99 | 0.86 | 91.2 | 86.2 | 33.9 | 31.5 |
| `dance2_subject3` | 0.99 | 1.00 | 0.99 | 1.00 | 104.4 | 104.3 | 45.4 | 42.3 |
| `walk2_subject1` | 1.00 | 1.00 | 0.91 | 0.95 | 115.2 | 101.7 | 43.2 | 41.0 |
| `walk1_subject1` | 1.00 | 1.00 | 0.96 | 0.99 | 77.7 | 69.3 | 33.9 | 30.6 |
| `walk2_subject4` | 0.99 | 0.99 | 1.00 | 0.98 | 91.7 | 77.6 | 40.3 | 36.7 |
| `run2_subject4` | 0.83 | 0.73 | 0.00 | 0.00 | 181.1 | 174.9 | 47.0 | 44.5 |
| `jumps1_subject1` | 0.98 | 0.97 | 0.05 | 0.14 | 155.3 | 143.3 | 42.6 | 40.6 |
| `obstacles3_subject3` | 0.58 | 0.80 | 0.62 | 0.75 | 165.6 | 160.1 | 54.7 | 51.1 |
| `walk2_subject3` | 0.64 | 0.76 | 0.45 | 0.00 | 134.0 | 138.8 | 50.2 | 48.8 |
| `walk3_subject4` | 0.00 | 0.00 | 0.00 | 0.00 | — | — | — | — |
| `obstacles2_subject1` | 0.00 | 0.00 | 0.00 | 0.00 | — | — | — | — |
| `walk3_subject1` | 0.00 | 0.00 | 0.00 | 0.00 | — | — | — | — |

The seventeen fall into four groups by motion type.

The ten walking, aiming and dance clips (`walk4_subject1` through
`walk2_subject4`) hold S_bm at 0.94 or better on both sides and S_poly at 0.86 or
better, with 61-115 mm global and 31-45 mm local error. All four metrics sit
close together across the two simulators.

Running and jumping (`run2_subject4`, `jumps1_subject1`) hold S_bm at 0.73-0.98
while S_poly drops to 0.00-0.14. They do not fail by falling, they fail by
drifting. Their global error of 155-181 mm is also the largest of the seventeen.
The faster the motion, the more heading error accumulates over a long clip. One
criterion alone hides this entirely.

`obstacles3_subject3` and `walk2_subject3` go the other way on S_bm: 0.58 and
0.64 in Isaac against 0.80 and 0.76 in MuJoCo. Something in the Isaac run is
harsher, and it is not transfer getting worse. `walk2_subject3` is the exception
worth naming — its S_poly falls from 0.45 to 0.00, the one cell where MuJoCo is
clearly worse.

The three with no completed rollout have no error to report.
`obstacles2_subject1` survives 8.6 % of its clip and is unconverged;
`walk3_subject1` and `walk3_subject4` reach 77-87 % and fail near the end, where
the LAFAN1 actor sits or lies down and the selection criterion never looked.
Retargeting Matters reports 96-100 % on the same LAFAN1, G1 and BeyondMimic, so
these three are a training and motion-selection problem, not a transfer problem.

Three things run through all of it. MuJoCo is worse than Isaac in only three
cells out of the fourteen that report error — global on `walk3_subject5` and
`walk2_subject3`, local on `walk3_subject5` — and equal or better everywhere
else. Error magnitude is set by motion difficulty rather than by simulator:
61-115 mm for walking, 104-166 mm for obstacles and dance, 155-181 mm for
jumping and running. And failure splits in two: a low S_bm means the robot fell,
which is a training and selection problem, while a low S_poly alone means
heading error accumulated over a long clip.

The two scorers were checked against each other first. Isaac's env 0 rollout was
dumped in full, rescored with the MuJoCo scorer, and compared against the online
values: all five metrics agree to four decimal places, the residual coming from
the dump being float16. No column was placed in a shared table before that check
passed.

## Posture crosses over, position does not

Joint angle error is 0.594 rad in Isaac and 0.593 rad in MuJoCo. Posture crosses
over with essentially no loss.

What does not cross over is position. Re-anchoring MuJoCo's 101.3 mm global body
error to the robot anchor drops it to 38.0 mm, so 62 % of the error is root
position and heading drift rather than posture. The same drift appears in Isaac
(108.3 → 40.6 mm), so it is not a MuJoCo artefact.

## The three that do not complete

Fourteen of seventeen complete the full clip. The remaining three are the zeros
in table B: `obstacles2_subject1`, `walk3_subject1` and `walk3_subject4`. They
differ in kind. The first survives 8.6 % of its clip and is unconverged; the
other two reach 77-87 % and fail near the end, where the LAFAN1 actor sits or
lies down and the selection criterion never looked for that.

## Code

| file | what it does |
| --- | --- |
| `src/sim2sim.py` | runs the onnx actor in MuJoCo |
| `src/sim2sim_polysim.py` | the five PolySim metrics |
| `src/score_standard.py` | scores one rollout set under both criteria |
| `src/sim2sim_trials.py` | N trials per sequence under Isaac's perturbation spec |
| `src/sym_table.py` | collects the three conditions into tables A, B and C |
| `src/sym_table_png.py` | renders the same tables as an image |
| `scripts/sym_all.sh` | re-measures all 17 under the three conditions |
| `scripts/make_video_pair.sh` | renders both panels on the same instant |
