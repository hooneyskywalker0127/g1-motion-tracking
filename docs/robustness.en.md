# Robustness — model mismatch and real-time inference

[← README](../README.en.md)

## Model mismatch — which discrepancy breaks it

There is no hardware, so sim-to-real cannot be measured. Instead, each axis on which a
simulator can disagree with a real robot was perturbed **on its own**, to see where the
consolidated policy fails. 14 clips, 64 rollouts per clip, domain randomization off,
full clip length.

| Axis | Completion |
|---|---|
| Baseline | 99.9% |
| **Latency** 1 step (20 ms) / 2 steps | **29.6% / 0%** |
| Torque limit ×0.85 / ×0.7 / ×0.5 | 92.6% / 56.4% / 0% |
| Mass ×0.9 / ×1.1 / ×1.2 | 99.8% / 91.7% / 25.0% |
| Observation noise ×1 / ×2 / ×3 | 98.4% / 78.1% / 32.5% |
| Friction ×0.5 / ×0.7 / ×1.5 | 70.3% / 99.9% / 99.2% |
| PD gain ×0.9 / ×1.1 | 97.8% / 100% |

**Latency is the only cliff.** Every other axis degrades gradually; one step of latency
removes 70 points. Latency was never in the domain randomization. Torque, gain, mass
and friction were varied during training, but the policy never saw a delayed command.
Fixing it means mixing latency into training or putting recent command history into
the observation.

The rest are one-sided. Only heavier, more slippery or weaker hurts. Higher gains are,
if anything, safer (100% at ×1.05 and ×1.1).

Errors are averaged over completed rollouts only, so **read them together with
completion**. At torque ×0.7 the E_g-mpbpe is lower than baseline because half the
rollouts dropped out and only the easy stretches remain.

## Real-time inference loop — does it fit the 20 ms budget

Since one step of latency is fatal, whether inference fits in the control period (50 Hz,
20 ms) is a deployment condition. The MuJoCo control loop was ported to C++ (`cpp/`).
A Python loop fed the same inputs is the control; after 1000 control steps both give
identical root position, orientation and joint angles.

| (ms, 8195 steps) | p50 | p99 | p99.9 | max |
|---|---|---|---|---|
| C++ inference | 0.42 | 0.99 | 1.15 | 1.51 |
| C++ full control cycle | 0.92 | 1.52 | 1.72 | 2.29 |
| Python full control cycle | 1.10 | 1.74 | 1.95 | 2.42 |

There is close to a tenfold margin. Before starting I expected Python's tail to be
heavier because of garbage collection. That was wrong. Python's cost is **a constant
0.2 ms per cycle**, not a heavier tail; the network is small and the GC has nothing to do.

The worst case came from the OS. In one run **both** implementations went over 20 ms,
with the spikes clustered in one stretch. So this measurement supports "inference costs
far less than the budget" and nothing more. It does not show that C++ guarantees real
time; that would take real-time scheduling and CPU isolation, which were not done.
