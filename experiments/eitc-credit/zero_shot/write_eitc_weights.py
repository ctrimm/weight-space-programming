"""ZERO-SHOT weight construction, Rule A (EITC). Attempt 1.

Written from PROTOCOL.md's rule text only. No training, no fitting, no
oracle sampling, no evaluation points seen. The reasoning:

A 1-D continuous piecewise-linear function with breakpoints b1<b2<b3 and
segment slopes m0,m1,m2,m3 is exactly a sum of hinges:

    f(x) = f(0) + m0*relu(x) + (m1-m0)*relu(x-b1) + (m2-m1)*relu(x-b2)
           + (m3-m2)*relu(x-b3)          (for x >= 0)

Rule A slopes: phase-in +0.40, plateau 0, phase-out -0.2106, exhausted 0.
Breakpoints: 17,880 / 23,350 / 57,310. f(0) = 0.

    f(x) = 0.40*relu(x) - 0.40*relu(x-17880)
           - 0.2106*relu(x-23350) + 0.2106*relu(x-57310)

Architecture scaling: input t = x/10000, output dollars = 1000*(W2.h + b2).
relu(x - b) = 10000 * relu(t - b/10000), so each hinge coefficient in
W2 is the dollar slope * 10:
    W2 = [4.0, -4.0, -2.106, 2.106, 0, ...],  b1 = [0, -1.788, -2.335, -5.731, ...]
Hidden units 4..15 are zeroed (unused).
"""
import numpy as np
from pathlib import Path

H = 16
W1 = np.zeros((H, 1))
b1 = np.zeros(H)
W2 = np.zeros((1, H))
b2 = np.zeros(1)

hinges = [  # (breakpoint in scaled t units, slope change in dollars per dollar)
    (0.0, 0.40),
    (1.788, -0.40),
    (2.335, -0.2106),
    (5.731, 0.2106),
]
for i, (bp, dslope) in enumerate(hinges):
    W1[i, 0] = 1.0
    b1[i] = -bp
    W2[0, i] = dslope * 10.0  # 10000 input scale / 1000 output scale

out = Path(__file__).with_name("eitc_weights.npz")
np.savez(out, W1=W1, b1=b1, W2=W2, b2=b2)
print("wrote", out)
