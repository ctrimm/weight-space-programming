"""ZERO-SHOT weight construction — SNAP allotment, household of 2,
member-level inputs. Written from PROTOCOL.md's rule text only.
No training, no fitting, no oracle sampling.

Work in $k (inputs /1000; output dollars = 1000 * final linear).
Inputs order: [e1, u1, e2, u2, s].

Rule, restated in $k, with E = e1+e2, U = u1+u2 (combined in layer 1
by giving both members' columns the same coefficients):
    a  = 0.8E + U - 0.209                  (adjusted income; can be < 0 here)
    t  = s - 0.5a = s - 0.4E - 0.5U + 0.1045
    X  = min(relu(t), 0.744) = relu(t) - relu(t - 0.744)
    z  = a - X                             (unfloored net income)
    N  = relu(z)
    B  = relu(0.546 - 0.3 N)

Layer 1 (hinges of the inputs; a needs a hinge pair because the
member-level domain lets a go negative, unlike the totals-level
SNAP experiment where a > 0 throughout):
    h0 = relu(a)            coeffs [0.8, 1, 0.8, 1, 0],    b = -0.209
    h1 = relu(-a)           coeffs [-0.8, -1, -0.8, -1, 0], b = +0.209
    h2 = relu(t)            coeffs [-0.4, -0.5, -0.4, -0.5, 1], b = +0.1045
    h3 = relu(t - 0.744)    same coeffs,                    b = -0.6395
Then z = h0 - h1 - h2 + h3 exactly (linear in layer-1 outputs).

Layer 2:
    g0 = relu(z)                    = N
    g1 = relu(0.3 z - 0.546)
The totals-level SNAP trick applies unchanged: g1 must implement
relu(0.3 * relu(z) - 0.546) with the floored N inside, but layer 2
sees only the unfloored z; the two are equal because the threshold
is positive (z <= 0 makes both sides relu of a negative number).

Output: B = 0.546 - 0.3*g0 + g1 (regimes checked in the totals-level
construction's write-up: N = 0 -> max allotment; middle -> formula;
exhausted -> 0). All other units zeroed.
"""
import numpy as np
from pathlib import Path

H = 24
W1 = np.zeros((H, 5)); b1 = np.zeros(H)
W2 = np.zeros((H, H)); b2 = np.zeros(H)
W3 = np.zeros((1, H)); b3 = np.zeros(1)

W1[0] = [0.8, 1.0, 0.8, 1.0, 0.0];    b1[0] = -0.209   # relu(a)
W1[1] = [-0.8, -1.0, -0.8, -1.0, 0.0]; b1[1] = 0.209   # relu(-a)
W1[2] = [-0.4, -0.5, -0.4, -0.5, 1.0]; b1[2] = 0.1045  # relu(t)
W1[3] = [-0.4, -0.5, -0.4, -0.5, 1.0]; b1[3] = -0.6395 # relu(t - 0.744)

W2[0, 0:4] = [1.0, -1.0, -1.0, 1.0];  b2[0] = 0.0      # N = relu(z)
W2[1, 0:4] = [0.3, -0.3, -0.3, 0.3];  b2[1] = -0.546   # relu(0.3 z - 0.546)

W3[0, 0] = -0.3; W3[0, 1] = 1.0; b3[0] = 0.546

out = Path(__file__).with_name("snapcomp_weights.npz")
np.savez(out, W1=W1, b1=b1, W2=W2, b2=b2, W3=W3, b3=b3)
print("wrote", out)
