"""ZERO-SHOT weight construction, Rule B (SNAP allotment). Attempt 1.

Written from PROTOCOL.md's rule text only. No training, no fitting, no
oracle sampling, no evaluation points seen. Work in $k throughout
(inputs e = E/1000, s = S/1000; output dollars = 1000 * final linear).

Rule, restated in $k:
    a      = 0.8e - 0.209                      (adjusted income; > 0 on domain)
    u      = s - 0.5a = s - 0.4e + 0.1045      (shelter over half of a)
    X      = relu(u)                           (excess shelter)
    Xc     = min(X, 0.744) = X - relu(u - 0.744)   (capped excess shelter)
    N      = relu(a - Xc)                      (net income, floored)
    B      = relu(0.546 - 0.3 N)               (allotment, floored)

Layer 1 computes the three raw ingredients as hinges of the inputs:
    h0 = relu(a)          = relu(0.8e - 0.209)              (= a on domain)
    h1 = relu(u)          = relu(-0.4e + s + 0.1045)        (= X)
    h2 = relu(u - 0.744)  = relu(-0.4e + s - 0.6395)
Then a - Xc = h0 - h1 + h2 exactly (linear in layer-1 outputs).

Layer 2:
    g0 = relu(h0 - h1 + h2)                       = N
    g1 = relu(0.3*(h0 - h1 + h2) - 0.546)
The one non-obvious step: g1 must equal relu(0.3*N - 0.546), with the
FLOORED N inside, but layer 2 can only see the unfloored combination
z = h0 - h1 + h2. The two are equal because the threshold is positive:
relu(0.3*relu(z) - 0.546) == relu(0.3*z - 0.546) for all z
(if z <= 0 both sides are relu of a negative number; if z > 0, relu(z)=z).

Output: B = 0.546 - 0.3*g0 + g1. Check the three regimes:
    N = 0:      B = 0.546 - 0 + relu(-0.546) = 0.546            (max allotment)
    0 < 0.3N < 0.546: B = 0.546 - 0.3N + 0                      (formula range)
    0.3N >= 0.546:    B = 0.546 - 0.3N + (0.3N - 0.546) = 0     (exhausted)
All other units are zeroed (unused).
"""
import numpy as np
from pathlib import Path

H = 24
W1 = np.zeros((H, 2))
b1 = np.zeros(H)
W2 = np.zeros((H, H))
b2 = np.zeros(H)
W3 = np.zeros((1, H))
b3 = np.zeros(1)

# Layer 1: rows are [e_coeff, s_coeff], bias
W1[0] = [0.8, 0.0]
b1[0] = -0.209  # a
W1[1] = [-0.4, 1.0]
b1[1] = 0.1045  # u
W1[2] = [-0.4, 1.0]
b1[2] = -0.6395  # u - 0.744

# Layer 2: combinations of (h0, h1, h2)
W2[0, 0:3] = [1.0, -1.0, 1.0]
b2[0] = 0.0  # N = relu(a - Xc)
W2[1, 0:3] = [0.3, -0.3, 0.3]
b2[1] = -0.546  # relu(0.3 * net_unfloored - max_allotment)

# Output: B_k = 0.546 - 0.3*g0 + 1.0*g1
W3[0, 0] = -0.3
W3[0, 1] = 1.0
b3[0] = 0.546

out = Path(__file__).with_name("snap_weights.npz")
np.savez(out, W1=W1, b1=b1, W2=W2, b2=b2, W3=W3, b3=b3)
print("wrote", out)
