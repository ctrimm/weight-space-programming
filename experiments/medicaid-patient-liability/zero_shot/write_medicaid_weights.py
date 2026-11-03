"""ZERO-SHOT weight construction, Medicaid patient liability. Attempt 1.

Written from PROTOCOL.md's rule text only. No training, no fitting, no
oracle sampling, no evaluation points seen. Work in $k (inputs i, si, p
in $k; output dollars = 1000 * final linear).

Rule restated:
    A = relu(2.64375 - si)          community spouse income allowance
    L = relu(i - 0.050 - A - p)     patient liability

Staging (2 hidden layers):
  Layer 1 (of inputs i, si, p):
    h0 = relu(2.64375 - si)   = A
    h1 = relu(i)              = i (pass-through; i >= 0 on domain)
    h2 = relu(p)              = p (pass-through)
  Layer 2:
    g0 = relu(h1 - h0 - h2 - 0.050) = L
  Output: L_k = g0 exactly.
All other units zeroed.
"""
import numpy as np
from pathlib import Path

H = 24
W1 = np.zeros((H, 3)); b1 = np.zeros(H)
W2 = np.zeros((H, H)); b2 = np.zeros(H)
W3 = np.zeros((1, H)); b3 = np.zeros(1)

# Layer 1: rows are [i_coeff, si_coeff, p_coeff]
W1[0] = [0.0, -1.0, 0.0]; b1[0] = 2.64375  # A
W1[1] = [1.0, 0.0, 0.0];  b1[1] = 0.0      # i pass-through
W1[2] = [0.0, 0.0, 1.0];  b1[2] = 0.0      # p pass-through

# Layer 2: L = relu(i - A - p - 0.050)
W2[0, 0:3] = [-1.0, 1.0, -1.0]; b2[0] = -0.050

# Output
W3[0, 0] = 1.0
b3[0] = 0.0

out = Path(__file__).with_name("medicaid_weights.npz")
np.savez(out, W1=W1, b1=b1, W2=W2, b2=b2, W3=W3, b3=b3)
print("wrote", out)
