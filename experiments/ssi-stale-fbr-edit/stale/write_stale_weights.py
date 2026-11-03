"""ZERO-SHOT weight construction, SSI payment with STALE FBR (2024, $943).

Identical derivation to ../../ssi-payment/zero_shot/write_ssi_weights.py
with FBR = 0.943 substituted for 0.994 throughout. This is the
"broken" network the edit experiment then repairs.

Written from PROTOCOL.md's rule text only. No training, no fitting, no
oracle sampling, no evaluation points seen. Work in $k (inputs e = E/1000,
u = U/1000; output dollars = 1000 * final linear).

Rule restated:
    CU = relu(u - 0.020)              countable unearned
    R  = relu(0.020 - u)              general-exclusion spill to earned
    X  = relu(e - 0.065 - R)          earned income over exclusions
    CE = X / 2
    C  = CU + CE
    P  = relu(0.943 - C)   [STALE: 2024 FBR]

Staging (3 hidden layers):
  Layer 1 (of inputs e, u):
    h0 = relu(u - 0.020)     = CU
    h1 = relu(0.020 - u)     = R
    h2 = relu(e)             = e (pass-through; e >= 0 on domain)
  Layer 2:
    g0 = relu(h2 - h1 - 0.065) = X
    g1 = relu(h0)              = CU (pass-through)
  Layer 3:
    k0 = relu(g1 + 0.5*g0 - 0.943) = relu(C - FBR)
    k1 = relu(g1)                  = CU (pass-through)
    k2 = relu(g0)                  = X  (pass-through)
  Output: P = 0.943 - k1 - 0.5*k2 + k0
    If C < 0.943: k0 = 0 and P = 0.943 - C.  If C >= 0.943:
    P = 0.943 - C + (C - 0.943) = 0.  Exact in both regimes.
All other units zeroed.
"""
import numpy as np
from pathlib import Path

H = 24
W1 = np.zeros((H, 2)); b1 = np.zeros(H)
W2 = np.zeros((H, H)); b2 = np.zeros(H)
W3 = np.zeros((H, H)); b3 = np.zeros(H)
W4 = np.zeros((1, H)); b4 = np.zeros(1)

# Layer 1: rows are [e_coeff, u_coeff]
W1[0] = [0.0, 1.0];  b1[0] = -0.020   # CU
W1[1] = [0.0, -1.0]; b1[1] = 0.020    # R (spill)
W1[2] = [1.0, 0.0];  b1[2] = 0.0      # e pass-through

# Layer 2: combos of (h0=CU, h1=R, h2=e)
W2[0, 0:3] = [0.0, -1.0, 1.0]; b2[0] = -0.065  # X = relu(e - R - 65)
W2[1, 0:3] = [1.0, 0.0, 0.0];  b2[1] = 0.0     # CU pass-through

# Layer 3: combos of (g0=X, g1=CU)
W3[0, 0:2] = [0.5, 1.0]; b3[0] = -0.943  # relu(C - FBR)
W3[1, 0:2] = [0.0, 1.0]; b3[1] = 0.0     # CU pass-through
W3[2, 0:2] = [1.0, 0.0]; b3[2] = 0.0     # X pass-through

# Output: P_k = 0.943 - CU - 0.5*X + relu(C - FBR)
W4[0, 0:3] = [1.0, -1.0, -0.5]
b4[0] = 0.943

out = Path(__file__).with_name("ssi_stale_weights.npz")
np.savez(out, W1=W1, b1=b1, W2=W2, b2=b2, W3=W3, b3=b3, W4=W4, b4=b4)
print("wrote", out)
