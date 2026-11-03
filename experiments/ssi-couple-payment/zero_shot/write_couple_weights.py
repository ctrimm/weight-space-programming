"""ZERO-SHOT weight construction, SSI couple payment. From
PROTOCOL.md's rule text only. Work in $k. Inputs (e1, u1, e2, u2).

  U = u1 + u2; E = e1 + e2
  CU = relu(U - 0.020); R = relu(0.020 - U)
  X  = relu(E - 0.065 - R); CE = X/2; C = CU + CE
  P  = relu(1.491 - C)

Staging (same shape as the individual construction, with a
combining first layer):
  L1: h0 = relu(u1 + u2 - 0.020) = CU; h1 = relu(0.020 - u1 - u2) = R;
      h2 = relu(e1 + e2) = E
  L2: g0 = relu(h2 - h1 - 0.065) = X; g1 = relu(h0) = CU
  L3: k0 = relu(g1 + 0.5*g0 - 1.491) = relu(C - FBR);
      k1 = relu(g1) = CU; k2 = relu(g0) = X
  Output: P = 1.491 - k1 - 0.5*k2 + k0.
All other units zeroed.
"""
import numpy as np
from pathlib import Path

H = 24
W1 = np.zeros((H, 4)); b1 = np.zeros(H)
W2 = np.zeros((H, H)); b2 = np.zeros(H)
W3 = np.zeros((H, H)); b3 = np.zeros(H)
W4 = np.zeros((1, H)); b4 = np.zeros(1)

# Layer 1 rows are [e1, u1, e2, u2]
W1[0] = [0.0, 1.0, 0.0, 1.0];  b1[0] = -0.020   # CU from combined U
W1[1] = [0.0, -1.0, 0.0, -1.0]; b1[1] = 0.020    # R (spill)
W1[2] = [1.0, 0.0, 1.0, 0.0];  b1[2] = 0.0       # combined E pass-through

W2[0, 0:3] = [0.0, -1.0, 1.0]; b2[0] = -0.065    # X = relu(E - R - 65)
W2[1, 0:3] = [1.0, 0.0, 0.0];  b2[1] = 0.0       # CU pass-through

W3[0, 0:2] = [0.5, 1.0]; b3[0] = -1.491          # relu(C - FBR_couple)
W3[1, 0:2] = [0.0, 1.0]; b3[1] = 0.0             # CU pass-through
W3[2, 0:2] = [1.0, 0.0]; b3[2] = 0.0             # X pass-through

W4[0, 0:3] = [1.0, -1.0, -0.5]
b4[0] = 1.491

out = Path(__file__).with_name("couple_weights.npz")
np.savez(out, W1=W1, b1=b1, W2=W2, b2=b2, W3=W3, b3=b3, W4=W4, b4=b4)
print("wrote", out)
