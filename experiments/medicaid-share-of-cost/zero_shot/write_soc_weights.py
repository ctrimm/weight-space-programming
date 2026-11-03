"""ZERO-SHOT weight construction, medically-needy share of cost.
From PROTOCOL.md's rule text only. Work in $k.

Rule restated:
  CU = relu(u - 0.020); R = relu(0.020 - u)
  X  = relu(e - 0.065 - R); CE = X/2; C = CU + CE
  SOC = relu(C - 0.600 - m)

Staging:
  L1: h0 = relu(u - 0.020) = CU; h1 = relu(0.020 - u) = R;
      h2 = relu(e); h3 = relu(m)
  L2: g0 = relu(h2 - h1 - 0.065) = X; g1 = relu(h0) = CU;
      g2 = relu(h3) = m
  L3: k0 = relu(g1 + 0.5*g0 - g2 - 0.600) = relu(C - m - 0.6) = SOC_k
  Output: SOC = k0 (weight 1, bias 0).
All other units zeroed.
"""
import numpy as np
from pathlib import Path

H = 24
W1 = np.zeros((H, 3)); b1 = np.zeros(H)
W2 = np.zeros((H, H)); b2 = np.zeros(H)
W3 = np.zeros((H, H)); b3 = np.zeros(H)
W4 = np.zeros((1, H)); b4 = np.zeros(1)

W1[0] = [0.0, 1.0, 0.0];  b1[0] = -0.020   # CU
W1[1] = [0.0, -1.0, 0.0]; b1[1] = 0.020    # R (spill)
W1[2] = [1.0, 0.0, 0.0];  b1[2] = 0.0      # e pass-through
W1[3] = [0.0, 0.0, 1.0];  b1[3] = 0.0      # m pass-through

W2[0, 0:4] = [0.0, -1.0, 1.0, 0.0]; b2[0] = -0.065  # X = relu(e - R - 65)
W2[1, 0:4] = [1.0, 0.0, 0.0, 0.0];  b2[1] = 0.0     # CU pass-through
W2[2, 0:4] = [0.0, 0.0, 0.0, 1.0];  b2[2] = 0.0     # m pass-through

# Layer-2 outputs: g0 = X (idx 0), g1 = CU (idx 1), g2 = m (idx 2).
# k0 = relu(0.5*X + CU - m - 0.600) = relu(C - M - MNIL).
W3[0, 0:3] = [0.5, 1.0, -1.0]; b3[0] = -0.600

W4[0, 0] = 1.0; b4[0] = 0.0

out = Path(__file__).with_name("soc_weights.npz")
np.savez(out, W1=W1, b1=b1, W2=W2, b2=b2, W3=W3, b3=b3, W4=W4, b4=b4)
print("wrote", out)
