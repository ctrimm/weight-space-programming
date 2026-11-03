"""ZERO-SHOT weight construction, Rule A (EITC). Attempt 2 (labeled;
attempt 1 stands in eitc_weights.npz and its scores are unchanged).

Diagnosis of attempt 1 (max err $0.024, confined to x >= 57,310):
attempt 1 trusted the stipulated completed-phaseout amount $57,310 as the
exact zero of the phase-out line. It is not: 0.2106 * (57,310 - 23,350)
= 7,151.976, not 7,152 -- the Rev. Proc. table rounds the completed
phaseout to the nearest $10, so the stipulated oracle is discontinuous
by $0.024 at x = 57,310 (formula value $0.024 just left of it, $0 at and
right of it). Attempt 1 placed its final hinge at 57,310 and so carried
a $0.024 plateau to the whole tail. PROTOCOL.md's "internal consistency
check" repeated the same arithmetic slip (it claims 7,152 / 0.2106 =
33,960.0 exactly; the true quotient is 33,960.114). See ERRATUM.md.

Attempt 2 changes exactly one number: the final hinge sits at the true
zero of the phase-out line, 23,350 + 7,152 / 0.2106 = 57,310.114.
Predicted residual: exact everywhere except the seam itself -- the
single evaluation point x = 57,310 keeps a $0.024 error, which is
irreducible for any continuous network that matches the formula on the
left (continuity forces f(57,310) = limit from the left = 0.024 while
the oracle says 0).
"""
import numpy as np
from pathlib import Path

H = 16
W1 = np.zeros((H, 1))
b1 = np.zeros(H)
W2 = np.zeros((1, H))
b2 = np.zeros(1)

true_zero_scaled = (23_350.0 + 7_152.0 / 0.2106) / 10_000.0  # 5.7310114...
hinges = [
    (0.0, 0.40),
    (1.788, -0.40),
    (2.335, -0.2106),
    (true_zero_scaled, 0.2106),
]
for i, (bp, dslope) in enumerate(hinges):
    W1[i, 0] = 1.0
    b1[i] = -bp
    W2[0, i] = dslope * 10.0

out = Path(__file__).with_name("eitc_weights_attempt2.npz")
np.savez(out, W1=W1, b1=b1, W2=W2, b2=b2)
print("wrote", out)
