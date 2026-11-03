# Erratum to PROTOCOL.md (2026-11-02, after scoring attempt 1)

PROTOCOL.md is frozen by hash in FREEZE.json, so this correction lives
here rather than as a silent edit.

PROTOCOL.md's Rule A "internal consistency check" is wrong. It claims
7,152 / 0.2106 = 33,960.0 and 23,350 + 33,960 = 57,310 "exactly."
In fact 0.2106 × 33,960 = 7,151.976, and 7,152 / 0.2106 = 33,960.114.
The Rev. Proc. 2024-40 completed phaseout amount ($57,310) is the
statutory phase-out line's zero rounded to the nearest $10; the line
itself reaches zero at $57,310.11.

Consequence: the stipulated Rule A oracle is discontinuous by $0.024
at x = $57,310 — the phase-out formula yields $0.024 just left of the
endpoint and the oracle's final branch yields $0 at and right of it.
No continuous network can match both sides at that point. This — not a
network or construction failure — is the whole of attempt 1's EITC
error, and the irreducible $0.024 single-point residual in attempt 2.

The same pattern (published table amounts rounded independently of the
statutory rates they are derived from) is worth remembering for
CivBench: constants that look mutually consistent in a table may not
multiply out, and a scorer that treats them as exact will manufacture
"model errors" of exactly this size at exactly the seams.
