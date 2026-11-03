# LinkedIn post — draft for Cory (2026-11-03)

Graphics: `graphic1_stat_card.png` (lead image), `graphic2_bakeoff.png`
(second image in the carousel). Post text below the line.

---

TLDR: A post this week from MIT — "Sparks of Weight-Space Programming" (Tomita et al.) — reports that a frontier model can write a neural network's weights directly, the way you'd write code. I tested the claim in the one domain where "correct" is defined by law: US benefit formulas. The claim held up better than I expected. The trained networks did not.

The setup: take a benefit formula — SNAP allotments, the EITC, SSI payments, Medicaid cost-sharing — stipulate it exactly from the regulation, and write a small network's weights by construction. No training data. No gradient descent. Then score it against the formula.

Every construction was exact. Worst error across seven rules and tens of thousands of test points: about $0.0000000000005, which is floating-point noise, not error.

Then I had two outside models try it cold. GPT-5.6 Sol and Gemini 3.1 Pro, given nothing but the formula text, wrote exact networks too. In two of the programs, all three of us produced the same circuit, up to relabeling of the hidden units.

The trained networks are the uncomfortable part. Trained on the same formulas, they reach average errors under a dollar — and they are structurally wrong. For a married couple on SSI, the law counts the couple's combined income once. The trained model's answer depends on which spouse earns the income in 98.9% of equivalent households, moving as much as $25.15. Its average error was $0.32. Accuracy hid it completely. A test of the properties the law implies — same combined income, same answer — found it immediately.

I tried to train the defect away. Augmentation shrank it fivefold. A symmetry penalty did worse. An architecture that sums the incomes first eliminated it exactly, because the violation became unrepresentable. Compilation eliminates it and the approximation error with it.

Why this matters beyond one repo: we evaluate benefit systems and caseworker AI mostly on accuracy. Accuracy will certify a model that treats legally identical households differently. The fix is boring and available: test structural properties alongside accuracy, and put guarantees in the architecture, not the loss function.

Scope, stated plainly: these are the arithmetic cores of programs, not eligibility determinations. Determinations belong in deterministic code. What the weights give you is an independent witness — a second implementation that agrees to the last digit, or tells you exactly where it doesn't.

Everything is public, including the failures and one retracted set of numbers I had to correct in the writeups. Total API cost of the whole program: $2.40.

https://github.com/ctrimm/weight-space-programming

---

## Shorter variant (if the long one feels heavy)

This week MIT posted "Sparks of Weight-Space Programming": a model writing neural network weights directly, no training. I tested it on US benefit formulas, where correctness is defined by law and checkable to the cent.

Written weights: exact on seven rules, worst error $0.0000000000005. Two outside models (GPT-5.6 Sol, Gemini 3.1 Pro) replicated the constructions cold.

Trained weights: average error $0.32 — and the answer changes depending on which spouse earns the income in 98.9% of equivalent households, by up to $25.15. The law says the couple's combined income is what counts.

Augmentation and symmetry penalties shrink that defect. They do not remove it. Building the rule into the architecture removes it. Compiling the rule into weights removes it and the error.

If you evaluate systems that decide benefits, test the properties the law implies, not just accuracy. Accuracy will certify the $25.15 problem every time.

Code, data, failures, and corrections: https://github.com/ctrimm/weight-space-programming
