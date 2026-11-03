# LinkedIn post — draft for Cory (2026-11-03, concise revision)

Graphics: `graphic1_stat_card.png` (lead image), `graphic2_bakeoff.png`
(second image in the carousel). Post text below the line.

---

TLDR: MIT's "Sparks of Weight-Space Programming" (Tomita et al.) says a frontier model can write a neural network's weights directly, no training. I tested it on US benefit formulas, where correct is defined by law. Written weights were exact. Trained weights were the problem.

Small models are my thing. I trained a custom on-device PII/PHI redaction model (a fine-tuned token classifier) that ships as a Mac app. I built a two-million-parameter speech-to-text model. So I ran this the way I'd want my own work run: pre-registered protocols, scored to the cent, failures published.

What I found:

- Seven benefit rules (SNAP, EITC, SSI individual and couple, Medicaid liability and share of cost) compiled into network weights by construction: exact. Worst error $0.0000000000005 across tens of thousands of test points.
- GPT-5.6 Sol and Gemini 3.1 Pro, given only the formula text, wrote exact networks too. In two programs, all three writers produced the same circuit up to relabeling.
- Networks trained on the same formulas reached a $0.32 average error, and changed an SSI couple's payment by up to $25.15 depending on which spouse earned the income. The law counts the couple's combined income, once. In 98.9% of equivalent households, the trained model gave different answers to the same household.
- Augmentation shrank that defect fivefold. A symmetry penalty did worse. An architecture that sums the incomes first eliminated it exactly. Compilation eliminated it and the error.

If you evaluate AI systems that touch benefits, accuracy alone will certify the $25.15 problem. Test the properties the law implies, not just the average error.

Scope: these are the arithmetic cores of the programs. Eligibility determinations belong in deterministic code. The weights are witnesses, independent implementations that agree to the last digit or tell you where they don't.

Everything is public, including the failures and one set of numbers I had to correct. Total API cost of the program: $2.40.

https://github.com/ctrimm/weight-space-programming
