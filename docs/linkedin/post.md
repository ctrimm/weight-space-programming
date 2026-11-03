# LinkedIn post — draft for Cory (2026-11-03, plain-language revision)

Graphics: `graphic1_stat_card.png` (lead image), `graphic2_repairs.png`
(second image in the carousel). Post text below the line.
(Supersedes the denser concise draft of the same day, which assumed
too much reader context.)

---

TLDR: MIT's "Sparks of Weight-Space Programming" says an AI can set the numbers inside a neural network directly, instead of training it. I tested that on US benefit formulas. It works. The usual way of building these models has a problem that accuracy scores do not catch.

Small, specialized models are my work. I trained a redaction model that runs on your device and ships as a Mac app. I built a speech-to-text model with two million parameters. Benefit formulas are a good test for a claim like this because the right answer is written in law. There is a formula. There is no opinion.

So instead of training networks on examples, I calculated their numbers straight from seven formulas: SNAP, the earned income tax credit, SSI for individuals and couples, and two Medicaid cost-sharing rules.

The results were exact. Across tens of thousands of test cases, the largest error was $0.0000000000005, which is rounding noise, not a wrong answer. Given the same formulas as plain text, GPT-5.6 Sol and Gemini 3.1 Pro wrote exact networks too.

Then I trained networks the standard way. On average they looked fine: 32 cents average error. But SSI counts a married couple's combined income. Which spouse earned it does not matter. The trained network's answer depended on exactly that in 98.9% of cases, moving a couple's payment by as much as $25.15 for the same household with the same total income.

More varied training data shrank the gap. It did not close it. The gap closed only when the rule was built into the model's design, so the model could not treat the two spouses differently in the first place.

If you evaluate AI that touches public benefits: a model can score well on average and still treat identical households differently. Test for that directly.

This covers the arithmetic only. Eligibility is a legal determination, and it belongs in ordinary, auditable code.

Everything is public, including the failures and one set of numbers I had to correct. Total API cost: $2.40.

https://github.com/ctrimm/weight-space-programming
