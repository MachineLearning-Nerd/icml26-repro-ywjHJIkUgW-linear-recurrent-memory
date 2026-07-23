# Claim 3 source audit

- Source: `https://ar5iv.labs.arxiv.org/html/2605.31261`
- Retrieved: `2026-07-23T06:09:31Z` with explicit user agent
  `OpenResearch-Reproduction-Audit/1.0`
- SHA-256: `e6ed4e6be5e80ab9f65eac02be852add36872fc660fa780cfcf1e68ac0f94250`
- Corollary 4.5 anchor: `S4.Thmtheorem5`
- Near-permutation Equation 64 anchor: `S5.E64`
- Time-varying ALF Equation 65 anchor: `S5.E65`
- Assumption 5.8 anchor: `S5.Thmtheorem8`
- Theorem 5.9 anchor: `S5.Thmtheorem9`

Corollary 4.5 quantifies over action-controlled models whose transition for
each action is a permutation, and retains arbitrary initialization through a
time-dependent nonlinear input. Theorem 5.9 assumes every action transition is
`P(a)+epsilon Q(a)`, finite initial logits, strictly positive emissions, and no
identical emission columns. It inherits all Theorem 5.7 conclusions, including
the limits and rate conditions; it is not merely a statement that actions may
be concatenated to observations.

The independent audit makes the abbreviated proof step explicit: every action
permutation maps distinct hypotheses to distinct hypotheses, Assumption 5.8
separates every recurrent emission pair, and the finite minimum one-step
Chernoff exponent is uniform over realized nonanticipative action histories.
