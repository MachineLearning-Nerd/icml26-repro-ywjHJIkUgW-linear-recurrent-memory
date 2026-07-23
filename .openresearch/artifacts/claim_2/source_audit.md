# Claim 2 source audit

- Source: `https://ar5iv.labs.arxiv.org/html/2605.31261`
- Retrieved: `2026-07-23T06:09:31Z` with explicit user agent
  `OpenResearch-Reproduction-Audit/1.0`
- SHA-256: `e6ed4e6be5e80ab9f65eac02be852add36872fc660fa780cfcf1e68ac0f94250`
- Definition 5.1 anchor: `S5.Thmdefinition1`
- ALF Equation 28 anchor: `S5.E28`
- Assumption 5.3 anchor: `S5.Thmtheorem3`
- Assumption 5.6 anchor: `S5.Thmtheorem6`
- Theorem 5.7 anchor: `S5.Thmtheorem7`
- κ definition anchor: `A4.E232`
- Illustrative experiment anchor: `S6.SS1`

The exact ALF recurrence is
`w_k = diag(P,I_r)(1-delta)w_(k-1) + delta log(E^T y_k)`.
The theorem first requires `delta_epsilon -> 0` and
`epsilon/delta_epsilon -> 0`, then states a double-limit vanishing result.
For `delta=lambda/log(1/epsilon)`, `lambda in (0,xi)`, it gives an
`epsilon log(1/epsilon)` upper-order result and compares this dominated rate
with the Bayes-optimal decoder.

Appendix D defines `kappa=-q_min alpha/lambda`, where alpha is chosen
sufficiently large to make a proof term negative. The paper does not identify
a unique minimal alpha for a finite numerical bound. Section 6.1 itself calls
`p_1000` an empirical proxy for the long-run limsup. These restrictions are
part of the contract, not hidden limitations.
