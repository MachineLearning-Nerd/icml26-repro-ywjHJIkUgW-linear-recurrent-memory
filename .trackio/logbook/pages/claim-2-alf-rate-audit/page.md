# Claim 2 — ALF asymptotic-rate audit

**Verdict: BLOCKED**

This reproduction implements the paper's Adaptive Logit Filter, not a generic perturbed linear filter. It also implements the exact Bayes-optimal log-odds filter (LOF) and the Section 6.1 two-state family.

The fixed sweep uses 23 equally spaced values of `1/ε` from 30 to 250, horizon `k=1000`, four deterministic replicate seeds, and 20,000 independent trajectories per ε per replicate. At the endpoints:

| Decoder | error at 1/ε=30 | 95% Wilson interval | error at 1/ε=250 | 95% Wilson interval |
| --- | ---: | --- | ---: | --- |
| ALF, δ=0.7/log(1/ε) | 0.07105 | [0.06929, 0.07285] | 0.0174375 | [0.016553, 0.018368] |
| Bayes-optimal LOF | 0.0511375 | — | 0.0111625 | — |

For the six smallest ε values, ALF error divided by `ε log(1/ε)` lies in `[0.7794, 0.8596]`; LOF lies in `[0.5061, 0.5464]`. The scalar two-state implementation and an independent full-vector implementation agree, and a label-swapped checker is rejected.

These results directly answer the judge's three finite-evidence criticisms: ALF is explicit, the claimed normalization is evaluated, and Bayes LOF is included. They do not prove Theorem 5.7. The theorem quantifies over a model class and takes `lim(k→∞)` followed by `limsup(ε→0)`; Appendix D defines κ through an existential “sufficiently large” α. A finite Monte Carlo sweep cannot establish those quantifiers or a unique finite κ.

Machine-readable evidence is under `evidence/claim_2/`.
