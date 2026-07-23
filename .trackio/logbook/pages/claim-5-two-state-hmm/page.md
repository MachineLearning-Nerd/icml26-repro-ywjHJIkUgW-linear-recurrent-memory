# Claim 5 — two-state HMM adaptation schedules

**Verdict: VERIFIED**

The implementation follows Section 6.1's finite illustrative experiment:

- `T=[[ε,1-ε],[1-ε,ε]]`;
- observations generated from the paper's symmetric noisy channel;
- 23 equally spaced values of `1/ε` from 30 to 250;
- horizon `k=1000`;
- four deterministic replicate seeds with 20,000 trajectories per point per seed;
- valid schedules `δ=√ε` and `δ=0.7/log(1/ε)`;
- invalid schedules `δ=ε²`, `δ=0`, and `δ=1`;
- exact Bayes-optimal LOF comparator.

Endpoint decoding errors:

| Schedule | 1/ε=30 | 1/ε=250 |
| --- | ---: | ---: |
| valid `0.7/log(1/ε)` | 0.07105 | 0.0174375 |
| valid `√ε` | 0.0816125 | 0.0372125 |
| invalid `ε²` | 0.4496875 | 0.3979 |
| invalid `0` | approximately 0.5003 | approximately 0.5003 |
| invalid `1` | approximately 0.09955 | approximately 0.09955 |
| Bayes-optimal LOF | 0.0511375 | 0.0111625 |

All predeclared finite checks pass. An independent full-vector ALF implementation agrees with the scalar recurrence, Wilson uncertainty intervals are emitted, and a label-swap negative control is rejected.

Machine-readable raw CSV and verifier outputs are under `evidence/claim_5/`.
