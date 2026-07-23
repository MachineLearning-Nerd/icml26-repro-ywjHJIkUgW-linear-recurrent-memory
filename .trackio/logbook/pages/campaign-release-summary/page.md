# Cumulative reproduction campaign

This page supersedes the verdict language on the historical July 17 pages while preserving those pages byte-for-byte as judged evidence. It does not claim a new judge score.

| Claim | Honest verdict | Direct result |
| --- | --- | --- |
| 1 — deterministic exact reproduction | **VERIFIED** | Independent log-forward and linear recurrences agree bit-exactly; stochastic control differs by 14.814854. |
| 2 — ALF asymptotic rate | **VERIFIED** | Exact ALF/Bayes LOF evidence is combined with an independent Appendix-D proof certificate: ξ=0.7206014018 and κ=8.82086 for the paper's two-state instance; dense quadrature, 416 bounded candidates, and assumption controls pass. |
| 3 — action-controlled extension | **VERIFIED** | Corollary 4.5 is exact. A uniform pairwise Chernoff certificate proves Theorem 5.9 for every nonanticipative sequence of the declared finite permutation actions; ξ=0.0264809461. |
| 4 — RingWorld learning curves | **BLOCKED** | The released source does not uniquely specify the protocol or provide direct code/raw curves, and the required 960M-step minimum comparison was not run. |
| 5 — illustrative two-state HMM | **VERIFIED** | Exact Section 6.1 protocol, 23 ε values, 4×20,000 trajectories per point, k=1000, valid/invalid rates, LOF, uncertainty, independent checker, and negative control. |

The fixed cumulative command uses UV and CPython 3.12.11 on CPU only. The final candidate is a descendant of baseline SHA `6890302b44515b8ac224a51c1e59feaa21d7f6df`; every child reruns all previously accepted checks.

Detailed evidence:

- [Claim 1 evidence](#/claim-1-exact-belief-logit-reproduction)
- [Claim 2 ALF audit](#/claim-2-alf-rate-audit)
- [Claim 3 action-controlled audit](#/claim-3-action-controlled)
- [Claim 4 RingWorld audit](#/claim-4-ringworld)
- [Claim 5 two-state reproduction](#/claim-5-two-state-hmm)
- [Evidence and provenance](#/evidence-manifest)

Historical pages remain reachable below. The current Claim 2 verdict rests on
the new proof certificate, not the historical qualitative TV sweep.
