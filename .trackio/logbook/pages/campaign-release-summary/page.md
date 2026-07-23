# Cumulative reproduction campaign

This page supersedes the verdict language on the historical July 17 pages while preserving those pages byte-for-byte as judged evidence. It does not claim a new judge score.

| Claim | Honest verdict | Direct result |
| --- | --- | --- |
| 1 — deterministic exact reproduction | **VERIFIED** | Independent log-forward and linear recurrences agree bit-exactly; stochastic control differs by 14.814854. |
| 2 — ALF asymptotic rate | **BLOCKED** | Exact ALF and Bayes-optimal LOF pass a paper-scale finite sweep, but a finite experiment cannot establish the theorem's universal double limit or its existential constant. |
| 3 — action-controlled extension | **BLOCKED** | Corollary 4.5 is verified structurally and numerically. Theorem 5.9 has direct finite support but retains the same universal-limit blocker. |
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

Historical pages remain reachable below. Their earlier Claim 2 “verified” wording is not the campaign's current verdict.
