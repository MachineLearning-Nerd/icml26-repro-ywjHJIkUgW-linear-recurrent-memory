# Claim 3 — action-controlled extension

**Combined verdict: BLOCKED**

## Corollary 4.5: VERIFIED

For four action-dependent 8-state permutation matrices, a structural certificate checks exactly one predecessor per destination. Therefore the log-sum-exp transition collapses algebraically for every finite input logit vector; induction covers every finite action sequence. A separate numerical implementation tests 64 arbitrary-initialization trials of horizon 127 and obtains maximum absolute error `5.684341886080802e-14`. A stochastic action-transition control produces error `86.2514838846667`. Matrix and index-based independent implementations agree bit-exactly.

## Theorem 5.9: BLOCKED

A direct action-controlled near-permutation family is tested over the same 23-point ε grid, with 20,000 trajectories per point and k=1000. ALF error decreases from `0.19295` to `0.03700`; Bayes-optimal LOF decreases from `0.16770` to `0.02590`. The invalid `δ=ε²` schedule remains high (`0.77025` to `0.63570`), and a wrong-action negative control remains near chance or worse (`0.7984` to `0.83305`).

This is direct finite evidence with action conditioning, not an HMM proxy. It cannot prove Theorem 5.9's universal double-limit statement. Because the campaign claim combines the exact corollary and asymptotic theorem, the overall verdict is BLOCKED.

Machine-readable evidence is under `evidence/claim_3/`.
