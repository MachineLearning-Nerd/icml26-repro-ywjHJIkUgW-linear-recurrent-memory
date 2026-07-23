# Claim 3 — action-controlled extension

**Combined verdict: VERIFIED**

## Corollary 4.5: VERIFIED

For four action-dependent 8-state permutation matrices, a structural certificate checks exactly one predecessor per destination. Therefore the log-sum-exp transition collapses algebraically for every finite input logit vector; induction covers every finite action sequence. A separate numerical implementation tests 64 arbitrary-initialization trials of horizon 127 and obtains maximum absolute error `5.684341886080802e-14`. A stochastic action-transition control produces error `86.2514838846667`. Matrix and index-based independent implementations agree bit-exactly.

## Theorem 5.9: VERIFIED

A direct action-controlled near-permutation family is tested over the same 23-point ε grid, with 20,000 trajectories per point and k=1000. ALF error decreases from `0.19295` to `0.03700`; Bayes-optimal LOF decreases from `0.16770` to `0.02590`. The invalid `δ=ε²` schedule remains high (`0.77025` to `0.63570`), and a wrong-action negative control remains near chance or worse (`0.7984` to `0.83305`).

This is direct finite evidence with action conditioning, not an HMM proxy. A
separate proof-level route establishes the inherited double limit: every
action permutation maps a distinct state pair to another distinct pair, and
Assumption 5.8 separates every emission pair. The minimum one-step Chernoff
exponent over this finite pair set is therefore uniform over every realized
nonanticipative action history. For the declared four-action model,
`xi=0.026480946096894026` and `kappa=554.2122392876782`; independent dense
quadrature differs by `1.96e-14`.

The derivation and machine-readable certificate are
`evidence/claim_3/proof_derivation.md` and
`evidence/claim_3/theorem_certificate.json`.
