# Repro - Why Linear Recurrent Memory Works in PORL

## Pages

| Page |
| --- |
| [Claim 1 — exact belief-logit reproduction](#/claim-1-exact-belief-logit-reproduction) |
| [Claim 2 — vanishing decoding error](#/claim-2-vanishing-decoding-error) |
| [Methods & environment](#/methods-environment) |
| [Negative controls & falsification](#/negative-controls-falsification) |
| [Conclusion](#/conclusion) |


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_b67a0ffd964d", "created_at": "2026-07-17T02:43:19+00:00", "title": "Linear Recurrent Memory (ywjHJIkUgW) — ICML 2026 reproduction"}
-->
Reproduction of 'Why Linear Recurrent Memory Works in Partially Observable RL' (arXiv 2605.31261, OpenReview ywjHJIkUgW). CPU-only (numpy/scipy). 2 claims, both VERIFIED with a negative control.

Headlines: C1 — a linear filter l_t = T l_{t-1} + log e_t EXACTLY reproduces the HMM pre-softmax belief logits under a deterministic transition matrix (max diff = 0.0, bit-exact); the proof is that the forward recursion's log-sum-exp collapses to a single term, linearizing in log-space. C2 — under a nearly-deterministic transition matrix the linear filter's state-decoding error vanishes (TV 0.50 -> 0.009 as the perturbation eps: 0.5 -> 0.001). Negative control: a STOCHASTIC transition matrix breaks the exact match (max diff 14.8), proving the linearity is specific to determinism. 3/3 unit tests pass.
