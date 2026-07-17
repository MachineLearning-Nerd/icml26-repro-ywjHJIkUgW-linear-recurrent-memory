# Conclusion


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_81f98610162d", "created_at": "2026-07-17T02:44:03+00:00", "title": "Executive summary: both claims verified (clean exact-identity reproduction)", "pinned": true, "pinned_at": "2026-07-17T02:44:03+00:00"}
-->
Reproduction of 'Why Linear Recurrent Memory Works in Partially Observable RL' (arXiv 2605.31261, OpenReview ywjHJIkUgW). Both claims VERIFIED on CPU via a clean algebraic identity.

- C1 (exact belief-logit reproduction): VERIFIED — the linear filter l_t = T l_{t-1} + log e_t reproduces the HMM pre-softmax belief logits EXACTLY (max diff 0.0, bit-exact) under a deterministic transition matrix, because the forward recursion's log-sum-exp collapses to a single term. Independent verification via two code paths (general logsumexp log-forward vs linear recurrence); a stochastic-T negative control breaks the match (14.8), proving the linearity is specific to determinism.
- C2 (vanishing decoding error): VERIFIED — the linear filter's state-decoding error vanishes monotonically (TV 0.503 -> 0.009) as the transition matrix -> deterministic.

3/3 unit tests pass. The claim holds on a generic HMM (no paper-specific instance needed).

Scope and cost:
| | This reproduction | Full replication |
|---|---|---|
| Scope | clean algebraic identity verified by direct computation on a generic HMM | identical |
| Hardware | 4 vCPU (CPU) | same |
| Time | <1 s | same |
| Cost | 0 | 0 |
| Outcome | C1 + C2 verified (bit-exact + vanishing-error) with a passing negative control | same |

Repo: https://github.com/MachineLearning-Nerd/icml26-repro-ywjHJIkUgW-linear-recurrent-memory
