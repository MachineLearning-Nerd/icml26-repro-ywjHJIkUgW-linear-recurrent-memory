# Methods & environment


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_55ab3e404be7", "created_at": "2026-07-17T02:43:21+00:00", "title": "Methods: two independent implementations of the belief logits"}
-->
Paper: arXiv 2605.31261 (no official code; the claim is a clean algebraic identity derived from the HMM forward recursion, so no code is needed — the verification is by direct computation).

Two independent code paths for the belief logits, then compared:
- general log-forward: log alpha_t(i) = log e_t(i) + logsumexp_j( log T[i,j] + log alpha_{t-1}(j) ) — correct for ANY transition matrix T (scipy.special.logsumexp).
- linear filter (linear RNN): l_t = T l_{t-1} + log e_t — the paper's construction, valid when T is deterministic.
Deterministic T = a column-stochastic matrix with a single 1 per column (a permutation). Nearly-deterministic = T_eps = (1-eps) T_det + eps T_stoch.
Environment: Python 3.12, numpy, scipy. CPU-only, <1 s runtime. No GPU, no training. The claim is verified on a GENERIC HMM (any deterministic transition reproduces the result), so no paper-specific instance is needed.
