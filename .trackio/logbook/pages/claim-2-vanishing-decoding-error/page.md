# Claim 2 — vanishing decoding error


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_5e33321792f5", "created_at": "2026-07-17T02:43:20+00:00", "title": "C2 VERIFIED: decoding error vanishes as transitions -> deterministic"}
-->
Claim: the linear filter achieves vanishing state-decoding error under nearly-deterministic transition matrices.

Verification: perturb the deterministic transition as T_eps = (1-eps) T_det + eps T_stoch (column-stochastic), apply the linear filter (which assumes determinism), and measure the total-variation distance between the linear-filter belief and the true (log-forward) belief, averaged over a fixed observation sequence. Fixed obs across eps so only T varies:
  eps=0.5   TV=0.503
  eps=0.2   TV=0.418
  eps=0.1   TV=0.296
  eps=0.03  TV=0.142
  eps=0.01  TV=0.074
  eps=0.003 TV=0.026
  eps=0.001 TV=0.009
The decoding error vanishes monotonically toward 0 as the transition matrix -> deterministic. (At eps=0 it is exactly 0, by C1.)
