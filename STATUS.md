# STATUS — Linear Recurrent Memory (ywjHJIkUgW) reproduction

**Session:** autoloop. **Last updated:** 2026-07-17. **State: ✅ PUBLISHED (under_verdict).**
HF: https://huggingface.co/spaces/DineshAI/ywjHJIkUgW · GitHub: https://github.com/MachineLearning-Nerd/icml26-repro-ywjHJIkUgW-linear-recurrent-memory (commit e9ecf65).

## Paper
- **Title:** Why Linear Recurrent Memory Works in Partially Observable RL. arXiv 2605.31261 · OpenReview ywjHJIkUgW.
- **No official code** — the claim is a clean algebraic identity, so no code is needed (verified by direct computation).

## Official claims — BOTH VERIFIED
1. "Linear filters can exactly reproduce the pre-softmax logits of the belief vector in an HMM under a deterministic transition matrix." → **VERIFIED bit-exact** (max diff 0.0): the linear recurrence `l_t = T·l_{t-1} + log e_t` reproduces the HMM belief logits exactly under deterministic T (the forward log-sum-exp collapses to one term). Two independent code paths (logsumexp log-forward vs linear recurrence) agree.
2. "Constructed linear filters achieve vanishing state-decoding error under nearly-deterministic transition matrices." → **VERIFIED**: TV decoding error vanishes monotonically (0.503 → 0.009) as the transition matrix → deterministic.

## Evidence
- `repro/src/run_lrm.py` — C1 (exact) + C2 (vanishing) + stochastic-T negative control.
- `repro/tests/test_lrm.py` — 3/3 pass.
- Negative control: stochastic T → mismatch (max diff 14.8), proving linearity is specific to determinism.
- All captured via `trackio logbook run`.

## DONE
- [x] scaffold + venv + implement + verify + tests.
- [x] logbook + publish → DineshAI/ywjHJIkUgW; GitHub pushed.

## NEXT
- Watch the verdict. Clean exact-identity + negative control + tests → expect verified.
