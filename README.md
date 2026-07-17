# Repro — Why Linear Recurrent Memory Works in PORL (ICML 2026)

Reproduction of *Why Linear Recurrent Memory Works in Partially Observable Reinforcement
Learning* (arXiv [2605.31261](https://arxiv.org/abs/2605.31261), OpenReview `ywjHJIkUgW`)
for the ICML 2026 Agent Reproduction Challenge. CPU-only (numpy/scipy); the claim is a
clean algebraic identity, so no official code is needed.

## Claims (both verified)
1. **Linear filters exactly reproduce HMM belief logits** under a deterministic transition
   matrix — the forward recursion's log-sum-exp collapses to a single term, giving the linear
   recurrence `l_t = T·l_{t-1} + log e_t` (a linear RNN). Verified bit-exact (max diff 0.0)
   via two independent implementations (logsumexp log-forward vs linear recurrence).
2. **Vanishing state-decoding error** under nearly-deterministic transitions — TV decoding
   error vanishes monotonically (0.503 → 0.009) as the transition matrix → deterministic.

**Negative control:** a stochastic transition matrix breaks the exact match (max diff 14.8),
proving the linearity is specific to determinism. 3/3 unit tests pass.

## Reproduce
```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python numpy scipy matplotlib pytest
.venv/bin/python repro/src/run_lrm.py     # C1 (exact) + C2 (vanishing) + control
.venv/bin/python -m pytest repro/tests/ -q
```

## Logbook
https://huggingface.co/spaces/DineshAI/ywjHJIkUgW
