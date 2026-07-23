# Claim 4 — RingWorld learning-curve audit

**Verdict: BLOCKED**

No proxy training result is presented as evidence for this claim.

The paper reports seven Figure 3 variants, eight seeds each, and 30 million environment steps per run: 56 trainings and 1.68 billion environment steps. Even the minimum comparison named in the campaign contract—LOF, ALF, S5 at hidden size 12, and deep ALF—requires 32 trainings and 960 million environment steps.

The released paper source does not uniquely specify:

- episode length `K`;
- numerical beacon angles;
- the eight seed values;
- full transition entries for all three actions;
- numeric learning curves or checkpoints.

The paper and its author publication entry provide no direct implementation link; the exact-title GitHub search returned no repository. The available S5 RL repository is adjacent baseline code, not this paper's implementation. The fixed environment also intentionally contains no unpinned JAX/Flax/Optax/Distrax/Gymnax stack.

Accordingly, CPU-only training was not launched: doing so would require inventing material protocol details and could not support a faithful comparison. A deliberately incomplete 12-state/4-observation shell is included only as a negative control; the verifier rejects it because it has no PPO, S5, deep ALF, seeds, or training steps.

Machine-readable blocker, source, resource, and negative-control evidence is under `evidence/claim_4/`.
