# Claim 4 blocker method

This node does not run a reduced experiment. It validates the full claim
contract, audits the released source and adjacent code, checks the frozen
environment for the cited JAX/Flax/Optax stack, and computes two independent
resource lower bounds.

Figure 3 contains seven variants. With eight seeds and 30 million steps each,
the complete comparison requires 56 training runs and 1.68 billion environment
steps. Thirty PPO epochs imply at least 50.4 billion sample-epoch passes.
Even the four variants minimally required by the campaign claim require 32
runs and 960 million environment steps.

The negative control presents a correct 12-state/4-observation shell with no
PPO, S5, Deep ALF, training timesteps, or seeds. The verifier must reject it,
ensuring that a toy environment cannot become a PASS.
