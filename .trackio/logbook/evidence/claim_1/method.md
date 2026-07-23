# Claim 1 method

An eight-state, twelve-observation HMM with a seeded permutation transition is
run for 40 observations. Optimal logits are computed through the general
SciPy log-sum-exp forward algorithm and through the linear recurrence. The
maximum absolute discrepancy is recorded. A separately generated fully
stochastic column-stochastic transition is fed to both paths as a negative
control and must produce a clear mismatch.

This check is rerun unchanged by every child in the experiment tree.
