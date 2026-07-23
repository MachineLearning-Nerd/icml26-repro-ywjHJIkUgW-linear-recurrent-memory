# Claim 5 method

The simulator follows the column-stochastic convention. At each step the
deterministic backbone swaps the state; with probability epsilon the irregular
transition stays instead. The observation equals the state with probability
0.9. ALF is evaluated at every rate in Equations 67--68 on identical paths,
and LOF is evaluated as the exact two-state Bayes filter.

Raw rows include seed, epsilon, inverse epsilon, filter, adaptation rate,
checkpoint, error count, trajectory count, Wilson interval, and normalized
error. Checkpoints 900, 950, and 1000 expose stabilization; the predeclared
verdict uses the paper's `p_1000`.
