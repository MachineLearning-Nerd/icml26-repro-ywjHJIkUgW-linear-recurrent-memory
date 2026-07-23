# Claim 3 method

Four actions use different permutations on eight states. A structural checker
certifies one unit entry per row and column, which is sufficient to prove the
log-sum-exp collapse for every finite logit vector. The executable comparison
then uses 64 arbitrary initial vectors, 127 action-observation steps each, and
the first-step correction from the corollary. A stochastic replacement for
one action is the exact-identity negative control.

For near-permutation transitions, each action uses
`T(a)=(1-epsilon)P(a)+epsilon U`, where `U` is uniform. A seeded randomized
policy exercises every action. Time-varying ALF Equation 65, exact
action-conditioned Bayes LOF, invalid `delta=epsilon^2`, and a wrong-action
control share paths over 23 inverse-epsilon values, 20,000 trajectories, and
1,000 steps. Wilson 95% intervals accompany all endpoint tests.

The proof-level route enumerates all actions and ordered state pairs. It first
certifies each action matrix is a permutation and preserves pair separation,
then computes the worst one-step Appendix-D exponent over the finite pair set.
This supplies a uniform conditional Chernoff factor for every
nonanticipative action history. Adaptive quadrature is checked by a dense
trapezoid rule; non-permutation and wrong-action controls must fail.
