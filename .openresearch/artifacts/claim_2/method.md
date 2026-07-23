# Claim 2 method

The executable suite uses Eq. (66):

`T=[[epsilon,1-epsilon],[1-epsilon,epsilon]]` and
`E=[[0.9,0.1],[0.1,0.9]]`.

It initializes `pi_0=[1,0]^T` and both ALF and LOF logits to `[0,0]^T`,
then runs the paper's 23 uniformly spaced inverse-epsilon values from 30 to
250 for 1,000 steps and 20,000 common-random-number trajectories per value.
The primary ALF schedule is `0.7/log(1/epsilon)`. Exact LOF supplies the
Bayes-optimal MAP comparator. Wilson 95% intervals accompany every estimate.

A scalar log-odds implementation is checked against separately written full
two-vector ALF and log-sum-exp LOF recurrences. A label-swap corruption must
be rejected by the Claim 5 verifier. The suite records normalized errors
`p/(epsilon log(1/epsilon))`, but does not reinterpret a finite sweep as a
proof of either limit.
