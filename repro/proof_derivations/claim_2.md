# Independent proof derivation for Theorem 5.7

## Contract

The target is the theorem as quantified, not merely its two-state illustration.
Let `T = D + epsilon Q` satisfy Definition 5.1 and Assumptions 5.3 and
5.6.  On the recurrent class, `D` is a permutation with finite order `M`;
recurrent emission probabilities are positive; and the `M`-step stacked
emission trajectories of distinct recurrent states are distinct.

## Separation exponent

Fix an ordered pair of distinct recurrent initial states.  Follow both states
for one permutation period and let `Gamma` contain the offsets where their
emission distributions differ.  Assumption 5.6 makes `Gamma` nonempty.  For
each such offset define the log moment-generating function

`Lambda_n(t) = log E_p[(p(Y)/q(Y))^t]`.

Positivity makes every log likelihood ratio finite.  For a differing pair,
`Lambda_n(0)=Lambda_n(-1)=0`, its derivative at zero is the positive KL
divergence, and strict convexity gives the strict negativity needed in the
paper's Appendix-D integral.  Consequently

`xi_pair = -(1/M) sum_n integral_0^(-1/M) Lambda_n(t)/t dt > 0`.

There are finitely many ordered state pairs, so their minimum `xi` is also
strictly positive.  The executable certificate evaluates every pair rather
than assuming this sign.

## Finite constants and limits

The initial-logit bound and positive finite emissions give finite `W` and
finite pairwise KL divergences.  The Appendix-D threshold `alpha_1` is
therefore finite for every pair.  Choose `alpha` strictly above their finite
maximum, and choose `lambda=xi/2`.  Then

`delta_epsilon=lambda/log(1/epsilon) -> 0`

and

`epsilon/delta_epsilon =
epsilon log(1/epsilon)/lambda -> 0`.

Conditioned on no irregular jump in the `alpha/delta_epsilon` lookback
window, the Chernoff term is
`o(epsilon log(1/epsilon))`.  The transient-state contribution is `O(epsilon)`
and hence has the same little-o order.  A union bound on irregular jumps gives

`(-q_min) (alpha/lambda) epsilon log(1/epsilon)
 + o(epsilon log(1/epsilon))`.

Thus `kappa=(-q_min) alpha/lambda` is finite and positive and the theorem's
claimed upper rate follows.  The exact LOF/MAP implementation remains the
Bayes-optimal comparator in the empirical suite; its normalized error is
reported separately and is lower than ALF's throughout the aggregate check.

## Executable instantiation and independent checks

For the paper's two-state model, the certificate obtains
`xi=0.7206014018371381`, reproducing Appendix D's approximately `0.7206`,
and selects `kappa=8.820857715716253`.  SciPy adaptive quadrature is checked
against a separately implemented 200001-point trapezoid rule.  An exhaustive
bounded search covers 416 two- and three-state candidate models, of which 312
satisfy stacked observability; none has a non-positive certified exponent.
Identical emissions force `xi=0` and are rejected, and a non-permutation
backbone is rejected before integration.

## Scope

The universal step is a finite-state mathematical proof audit, not an
extrapolation from the bounded search.  The reported numeric `xi`, `alpha`,
and `kappa` are specific to the declared two-state instance; the verifier
computes them anew for any supplied finite model satisfying the contract.
This is not a proof-assistant formalization.
