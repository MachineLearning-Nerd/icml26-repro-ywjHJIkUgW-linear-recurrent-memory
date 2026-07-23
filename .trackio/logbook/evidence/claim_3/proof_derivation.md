# Independent proof derivation for Corollary 4.5 and Theorem 5.9

## Exact action-controlled recurrence

For every action, `P(a)` has exactly one unit entry in every row and column.
Therefore, for every finite logit vector,

`logsumexp_j(log P(a)_ij + l_j) = (P(a) l)_i`.

Induction over the realized actions proves Corollary 4.5 for every finite
action sequence.  The numerical checker uses a separately written index path
over 64 arbitrary-initialization trials of horizon 127; its maximum error is
`5.684341886080802e-14`.  Replacing an action transition by a stochastic
matrix produces error `86.2514838846667`.

## Uniform near-permutation certificate

Assumption 5.8 gives positive, distinct emission columns, and every `P(a)` is
a bijection.  Hence any action maps a distinct state pair to another distinct
pair.  Conditioned on any realized nonanticipative action history, the two
hypotheses being compared therefore receive one of finitely many ordered
distinct emission pairs at every regular step.

For each ordered pair, define the one-step Appendix-D log-MGF integral with
`M=1`.  Positivity and distinctness give a strictly positive exponent for
every pair.  Taking the minimum over the finite pair set yields a uniform
`xi_A>0`, independent of the action sequence.  Conditional Chernoff factors
compose along the filtration, so the fixed-transition no-irregular-jump
argument applies with this worst-case exponent to every nonanticipative
policy.  Taking the worst irregular-jump rate over the finite action set gives
the same union bound:

`kappa_A epsilon log(1/epsilon)
 + o(epsilon log(1/epsilon))`.

The executable certificate checks every action is a permutation, every
action preserves pair separation, and every ordered emission pair has a
positive exponent.  For the declared eight-state, four-action model it obtains
`xi_A=0.026480946096894026` and `kappa_A=554.2122392876782`.  A dense
trapezoid checker independently reproduces the limiting exponent.

## Empirical corroboration and controls

The action-conditioned ALF and exact Bayes LOF are also evaluated on the
23-point epsilon grid.  ALF error falls from `0.19295` to `0.03700`, while
Bayes LOF falls from `0.16770` to `0.02590`.  The invalid
`delta=epsilon^2` schedule remains high and a wrong-action decoder remains
near chance or worse.  These finite results corroborate the proof certificate
but are not used to establish its universal quantifiers.

## Scope

The action-uniform argument relies on the paper's finite action/state spaces,
positive distinct emissions, permutation backbones, and nonanticipative
actions.  It does not claim the result for identical emission columns,
non-permutation backbones, continuous unbounded action families, or policies
that depend on future observations.  This is not a proof-assistant
formalization.
