# Methodology

What each public feature computes, and which published material it comes from.
Mirrors augsynth-py's `docs/methodology.md`.

Everything here is implemented clean-room from published material. The R
`GeoLift` package is a parity oracle driven via rpy2 in
`tests/validation_against_r/`; its source is never read for implementation and
never translated.

## Feature map

| Feature | Source | Where |
|---|---|---|
| Placebo-in-time power per candidate design | consumed from `augsynth_py.power` (Abadie 2021 §5; CWZ 2021) | `rank_designs` |
| Candidate ranking | geoexp construct, see below | `selection._apply_ranking` |
| Interpolated MDE | geoexp construct, see below | `selection._interp_mde` |
| Treated share / holdout | panel arithmetic | `selection._summarize` |
| Investment (`cpic`) | `cpic × effect_size × Σ Y_treated` over the treatment window | `selection._summarize` |
| Candidate enumeration | correlation criterion, see below | `enumerate_candidates` |
| Null bias / H1 calibration | see below | `selection._summarize` |
| Budget-constrained selection | GeoLift methodology publications | 0.2.x |

## The power layer is borrowed, not reimplemented

geoexp computes no power. `rank_designs` calls
`augsynth_py.simulate_power` once per candidate and reduces the returned curve
to a ranking row. Effect injection, refitting, and the conformal p-value are
augsynth-py's, and are validated there against the R oracle. The package
boundary is *"given a design"* (augsynth-py) versus *"choose a design"*
(geoexp).

Contract patterns this imposes, all documented in `simulate_power`'s Notes:
one child generator per candidate via `rng.spawn`; parallelism across
candidates with `n_jobs=1` passed down, because joblib does not nest workers.

## `mde` is a geoexp construct, not GeoLift's `Average_MDE`

**This is the most important thing on this page.** The `mde` column is not
comparable to R GeoLift's `Average_MDE` and no parity test asserts that it is.

`augsynth_py.PowerResults.mde` returns a **grid point** — the smallest
evaluated effect size whose power reaches the target. On the default six-point
grid that leaves most candidates tied: measured on the `GeoLift_PreTest`
fixture, the grid MDE takes **4 distinct values across 39 candidates**. A
ranking whose primary key has four values is not a ranking.

geoexp therefore interpolates. From the power curve it drops the
`effect_size == 0` row (that row is the empirical size, not a detectable
effect), enforces monotonicity by running maximum (simulated power curves dip),
and interpolates linearly to the first crossing of `target_power`. When the
curve never reaches the target it returns null — never an extrapolated number.
`mde_grid` carries the unmodified upstream grid point alongside, so the strict
answer is always one column away.

GeoLift's `Average_MDE` also lies off-grid — measured on the same fixture, none
of its values falls on the supplied grid and one is negative despite an
all-positive grid — but its derivation is not specified in the published
methodology, and establishing it would require reading the package source.
The two numbers are therefore near in spirit and not comparable in value.

**Rank order by `mde` is stable within an implementation, not across them.**
geoexp and GeoLift agree on 226 of 240 power cells (mean absolute difference
0.0063) yet order candidates at Kendall τ ≈ 0.40. Collapsing a curve into one
scalar amplifies noise where the curve crosses the decision threshold. Small
rank differences near ties carry no information; `power_at_target` and the full
curve are the stable quantities. See
`docs/geolift-oracle-characterization.md` §9.

## Candidate enumeration

`enumerate_candidates` ranks units by the Pearson correlation between each
unit's outcome series and the mean of every *other* unit's series, then draws
combinations of the requested sizes from the best-ranked units, capped per
size. A test market the donor pool tracks closely is one a synthetic control
can reconstruct, which is the published GeoLift guidance that test and control
areas should resemble each other historically; correlation is also GeoLift's
own documented default criterion (`dtw_emphasis = 0`).

It is **not** a reimplementation of GeoLift's selection step, which combines
correlation with dynamic time warping in a way the published material does not
specify. No parity test compares the two candidate lists, and none should. The
function returns a plain list of sets, so a caller who disagrees with the
criterion can build candidates any way they like and pass them to
`rank_designs` directly.

With `rng` supplied, candidates are sampled from the eligible pool instead of
taken in similarity order. Random sampling risks interpolation bias when units
are dissimilar --- the same caution GeoLift documents for
`run_stochastic_process` --- so the deterministic order is the default.

## Null bias and H1 calibration

Two diagnostics adapted from mrcsvg/augsynth-py#22.

- **`null_bias`** is the mean recovered lift when a zero effect was injected.
  It is the honest analog of GeoLift's `abs_lift_in_zero`: rather than trying
  to reproduce their number, whose definition is not published, geoexp reports
  the estimator's own recovered lift under the null. Signed, so the direction
  of the bias is visible. A design can have excellent power and still be
  systematically off.
- **`h1_calibration_error`** is the absolute gap between the effect recovered
  at `mde_grid` and the effect actually injected there. It catches the failure
  mode power cannot see: detected, but at the wrong magnitude. Defined for
  multiplicative effects only, where `att_pct` and the effect size are the same
  quantity; null for additive effects rather than a meaningless comparison.

**They overlap more than they look.** Measured on the `GeoLift_PreTest`
fixture, the two track each other closely (0.0317 vs 0.0329, 0.0268 vs 0.0273,
0.0403 vs 0.0408 for the top-ranked designs) --- expected, since an estimator
that recovers `e + bias` produces a calibration error of about `|bias|`. They
separate only when the bias depends on the effect size. Read
`h1_calibration_error` as a confirmation that the bias is effect-independent,
not as a second independent signal.

## Ranking criteria

Default order: `mde` ascending, then `empirical_size`, then `rmspe_pre`, then
the candidate key.

- **`mde`** answers the question the package exists to ask.
- **`empirical_size`** — power at `effect_size == 0` — guards against a design
  that "detects" because it over-rejects. It is a rate over the lookback
  window: at `lookback_window = 1` there is a single test and the value is 0 or
  1, a coin flip rather than a false-positive rate. `rank_designs` warns below
  10.
- **`rmspe_pre`** guards against a design whose pre-period fit was poor to
  begin with. It is augsynth-py's normalized pre-period RMSPE and is **not**
  GeoLift's `AvgScaledL2Imbalance` — a different normalization, measured at
  0.0334 against 0.4462 for the same candidate.
- **The candidate key** makes the order deterministic. It is deliberately not a
  rank key: designs identical on every substantive criterion share a rank.

## Features with no reproducible counterpart

Recorded here as the amended validation rule in `CLAUDE.md` requires. Each is
listed with its reason in `docs/geolift-oracle-characterization.md` §6, and is
covered by unit tests rather than parity tests.

| Column | Why no parity test |
|---|---|
| `mde` | GeoLift's `Average_MDE` derivation is not in published material; and rank order by MDE is not stable across implementations |
| `rank` | Composite over `mde`; inherits the above |

Every other public column is either asserted against the oracle
(`power`, `att`, `investment`) or is panel arithmetic with no oracle
counterpart to disagree with.

## References

- Abadie, A. (2021). *Using Synthetic Controls: Feasibility, Data Requirements,
  and Methodological Aspects.* Journal of Economic Literature 59(2). §5 covers
  in-time placebos, the basis of the power simulation consumed here.
- Chernozhukov, Wüthrich & Zhu (2021). *An Exact and Robust Conformal Inference
  Method for Counterfactual and Synthetic Controls.* JASA 116(536). Consumed
  via augsynth-py.
- GeoLift methodology publications (Meta) — the reporting shape this package
  mirrors. The R package itself is a parity oracle only.
