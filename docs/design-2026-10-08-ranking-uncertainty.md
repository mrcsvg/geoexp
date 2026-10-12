# Design: ranking with uncertainty

Status: **accepted** 2026-10-08 by the maintainer, after the 2026-10-07 choice
of the feature and of covering both noise sources. Builds on 0.2.0; nothing
accepted there changes behaviour here. Amendments made during implementation
are marked *Amended during implementation* in place.

## Why this, first

Every market-selection tool in use — R GeoLift, the Google geo tooling, the
unofficial ports — returns a point ranking, as if the order were known.
This repository has already measured that it is not: geoexp and GeoLift agree
on 226 of 240 power cells and still order candidates at Kendall τ ≈ 0.40
(characterization §9). The README warns that "rank order by `mde` is stable
within an implementation, not across them". This release makes the API say
that with numbers instead of a sentence.

The quantity behind every ranking row is small. `power` for one cell is the
share of `lookback_window` placebo tests that detected the injected effect —
ten Bernoulli trials at the default used throughout this repo. `mde` is an
interpolation across six such cells. A design's rank is a nonlinear function of
a handful of coin flips, and the ranking frame reports it to three decimals.

Two sources of randomness feed a row, with very different costs:

| Source | What moves | How to measure it | Cost |
|---|---|---|---|
| **Which placebo windows** were in the lookback | the power rate itself, in steps of `1/lookback_window` | resample the windows already simulated | none — no refit |
| **Monte Carlo noise** in the conformal p-value (`ns` permutations) | cells whose p-value sits near `alpha` flip (§7.3: 225/240 identical between seeds) | re-run the simulation with another seed | one extra fit per replicate |

The maintainer chose (2026-10-07) to cover **both**. The design below does it
with **one resampling mechanism** over a pooled evidence base, so the two
sources add one parameter and one method rather than two features.

| Feature | Public surface | Oracle parity |
|---|---|---|
| Seed replicates | `rank_designs(..., replicates=)`; pooled `PowerResults` from `res.power()` | No — GeoLift has none; excluded, §6 |
| Ranking stability | `DesignRanking.stability()` → `RankStability` | No — GeoLift returns a point ranking; excluded, §6 |

## Maintainer decisions recorded here

`CLAUDE.md` reserves these for the maintainer. Each was accepted with the
design on 2026-10-08, at the recommendation stated:

1. **Public API changes** — new keyword `replicates` on `rank_designs`; new
   method `DesignRanking.stability`; new exported class `RankStability`. No
   existing behaviour changes at the defaults (`replicates=1`, method unused).
2. **New exclusions from the parity rule** — `replicates` and `stability`,
   recorded in `docs/geolift-oracle-characterization.md` §6 with the reason
   "GeoLift has no counterpart: it returns a point ranking and runs one
   simulation per cell". Covered by unit tests plus the `docs/methodology.md`
   entry in §5 below, and by a *measurement* (§6) that is recorded, not
   asserted.
3. **No new dependency.** polars and numpy do all of it.
4. **Point estimates pool the replicates** (§1). The alternative — keep
   replicate 0 as the point estimate and use the others only for spread —
   wastes the extra fits; rejected unless the maintainer prefers a ranking
   frame that is bit-identical across `replicates` values.
5. **Names and defaults** — `stability` / `RankStability`; `n_draws=500`,
   `level=0.90`, `top=3`. Alternatives considered: `rank_uncertainty`
   (longer, says the same), `resample` (says how, not what). Accepted as
   proposed; §12 of the characterization checks the defaults against the
   fixture.

## 1. `replicates` — seed replicates, pooled

### Surface

```python
rank_designs(..., replicates: int = 1)   # ValueError if < 1
```

With `replicates=k`, `simulate_power` runs `k` times per candidate, each with
its own child generator: `rng.spawn(len(candidates) * k)`, so the result is
order-independent and restartable under candidate-level parallelism exactly
as today. The `k × n` tasks share one `joblib.Parallel(n_jobs=n_jobs)` with
`n_jobs=1` passed down — the contract in `simulate_power`'s Notes, unchanged.

### Pooling

The `k` `PowerResults` of a candidate are pooled into one by concatenating
their `simulations` frames with a new `replicate` column (`0..k-1`) and
rebuilding `PowerResults(simulations, alpha, effect_type, params)` with the
first replicate's `params` (identical across replicates; only the generator
differed). Constructing `PowerResults` directly is a documented use of
augsynth-py's public API ("rebuilding from a filtered `simulations` frame"),
so the consumption rule holds.

Everything downstream is unchanged and automatically correct on the pool:
`power_curve()` groups by `(duration, effect_size)` and counts rows, so
`power` becomes a rate over `k × lookback_window` detections; `mde()`,
`_interp_mde`, `empirical_size`, `rmspe_pre`, `null_bias` and
`h1_calibration_error` are all means over `simulations` rows. **The ranking
frame's columns keep their names and meanings; with `k > 1` they are better
estimates of the same quantities.** A row at `replicates=3` is not expected to
equal the row at `replicates=1`; the CHANGELOG and the docstring say so.

`res.power(candidate, duration)` returns the pooled `PowerResults`. The
`replicate` column is the only trace of pooling and is ignored by every
augsynth-py method (they group by `duration` and `effect_size`).

### What it is not

Pooling replicates is **not** equivalent to raising `ns`. Each replicate
re-draws the permutations for the *same* placebo windows, so it tightens the
estimate of each window's detection indicator; it cannot add windows. The
dominant uncertainty — how many windows there are — is addressed by §2, and
`replicates` sharpens it only at the decision boundary. The docs say this so
nobody buys `replicates=10` expecting narrow intervals from `lookback_window=5`.

## 2. `stability` — one paired bootstrap over the evidence

### Surface

```python
res.stability(
    *,
    rng: np.random.Generator,   # required, no default (repo rule)
    n_draws: int = 500,
    level: float = 0.90,        # rank and MDE interval coverage
    top: int = 3,               # "in the top `top`" for p_top
) -> RankStability
```

```python
@dataclass(frozen=True)
class RankStability:
    frame: pl.DataFrame  # one row per row of res.ranking, same order
    n_draws: int
    level: float
    top: int
    pool_size: int  # replicates × lookback_window
```

`frame` columns:

| Column | Meaning |
|---|---|
| `candidate`, `duration`, `rank` | copied from `res.ranking` — the point ranking |
| `rank_low`, `rank_high` | `(1-level)/2` and `1-(1-level)/2` quantiles of the design's rank across draws, nearest-rank so they stay integers |
| `p_top` | share of draws in which the design's rank is `≤ top` (ties at the boundary count, as competition ranking already does) |
| `p_undetectable` | share of draws in which the resampled curve never reaches `target_power` (`mde` null, sorted last) |
| `mde_low`, `mde_high` | same quantiles of the interpolated `mde` over the draws where it is defined; both null when `p_undetectable > 1 - level`, because an interval over a minority of draws would mislead |

### Mechanism

The evidence base of one design is its pool of `(replicate, window)` pairs:
`pool_size = replicates × lookback_window` rows per `(duration, effect_size)`
cell. Every candidate ran the **same** windows (same calendar), so the pool
index is shared across candidates.

One draw samples `pool_size` indices **with replacement**, once, and applies
the same indices to **every** candidate — a paired bootstrap. Pairing matters:
a calendar window with a shock that hurts every synthetic control should hurt
every candidate in the same draw, or the draw would manufacture disagreement
between candidates that the data does not contain. Across replicates the
pairing is a no-op (different candidates used different child generators), so
it costs nothing and is harmless.

Per draw, per `(candidate, duration)`: recompute `power` per effect size from
the resampled rows, then `mde` via the existing `_interp_mde`, `mde_grid` via
the same grid rule as `PowerResults.mde`, `empirical_size` (the `effect_size
== 0` cell) and `rmspe_pre` (mean over the resampled rows). Those are exactly
the `_RANK_KEYS`, so `_apply_ranking` orders the draw's frame with the
production rule — same keys, same tie handling, same nulls-last — and the
draw's `rank` column is read off. No new ranking logic exists; `stability`
reuses the one that produces `res.ranking`.

Columns that do not enter the rank order (`null_bias`,
`h1_calibration_error`, `proportion_total_y`, `holdout`, `investment`,
`investment_mde`) are not resampled.

The designs ranked per draw are those **in `res.ranking`**. A `budget` is not
re-applied per draw: a design kept at its point `investment_mde` stays in the
comparison even if a draw's `mde_grid` would price it out. Re-applying the
budget per draw is a different question ("how often does this design fit?")
and is left out; §7.

### Cost and implementation shape

No refit. All candidates' `simulations` are concatenated once into a long
frame keyed by `(candidate, duration, effect_size, pool_idx)`, draws are
expressed as a `(draw, pool_idx, weight)` frame of multinomial counts, and a
join plus weighted `group_by` yields every draw's curves in one polars
pipeline. `_interp_mde` runs once per `(draw, candidate, duration)` — 40 000
calls at 500 draws × 40 candidates × 2 durations, a second or two. The whole
method is sub-minute on the parity fixture; it must stay cheap enough that the
docs can recommend calling it always.

### Validation and degenerate inputs

- `rng` missing → `TypeError` (keyword-only, no default).
- `n_draws < 1`, `level` outside `(0, 1)`, `top < 1` → `ValueError`.
- `pool_size == 1` (`lookback_window=1`, `replicates=1`) → `ValueError`:
  one window cannot be resampled. The message names both ways out.
- `pool_size < 10` → `UserWarning`, mirroring `rank_designs`'s own warning on
  small `lookback_window`: the bootstrap is crude on tiny pools.
- A `DesignRanking` not built by `rank_designs` (no `_results`) → `ValueError`,
  as `fit_path` already does.

### What it measures, and what it does not

This is the part that goes into `docs/methodology.md` verbatim.

- It measures how much the ranking depends on **which placebo tests and
  which permutation draws** produced it. Those are the two sources of
  randomness between `rank_designs` and its own inputs.
- It does **not** measure sampling variability of the panel (the pre-period
  is one realization and is never resampled), model misspecification, or the
  choice of estimator, grid, `alpha` or `permutation_type`. Those are the
  caller's design choices, and a stability band that contained them would be
  a different, much larger object.
- **Windows overlap.** Consecutive placebo windows share `duration − 1`
  periods, so the pool is serially dependent and a bootstrap over it
  understates the variance. The intervals are therefore **optimistic**: a
  design whose band already overlaps another's is tied at least that much. A
  block bootstrap over windows would be the usual remedy; with ten windows it
  has nothing to work with, so it is not offered. Say this plainly rather than
  dressing the output as a confidence interval.
- It is **not comparable to anything in GeoLift**, which has no equivalent.
  No parity test exists or can exist; §6 records the measurement that stands
  in for one.

## 3. Reading the output

Two usage patterns the docs should show, both one-liners on `frame`:

```python
stab = res.stability(rng=np.random.default_rng(11))
stab.frame.filter(pl.col("p_top") >= 0.9)  # the robust top set
stab.frame.filter(pl.col("rank_low") <= 1)  # anything that can win
```

A `robust_top()` method was considered and left out: it is a filter the user
can read, and a threshold with a name invites treating 0.9 as a standard.
If the maintainer wants the convenience, it is one method; §7.

Ties: two designs whose `[rank_low, rank_high]` overlap are not distinguished
by the evidence. The docs recommend reporting the robust set, not the winner.

## 4. Interaction with existing features

- `within_budget()` returns a new `DesignRanking`; `stability()` on it ranks
  the budgeted set (above). A `RankStability` is a snapshot and is not
  re-filtered.
- `plot_power` / `plot_fit` are unchanged. A rank-interval plot is presentation
  and is deferred; §7.
- `enumerate_candidates` is unaffected.
- `on_error="record"` leaves null `pvalue`s; `power_curve` already excludes
  them from the denominator, and the resampled curves do the same (a null
  pvalue never counts as detected and never counts in the denominator).
  *Amended during implementation:* a cell whose rows are all failed has null
  power, and `_interp_mde` read that as the smallest grid effect — a
  pre-existing bug the bootstrap made reachable (a draw can resample only the
  failed rows). `_interp_mde` now skips cells with unknown power. Recorded
  as a fix in the CHANGELOG.

## 5. Docs

- `docs/methodology.md`: new rows in the feature map (`replicates`,
  `stability`) and a section "Ranking stability" carrying §2's "what it
  measures" verbatim, plus the pooling note from §1.
- `docs/geolift-oracle-characterization.md` §6: two new exclusion rows. New
  §12 with the measurement from §6 below.
- README: three lines under the quickstart showing `stability` and the robust
  top set; the existing "rank order is stable within an implementation" bullet
  points to it.
- CHANGELOG under `[Unreleased]`, with the "How Claude was used" section the
  maintainer requires.

## 6. Testing

| Feature | Unit (no R) | Parity (rpy2) |
|---|---|---|
| `replicates` | `replicates=k` gives `k × lookback_window` rows per cell with `replicate` in `0..k-1`; pooled `power` equals the mean of the per-replicate powers; `res.power()` keeps `params`; reproducible across `n_jobs`; `replicates=0` raises; `replicates=1` is byte-identical to today | excluded |
| `stability` | one row per ranking row, same order; `p_top`, `p_undetectable` in `[0, 1]`; reproducible for a given `rng`; a synthetic panel with one dominant candidate gives `p_top = 1`, `rank_low = rank_high = 1`; a pool of identical windows gives zero-width intervals; `pool_size == 1` raises; small pool warns; budgeted result ranks only the kept set; each `_RANK_KEYS` column is recomputed (a test that corrupts `rmspe_pre` on one window moves that design's band) | excluded |

**Measurement, recorded in the characterization (§12), not asserted:** on
`GeoLift_PreTest`, `lookback_window=10`, the 40 single-unit candidates used
throughout §7 —

1. the distribution of `rank_high − rank_low` at `level=0.90`, and how many
   designs have `p_top ≥ 0.9` at `top=3`;
2. Kendall τ between the point ranking and each bootstrap draw's ranking.
   If the within-geoexp τ sits near the 0.40 measured against GeoLift in §9,
   the cross-implementation disagreement is explained by the evidence base,
   not by either implementation — the strongest statement available for a
   quantity with no oracle. *Amended during implementation:* it does not —
   measured median τ 0.94 within geoexp against 0.40 across; the
   disagreement is in the derivation of the scalar, not in the noise.
   Characterization §12 records it;
3. a crude coverage check: rank the same candidates at seed 99 and count how
   many seed-99 ranks fall inside the seed-7 band. Reported as a number, with
   the overlap caveat; not a test, because τ and coverage are themselves noisy.

Anything from this measurement that changes a default (`n_draws`, `level`)
is amended into this document, as the 0.2 doc did for lookback-averaged
`Investment`.

## 7. Not in this release

Per-draw budget re-application (`p_within_budget`); `robust_top()`; a
rank-interval plot; a block bootstrap over windows (no room with ten
windows); spillover neighbours; multi-cell (0.4.x, moved from 0.3.x because
this release took that line). Version bump and release
are the maintainer's.
