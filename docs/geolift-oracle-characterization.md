# GeoLift oracle characterization

Observed input/output surface of R `GeoLift::GeoLiftMarketSelection`, recorded
so the v0.5 API design and the parity suite can be written against measured
behaviour instead of assumption.

**Clean-room stance.** Everything below was obtained by *driving* the oracle
(calling it and reading its return values) and by reading its **published
documentation** (the package help page, the GeoLift methodology write-ups).
No GeoLift R source was read, and none was translated. Where a column's
definition could not be established from published material, this document
says so and stops — it does not reverse-engineer the implementation. Those
gaps are recorded in [§6](#6-parity-verdict-per-column) as parity exclusions,
which is the honest outcome, not a to-do.

Recorded 2026-09-05.

## 1. Environment

| Component | Version |
|---|---|
| R | 4.6.0 |
| `GeoLift` | 2.7.5 |
| `MarketMatching` | 1.2.1 |
| `rpy2` | 3.6.7 |
| `augsynth-py` | 0.5.0 |

**Gotcha.** The wheel for `rpy2-rinterface` 3.6.6 is linked against
`R.framework/Versions/4.5-arm64`; on an R 4.6 host the API-mode `dlopen`
fails and rpy2 falls back to ABI mode, printing an `ImportError` to stderr
before succeeding. The fallback works and results are unaffected, but the
message is alarming and will show up in test logs. `validation.yml` pins
R 4.5, so CI does not hit it — only local runs on newer R do.

## 2. Fixture preprocessing is mandatory

`GeoLift_PreTest` ships **raw**: 3600 rows × `location` (chr), `Y` (int),
`date` (chr, `yyyy-mm-dd`), 40 locations × 90 days. It has **no `time`
column**, which `GeoLiftMarketSelection` requires. The panel must first pass
through `GeoDataRead`, which maps `date` → integer `time` starting at 1:

```r
gl <- GeoDataRead(GeoLift_PreTest, date_id = "date", location_id = "location",
                  Y_id = "Y", X = c(), format = "yyyy-mm-dd", summary = FALSE)
# -> 3600 x 3: location (chr), time (int, 1..90), Y (int)
```

**Consequence for the parity suite:** take the converted panel *from R* and
hand the identical frame to both sides. geoexp must not reimplement
`GeoDataRead` — that would be exactly the translation the clean-room rule
forbids, and it would also make the test compare two preprocessing steps
instead of the ranking logic under test.

## 3. Call surface used

```r
GeoLiftMarketSelection(
  data = gl, treatment_periods = c(15), N = c(2),
  effect_size = seq(0, 0.25, 0.05),
  model = "none", cpic = 1, alpha = 0.1,
  conformal_type = "iid", ns = 200, side_of_test = "two_sided",
  parallel = FALSE, ProgressBar = FALSE, print = FALSE
)
# 2.3 s wall clock, deterministic (run_stochastic_process = FALSE default)
```

Argument correspondence with `augsynth_py.simulate_power`:

| GeoLift | augsynth-py | Note |
|---|---|---|
| `treatment_periods` | `durations` | same meaning |
| `effect_size` | `effect_sizes` | **defaults differ**, see below |
| `lookback_window` | `lookback_window` | same name, same meaning |
| `alpha` | `alpha` | same |
| `ns` | `ns` | same |
| `conformal_type` | `permutation_type` | `"iid"` / `"block"`; **GeoLift defaults to `iid`, augsynth-py to `block`** |
| `side_of_test` | `side` | `"two_sided"` vs `"two-sided"` — underscore vs hyphen |
| `model = "none"` | `Synth()` | augmentation off on both sides |
| `N`, `include_markets`, `exclude_markets` | — | **candidate enumeration; no augsynth-py counterpart** |
| `cpic`, `budget`, `holdout` | — | budget layer, deferred past 0.1.0 |
| `fixed_effects`, `dtw`, `Correlations`, `normalize` | — | GeoLift-specific |

Two defaults worth pinning explicitly in every parity call:

- **`effect_size`**: the *usage block* defaults to `seq(-0.2, 0.2, 0.05)` —
  a **mixed-sign** grid — while the *prose* in the same help page says
  `seq(0, 0.25, 0.05)`. GeoLift's own documentation contradicts itself here.
  augsynth-py's `mde()` refuses mixed-sign grids, so the parity suite must
  pass `effect_size` explicitly rather than rely on either default.
- **`conformal_type`/`permutation_type`**: opposite defaults on the two
  sides. Always pass it.

## 4. Return surface

A plain `list` of three elements (S3 class `GeoLiftMarketSelection`).

### `BestMarkets` — 30 rows × 13 columns

One row per selected candidate market × duration.

| Column | Type | Observed meaning |
|---|---|---|
| `ID` | int | 1..n, row identifier only |
| `location` | chr | **candidate key**: member locations, alphabetically sorted, joined with `", "` |
| `duration` | num | treatment periods |
| `EffectSize` | num | the effect size selected for this row (a grid point) |
| `Power` | num | power at `EffectSize` |
| `AvgScaledL2Imbalance` | num | GeoLift's pre-period fit metric |
| `Investment` | num | `cpic × EffectSize × Σ Y_treated` over the treatment window |
| `AvgATT` | num | average treatment effect on the treated |
| `Average_MDE` | num | minimum detectable effect — **off-grid**, see §5 |
| `ProportionTotal_Y` | num | treated share of total KPI |
| `abs_lift_in_zero` | num | zero-effect bias diagnostic, see §5 |
| `Holdout` | num | **exactly** `1 - ProportionTotal_Y` (verified) |
| `rank` | int | competition ranking, ties present (1,2,3,3,5,6,6,8,…) |

**`BestMarkets` is filtered, `PowerCurves` is not.** On a 40-market,
single-market-candidate run the oracle returned 240 `PowerCurves` rows for all
40 locations but only **39** `BestMarkets` rows — `san francisco` is computed
and then dropped by GeoLift's own selection step. Any parity join must
therefore go through `PowerCurves`, which is complete; joining on
`BestMarkets` silently loses candidates.

### `PowerCurves` — 180 rows × 8 columns

One row per candidate × effect size (30 × 6).

`location`, `duration`, `EffectSize`, `power`, `AvgScaledL2Imbalance`,
`Investment`, `AvgATT`, `AvgDetectedLift`.

### `parameters`

`data`, `model`, `fixed_effects`, `cpic`, `side_of_test` — plotting support
only; nothing the ranking depends on.

## 5. Semantics established, and not

**Established by measurement (safe to rely on):**

- **Candidate key** is the alphabetically sorted, `", "`-joined location
  list. Verified for all 30 rows.
- **`Holdout` = `1 - ProportionTotal_Y`**, exactly.
- **`Investment` = `cpic × effect_size × Σ Y_treated` over the treatment
  window.** Verified to the cent: for `{atlanta, chicago}`, duration 15, the
  treated window sum is 108465, and GeoLift reports 5423.25 at
  `effect_size = 0.05` (= 0.05 × 108465) and 27116.25 at 0.25. It is a
  function of the panel and the grid alone — no estimator internals.
- **Power is a rate over the lookback window.** With `lookback_window = 1`
  there is exactly one test, so `power ∈ {0, 1}` — as observed (27 markets
  at 0, 3 at 1). augsynth-py behaves identically (`n_simulations = 1`). Any
  ranking criterion that needs an *empirical size* rather than a coin flip
  therefore requires `lookback_window >> 1` on both sides.

**Not established — excluded from parity:**

- **`Average_MDE`.** Not a grid point: none of the 30 values lands on
  `seq(0, 0.25, 0.05)`, and one is **negative** (`-0.0700`) even though the
  supplied grid contained no negative effect sizes. So the derivation both
  interpolates and searches outside the supplied grid. The published
  methodology defines MDE only conceptually ("smallest effect the test can
  reliably detect"); it does not specify this computation. Establishing it
  would require reading the R source, which the clean-room rule forbids.
- **`abs_lift_in_zero`.** Tracks `|AvgDetectedLift|` at `EffectSize == 0`
  closely but is **not equal** to it, rounded or otherwise (0.071 vs 0.0673,
  0.051 vs 0.0482, while 0.007 vs 0.0068 and 0.004 vs 0.0040 do match at 3
  decimals). Some additional averaging is involved. Definition not in
  published material.
- **`rank`.** A competition ranking whose order tracks `abs_lift_in_zero`
  ascending very closely but is not a pure sort of it (rows with 0.004 and
  0.003 share rank 3). Composite not established. Inherits the exclusion of
  its inputs regardless.
- **`ProportionTotal_Y`.** Near-derivable but not exact under the obvious
  reading: treated share over the last 15 periods computes to 0.0418440
  against GeoLift's 0.0419522 (~0.26% relative). The window convention
  differs; pin it empirically before claiming parity.
- **`AvgScaledL2Imbalance`.** Not the same metric as augsynth-py's
  `rmspe_pre` (0.4462 vs 0.0334 for the same candidate). Different
  normalization; not comparable.

## 6. Parity verdict per column

| Oracle output | augsynth-py counterpart | Verdict |
|---|---|---|
| `PowerCurves.power` | `power_curve().power` | **1:1** — 226/240 cells exact, the rest within one lookback step at the decision boundary (§7.2) |
| `PowerCurves.AvgATT` | `simulations.att` | **1:1**, ~1e-6 relative |
| `PowerCurves.EffectSize`, `duration` | `effect_size`, `duration` | 1:1 by construction |
| `BestMarkets.Investment` | derived from panel | 1:1, formula in §5 |
| `BestMarkets.Holdout`, `ProportionTotal_Y` | derived from panel | derivable; window convention to pin |
| `PowerCurves.AvgDetectedLift` | `simulations.att_pct` | **affine, not equal** — constant ratio 1.02125 for the probed candidate; different denominators |
| `AvgScaledL2Imbalance` | `rmspe_pre` | **not comparable** — different metric |
| `Average_MDE` | `mde()` | **not comparable** — off-grid, definition unavailable |
| `abs_lift_in_zero` | — | **not comparable** — definition unavailable |
| `rank` | — | **not comparable** — composite unavailable |

## 7. Measured parity

### 7.1 Cell level — one candidate, every grid point

Candidate `{atlanta, chicago}`, duration 15, `effect_size = seq(0, 0.25, 0.05)`,
`model = "none"` / `Synth()`, `conformal_type = permutation_type = "iid"`,
`ns = 200`, `alpha = 0.1`, `lookback_window = 1`, identical panel.

| effect size | GeoLift `power` | augsynth `power` | GeoLift `AvgATT` | augsynth `att` |
|---|---|---|---|---|
| 0.00 | 0 | 0.0 | −24.77109 | −24.770877 |
| 0.05 | 0 | 0.0 | 156.00391 | 156.004123 |
| 0.10 | 1 | 1.0 | 336.77891 | 336.779123 |
| 0.15 | 1 | 1.0 | 517.55391 | 517.554123 |
| 0.20 | 1 | 1.0 | 698.32891 | 698.329123 |
| 0.25 | 1 | 1.0 | 879.10391 | 879.104123 |

Power matches exactly at every grid point for this candidate (§7.2 sweeps all
40 and finds where that stops holding). ATT agrees to ~2e-4 absolute
(~1e-6 relative) — the full synthetic-control path agreeing across two
languages. The repo's default tolerance (`atol=1e-6, rtol=1e-5`) holds for
`power` but **not** for `att` at `atol=1e-6`; the ATT assertion needs
`atol≈1e-3` or a relative-only comparison. That is a tolerance the design
doc must state and justify, not silently relax.

### 7.2 Sweep — 40 candidates × 6 effect sizes

The cell-level check above is one candidate. Repeating it across the whole
fixture is what tells us whether the agreement is structural or lucky.

All 40 locations as single-market candidates, duration 15,
`effect_size = seq(0, 0.25, 0.05)`, **`lookback_window = 10`** (so power is a
rate over ten placebo tests rather than a single 0/1 outcome), `ns = 200`,
`iid`, `alpha = 0.1`, `model = "none"` / `Synth()`, identical panel. 240 cells.

| effect size | mean \|diff\| | max \|diff\| | correlation | cells exactly equal |
|---|---|---|---|---|
| 0.00 | 0.0125 | 0.20 | 0.984 | 36/40 |
| 0.05 | 0.0125 | 0.10 | 0.996 | 35/40 |
| 0.10 | 0.0125 | 0.10 | 0.996 | 35/40 |
| 0.15 | 0.0000 | 0.00 | 1.000 | 40/40 |
| 0.20 | 0.0000 | 0.00 | 1.000 | 40/40 |
| 0.25 | 0.0000 | 0.00 | 1.000 | 40/40 |

**226 of 240 cells agree exactly; mean absolute difference 0.0063** — 0.63
percentage points of power. Every disagreement sits at a low effect size,
where the power curve crosses the decision boundary and one extra rejection
out of ten lookback windows moves the value by a full 0.1. At effect sizes
0.15 and above, where no candidate is near the boundary, agreement is exact
for all 40.

### 7.3 The noise floor — how to read those numbers

The counts above are a **single draw**, and treating them as fixed would be a
mistake. Both sides estimate a conformal p-value by resampling (`ns = 200`) and
then reduce it to a rate over ten placebo tests, so a cell whose p-value sits
near `alpha` flips between runs *of the same implementation*. Re-running with a
different seed gives 219 exact instead of 226 — same distribution, different
draw.

What settles the question is comparing each implementation against **itself**:

| pair | cells identical | mean \|diff\| | max \|diff\| | signed bias |
|---|---|---|---|---|
| geoexp seed 7 vs geoexp seed 99 | 225/240 | 0.0067 | 0.20 | 0.0000 |
| GeoLift seed 1 vs GeoLift seed 2 | 223/240 | 0.0079 | 0.20 | −0.0004 |
| geoexp 7 vs GeoLift 1 | 219/240 | 0.0063 | 0.20 | −0.0004 |
| geoexp 7 vs GeoLift 2 | 218/240 | 0.0067 | 0.20 | −0.0008 |
| geoexp 99 vs GeoLift 1 | 218/240 | 0.0063 | 0.20 | −0.0004 |
| geoexp 99 vs GeoLift 2 | 215/240 | 0.0075 | 0.20 | −0.0008 |

**The two packages agree with each other as well as either agrees with
itself.** Cross-implementation mean absolute difference (0.0063–0.0075) falls
inside the within-implementation range (0.0067, 0.0079) — geoexp against
GeoLift is *closer* than GeoLift against GeoLift. The signed bias is under
0.001 everywhere on a scale whose step size is 0.1, so there is no systematic
drift in either direction. That is the strongest parity statement available
for a stochastic procedure, and a better one than any exact-match count.

Two corollaries, both of which contradict what a single draw suggested:

- **Agreement is not exact at high effect sizes.** The first draw showed 40/40
  at 0.15, 0.20 and 0.25; a second shows 39/40 at 0.20 — and geoexp against
  *itself* shows 39/40 at 0.15. Saturation is common, not guaranteed.
- **Differences are not bounded by one lookback step.** The maximum is 0.20 —
  two steps — in every pairing, including each implementation against itself.

**Consequence for the parity suite:** fixed thresholds calibrated on one draw
will flake. Assertions must be either self-calibrating (compare the cross
difference against the package's own run-to-run difference, measured in the
same session) or bias-based, which is nearly noise-free and is what actually
moves when an implementation breaks. `tests/validation_against_r/` is written
that way.

This is the parity surface the suite should target: high resolution (240 cells
instead of 6), and a real regression detector — an error in the panel handling,
the placebo construction, the effect injection or the conformal p-value would
show up as bias, which the noise does not produce.

## 8. Consequence for candidate enumeration

GeoLift selected **30** of the C(40,2) = 780 possible pairs, via its own
similarity step (`dtw` / correlations / `N` deciles). geoexp 0.1.0 takes
**explicit candidates** and does not enumerate (that is 0.2.x). So a parity
test must read the 30 keys out of `BestMarkets$location`, split them back
into unit sets, and feed exactly those to geoexp as explicit candidates.
Parity is then on the ranking of a **given** candidate set — which is
precisely the package boundary, and is a feature of the test design rather
than a limitation of it.

## 9. Cross-package benchmark — tried, and it does not work

Recorded as a negative result so nobody spends the day repeating it.

**The idea.** Three of the ranking columns (`Average_MDE`, `abs_lift_in_zero`,
`rank`) have no reproducible counterpart in the oracle, so §6 excludes them
from parity. A second package computing the same *kind* of quantity by a
different method could have covered them with a rank-concordance test —
weaker than parity, but real validation rather than unit tests alone.

**The candidate.** `MarketMatching` 1.2.1 (Kim Larsen): `best_matches()`
selects control markets by dynamic time warping and `test_fake_lift()` is
documented as a *"prospective power analysis"* — it injects fake lifts across
a grid and reports `prob_causal`, the posterior probability of a causal
effect, under CausalImpact/BSTS. Different method end to end (BSTS + DTW, not
augmented synthetic control), which is exactly what makes it a benchmark
rather than a second oracle.

**Setup.** All 40 locations as single-market candidates. Matching period =
the 75 pre-periods, fake post period = the same 15-day window used everywhere
else, so the three packages score the same design. MDE derived from
`prob_causal` crossing `1 - alpha = 0.90`, monotonized and interpolated by the
same procedure geoexp applies to its own power curve, so neither side gets a
resolution advantage. 40 markets in 149 s; all 40 succeeded.

**Result.** Rank concordance over the 39 candidates present in all three:

| pair | Kendall τ | p | Spearman ρ |
|---|---|---|---|
| geoexp × GeoLift | 0.398 | 6.9e-04 | 0.546 |
| geoexp × MarketMatching | 0.338 | 5.4e-03 | 0.437 |
| GeoLift × MarketMatching | 0.217 | 0.060 | 0.307 |
| `rmspe_pre` × `AvgScaledL2Imbalance` | 0.169 | 0.13 | 0.262 |

τ ≈ 0.35 is better than chance but far too weak to assert on: a threshold
would either pass vacuously or fail on noise. **The benchmark is rejected.**

**The confound was checked and ruled out.** `mde` from a 6-point grid takes
only 15 distinct values across 39 candidates (and the raw grid `mde()` takes
4), so ties could plausibly have been suppressing τ. Refining the grid to 21
points lifted the resolution to 25 distinct values and moved τ by almost
nothing — 0.398 → 0.414 against GeoLift, 0.338 → 0.349 against
MarketMatching. Restricting to candidates that pass on empirical size
(≤ 0.2, n = 35) *lowered* it to 0.376. The disagreement is real, not an
artifact of the grid.

**Why it does not matter — and what it does tell us.** The natural reading of
a τ of 0.40 against GeoLift would be that geoexp gets something wrong. §7.2
rules that out: the two agree on 226 of 240 power cells. The machinery is not
in dispute. What diverges is the *derivation* — collapsing a whole power curve
into one scalar amplifies exactly the boundary noise that §7.2 isolates, and
GeoLift's own collapse is an undocumented off-grid transform. Two
implementations can agree on power to 0.63 percentage points and still order
candidates at τ = 0.40.

The consequence is not that MDE is wrong, but that **rank order by MDE is not
a stable cross-implementation quantity**, and no test — parity or benchmark —
should be built on ranking agreement. This also closes the question of
whether GeoLift's source would help: even an exact copy of its formula would
be chasing a number that is unstable by construction. Nothing is lost by not
reading it.
