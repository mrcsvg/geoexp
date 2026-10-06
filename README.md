# geoexp

Geo-experiment **design** on top of
[augsynth-py](https://github.com/mrcsvg/augsynth-py): market selection,
power-based design ranking, and experiment reporting.

The boundary between the two packages is *"given a design"* vs. *"choose a
design"*:

- **augsynth-py** answers *what can this design detect?* — synthetic-control
  estimators, conformal inference, and power/MDE for a treated set the caller
  already fixed.
- **geoexp** (this package) answers *which design should I run?* — ranking
  candidate treated sets x durations by what each can detect, with
  GeoLift-shaped reporting. It consumes augsynth-py through its public API
  only.

## Quickstart

```python
import numpy as np
import polars as pl
from augsynth_py import AugSynth

from geoexp import enumerate_candidates, rank_designs

# Build candidates, or pass your own explicit sets.
candidates = enumerate_candidates(
    panel,
    unit="location",
    time="time",
    outcome="Y",
    sizes=[1, 2],
    max_per_size=15,
)

res = rank_designs(
    panel,  # long format: unit, time, outcome
    estimator=AugSynth(lambda_=1.0),  # freeze the penalty for large grids
    unit="location",
    time="time",
    outcome="Y",
    candidates=candidates,
    durations=[15, 30],
    lookback_window=10,
    permutation_type="iid",
    rng=np.random.default_rng(7),
    n_jobs=-1,  # parallel across candidates
)

res.ranking  # one row per candidate x duration, best first
res.power("chicago, portland", 15)  # the underlying augsynth_py.PowerResults
```

`res.ranking` carries `candidate`, `duration`, `n_units`, `mde`, `mde_grid`,
`power_at_target`, `empirical_size`, `rmspe_pre`, `null_bias`,
`h1_calibration_error`, `proportion_total_y`, `holdout`, `rank`, and
`investment` and `investment_mde` when `cpic` is given.

Two things worth knowing before you read the output:

- **`mde` is interpolated off the power curve**, not a grid point, because the
  grid value leaves most candidates tied. It is a geoexp construct and is not
  R GeoLift's `Average_MDE`. `mde_grid` carries the upstream grid point.
- **Rank order by `mde` is stable within an implementation, not across them.**
  Small differences near ties carry no information. See
  [`docs/methodology.md`](docs/methodology.md).
- **`null_bias` and `h1_calibration_error` are near-duplicates** on
  well-behaved panels; they separate only when the estimator's bias depends on
  the effect size.

## From a raw export to a design

```python
import geoexp_units  # optional companion: ready-made hierarchies
import numpy as np
from augsynth_py import AugSynth

from geoexp import aggregate_panel, prepare_panel, rank_designs

panel = prepare_panel(  # GeoDataRead-compatible, but says what it drops
    raw,
    unit="municipio",
    time="date",
    outcome="sales",
    date_format="%Y-%m-%d",
    every="1w",  # daily export -> whole weeks
)
panel = aggregate_panel(  # municipalities -> states
    panel,
    unit="municipio",
    time="date",
    outcome="sales",
    hierarchy=geoexp_units.br.ibge(),
    to="uf",
    use_labels=True,
)
res = rank_designs(
    panel,
    estimator=AugSynth(lambda_=1.0),
    unit="uf",
    time="date",
    outcome="sales",
    candidates=candidates,
    durations=[4, 8],
    lookback_window=10,
    cpic=12.0,
    budget=50_000,  # keep designs whose cost at their MDE fits
    rng=np.random.default_rng(7),
)
res.within_budget(80_000)  # try another budget without recomputing
res.plot_power("PR, SC")  # needs: pip install 'geoexp[plot]'
res.plot_fit("PR, SC", 8)  # actual vs synthetic, test window shaded
```

A design's cost is `investment_mde = cpic × mde_grid × Σ Y_treated` — exactly
R GeoLift's `Investment` — and it is an upper bound, so a design that fits the
budget can afford the lift it can detect. See
[`docs/methodology.md`](docs/methodology.md).

## Scope

0.2.x is single-cell design: candidate enumeration, power-based ranking,
budget-constrained selection, panel preparation, aggregation along unit
hierarchies, and design plots. Multi-cell designs are 0.3.x+.

`enumerate_candidates` is a convenience, not a gate — `rank_designs` takes any
list of sets, so bring your own candidates if you prefer.

## Not only marketing

The API assumes no geography: `rank_designs` takes a panel and the *name* of
its unit column, so the units can be cities, stores, schools, hospitals or
firms. "Market" here is vocabulary inherited from R GeoLift, whose reporting
shape this package mirrors, and `cpic` is arithmetically a cost per unit of
outcome — only its name is marketing's.

A public-policy pilot is the same question in different clothes: *which
municipalities should receive the program, and for how long, to detect a 5%
effect?* That is `rank_designs` unchanged.

What does not fit is the retrospective question — *did the policy that already
happened work?* — where you never chose the treated set. That is estimation
rather than design, so it belongs to
[augsynth-py](https://github.com/mrcsvg/augsynth-py) instead. Retrospective
policy evaluation is in fact where synthetic control started.

One caveat, stated rather than buried: the parity suite runs on
`GeoLift_PreTest`, a marketing panel, so every number quoted in these docs was
measured on marketing data. Nothing here measures a non-marketing panel.

## Validation

Every feature with a reproducible R `GeoLiftMarketSelection` counterpart is
asserted against it via rpy2 on the `GeoLift_PreTest` fixture. Parity is
established on the power curve at 240 cells (40 candidates x 6 effect sizes):
the two implementations agree with each other as well as either agrees with
itself, with no systematic bias. Features with no reproducible counterpart are
listed with their reason in
[`docs/geolift-oracle-characterization.md`](docs/geolift-oracle-characterization.md).

Clean-room: the R `GeoLift` package is a numerical oracle only, driven via
rpy2. Its source is never read for implementation and never translated.

## Development and AI use

This package was written by its maintainer working with
[Claude Code](https://claude.com/claude-code), Anthropic's coding agent. That
is disclosed here because it should shape how you read the rest of this page,
not as a novelty.

The working rule throughout was that nothing gets asserted that was not
measured. The parity figures above come from real runs against the R oracle and
are reproducible from `tests/validation_against_r/`; the noise floor they are
judged against was measured rather than assumed, after fixed thresholds
calibrated on a single draw turned out to flake. Where a quantity could not be
validated — GeoLift's `Average_MDE` and most of its ranking columns — that is
recorded as a limitation in
[`docs/geolift-oracle-characterization.md`](docs/geolift-oracle-characterization.md)
rather than papered over, and a cross-package benchmark that did not work is
written up there as a negative result instead of being quietly dropped. The
public API was designed and signed off before implementation, the
implementation was written test-first, and the clean-room rule stated above
held throughout: the R GeoLift source was never read.

## Install

```bash
pip install geoexp            # core
pip install 'geoexp[plot]'    # + plot_power / plot_fit (matplotlib)
pip install geoexp-units      # + ready-made hierarchies (IBGE first)
```

Requires `augsynth-py>=0.5.0` (the release that froze the power-API contract
this package is written against).

## License

MIT. No association with Meta's GeoLift is implied; the R package is used
strictly as a numerical validation oracle.
