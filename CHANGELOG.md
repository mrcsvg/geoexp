# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
(with the usual 0.x caveat: minor versions may break the API).

## [Unreleased]

## [0.2.0] - 2026-10-05

Project milestone v0.5+ (release line 0.2.x), per
`docs/design-2026-09-25-0.2-api.md`.

### Added

- **Budget-constrained selection.** `rank_designs(..., budget=)` keeps the
  designs whose cost at their own MDE fits, and re-ranks them. The cost is the
  new `investment_mde` column, `cpic × mde_grid × Σ Y_treated` — exactly R
  GeoLift's `Investment`, asserted in `tests/validation_against_r/test_budget.py`
  (27 of 30 candidates share a grid MDE with the oracle; on those, cost agrees
  to 1e-9 and the kept set is identical). `DesignRanking.within_budget()`
  re-applies another budget, tighter or looser, without recomputing.
- **`prepare_panel`**: from a raw export to a balanced long panel. Reproduces
  R `GeoDataRead` (lower-cased names, duplicates summed, incomplete units
  dropped — frame equality in `tests/validation_against_r/test_panel.py`), but
  warns with each dropped unit and the reason, keeps real dates by default,
  and adds temporal aggregation (`every=`) and `on_incomplete="fill_zero"`.
- **`Hierarchy` and `aggregate_panel`**: roll a panel up a unit hierarchy
  (municipality → state, store → district). Non-nesting hierarchies are
  allowed; an aggregation direction that would double-count is refused.
  Ready-made hierarchies ship in the new companion package `geoexp-units`.
- **Design plots**, behind the new optional extra `geoexp[plot]` (matplotlib,
  imported lazily): `DesignRanking.plot_power()` and `plot_fit()`, plus
  `fit_path()`, which returns the numbers behind `plot_fit` without matplotlib.
- `HierarchyError` in `geoexp.exceptions`.

### Fixed

- **`investment` was priced at the last test window only.** GeoLift averages
  the treated outcome over the lookback windows; the two agree only at
  `lookback_window = 1`, which is the only setting 0.1.0 had verified. Both
  `investment` and the new `investment_mde` now average over the lookback
  windows. Expect different `investment` values for `lookback_window > 1`.
- `rank_designs` assumed an integer time column when locating the treatment
  window (`time > max(time) - duration`); it now counts the last `duration`
  distinct periods, so date columns work.

### Changed

- Docs: the README and `docs/methodology.md` now state that `unit` is any
  panel unit and that "market" and `cpic` are vocabulary inherited from R
  GeoLift, with the prospective/retrospective boundary against augsynth-py
  spelled out.
- `docs/geolift-oracle-characterization.md`: new §10 (`GeoDataRead`) and §11
  (budget) measurements; §2 no longer says geoexp must not reimplement
  `GeoDataRead` (that rule protects the parity harness, which is unchanged);
  §5 amended for lookback-averaged `Investment`; new exclusions listed in §6.
  Records that the oracle is not deterministic between identical calls, and a
  partial, unestablished observation about `Average_MDE`.

### How Claude was used

This release was built in one session with Claude Code (Anthropic's coding
agent, model Claude Opus 5.5), with the maintainer deciding and Claude
proposing, measuring and writing.

- **Decided by the maintainer**, each asked explicitly and recorded in the
  design doc: the scope of the release; matplotlib as an *optional* extra;
  pricing the budget at `mde_grid`; dropping over-budget designs *and* offering
  `within_budget`; dropping incomplete units with a warning; shipping
  hierarchies as a separate package, and its name `geoexp-units`. The
  maintainer approved the design section by section before any code was
  written, and keeps commits, pull requests and releases to themselves.
- **Measured by Claude**, as a black box, against R GeoLift 2.7.5 via Rscript
  and rpy2: the semantics of `budget`, the behaviour of `GeoDataRead`, the
  oracle's run-to-run non-determinism, and the lookback-averaged
  `Investment` — which exposed the 0.1.0 pricing bug above. The GeoLift source
  was not read.
- **Written by Claude**: the code, the tests (test-first, each seen failing
  before its implementation), the parity tests, the docs and this entry. Along
  the way the declared dependency floors were exercised in a throwaway
  environment, which caught a Polars 1.0 incompatibility before CI would have.
- **What that does not guarantee**: the parity figures are from single runs
  on one marketing fixture; the plots were checked by eye on synthetic data
  only.

## [0.1.0] - 2026-09-20

### Added

- **Public API for single-cell market selection** (project milestone v0.5),
  per `docs/design-2026-09-05-v0.5-api.md`: `rank_designs()` ranks explicit
  candidate treated sets x durations by what each design can detect, and
  `DesignRanking` carries the ranking frame plus the underlying
  `augsynth_py.PowerResults`. Consumes augsynth-py through its public API only,
  with one child generator per candidate via `rng.spawn` and candidate-level
  parallelism.
- `mde` interpolated off the power curve, with the upstream grid point kept
  alongside as `mde_grid`. The grid value takes only four distinct values
  across the 39 fixture candidates, which makes it useless as a sort key; see
  `docs/methodology.md`.
- `investment` column when `cpic` is supplied — exactly parity-testable.
- `enumerate_candidates()`, pulled forward from 0.2.x: ranks units by
  correlation with the rest of the panel and draws capped combinations of the
  requested sizes, with `include`/`exclude` and optional reproducible random
  sampling. A separate function, not a `rank_designs` parameter, so bringing
  your own candidates stays the primary path.
- `null_bias` and `h1_calibration_error` diagnostics, adapted from
  mrcsvg/augsynth-py#22. `null_bias` is the honest analog of GeoLift's
  `abs_lift_in_zero`; `h1_calibration_error` catches a design that detects at
  the wrong magnitude. The two overlap on well-behaved panels — see
  `docs/methodology.md`.
- Parity suite against R `GeoLiftMarketSelection` on the power curve at 240
  cells, with self-calibrating tolerances derived from the measured noise
  floor rather than from a single draw.
- `docs/geolift-oracle-characterization.md`: measured input/output surface of
  the R oracle, which of its columns cannot be reproduced under the clean-room
  rule, and the rejected cross-package benchmark experiment.

### Changed

- `CLAUDE.md` validation rule now requires parity only for features with a
  reproducible oracle counterpart; features without one must be listed with a
  reason and covered by unit tests plus a methodology entry.
- The `validation` extra no longer pulls `pandas` or `pyarrow`: the R bridge
  converts to Polars directly (adapted from mrcsvg/augsynth-py#22).

### Added (skeleton)

- Repository skeleton per `docs/geoexp-bootstrap-spec.md`: src layout with
  `py.typed`, tooling ported from augsynth-py (ruff, mypy strict, pytest),
  CI workflows (lint/type/unit + min-deps, GeoLift validation, tag-triggered
  PyPI trusted publishing), and placeholder `selection` module. No public
  API yet — that lands with the v0.5 design doc.

[Unreleased]: https://github.com/mrcsvg/geoexp/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/mrcsvg/geoexp/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/mrcsvg/geoexp/releases/tag/v0.1.0
