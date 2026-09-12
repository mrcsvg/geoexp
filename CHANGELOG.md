# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
(with the usual 0.x caveat: minor versions may break the API).

## [Unreleased]

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
