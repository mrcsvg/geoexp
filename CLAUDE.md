# geoexp

Geo-experiment design on top of `augsynth-py`: market selection, power-based
design ranking, and experiment reporting. This file is the entry point for AI
coding assistants working on this repo. Read it before starting any task.

---

## Project goal

Answer *which design should I run?* for single-cell (and later multi-cell)
geo-experiments: rank candidate treated sets × experiment durations by what
each design can detect, with GeoLift-shaped reporting.

The **authoritative statement of the package boundary** lives in the
package-boundary section of
[augsynth-py's CLAUDE.md](https://github.com/mrcsvg/augsynth-py/blob/main/CLAUDE.md):
*"given a design"* (augsynth-py: estimators, inference, power/MDE) vs.
*"choose a design"* (geoexp: market selection, budget/ROI, multi-cell,
reporting). Do not add estimators, inference, or power simulation here.

## Consumption rule (the one rule augsynth-py doesn't need)

geoexp consumes augsynth-py **through its public API only**:

- Never import an underscore-prefixed name from `augsynth_py`.
- If market selection appears to need something the public API cannot
  express, the fix is a proposal against augsynth-py's API — never a
  private-attribute reach-in, never a fork of the logic.
- Contract patterns (documented in `augsynth_py.simulate_power`'s Notes):
  one child generator per candidate via `rng.spawn(n)`; parallelize across
  candidates with `n_jobs=1` passed down (joblib does not nest workers);
  recommend `AugSynth(lambda_=...)` in docs for large grids.

## Implementation strategy: clean-room from published material

Implemented from published material (Abadie 2021 §5 in-time placebos; the
GeoLift methodology publications; CWZ 2021 via augsynth-py). The R `GeoLift`
package is a **parity oracle only**, driven via `rpy2` in
`tests/validation_against_r/` — its source is never translated and never
copied into this repo. MIT license; no association with Meta's GeoLift is
implied (the working name `geolift-py` was dropped for exactly that reason).

## Validation rule (non-negotiable)

Every public ranking/selection feature **that has a reproducible
`GeoLiftMarketSelection` counterpart** must have at least one test in
`tests/validation_against_r/` that runs the same input through this package
and through R `GeoLiftMarketSelection` via rpy2, asserting numerical
agreement within a documented tolerance (default `atol=1e-6, rtol=1e-5`;
relax with justification only), on the `GeoLift_PreTest` fixture.

A feature with no reproducible counterpart is **not exempt by default**: it
must be listed in `docs/geolift-oracle-characterization.md` §6 with the
reason it cannot be compared, and covered by unit tests plus a
`docs/methodology.md` entry stating what it is and what it is not. Adding an
entry to that list is a maintainer decision, not a contributor's.

PRs that add a public feature without either a parity test or a recorded
exclusion should be rejected.

## Version map

| Project milestone | geoexp release | Content |
|---|---|---|
| v0.5 | 0.1.x | Single-cell market selection, candidate enumeration, cpic |
| v0.5+ | 0.2.x | Budget-constrained selection |
| v0.6 | 0.3.x+ | Multi-cell designs |

augsynth-py's own releases number independently; geoexp's dependency floor
is `augsynth-py>=0.5.0` (the release that froze the power-API contract —
see `docs/power-api-contract-review.md` in that repo).

## Repository layout

```
geoexp/
├── CLAUDE.md                       <- you are here
├── README.md
├── CHANGELOG.md                    <- Keep a Changelog format
├── LICENSE                         <- MIT
├── pyproject.toml                  <- hatchling, PEP 621, src layout
├── src/geoexp/
│   ├── __init__.py                 <- public exports
│   ├── _version.py                 <- single source of truth for the version
│   ├── py.typed                    <- PEP 561 marker
│   ├── exceptions.py               <- domain exceptions
│   └── selection.py                <- v0.5 module (split into a package
│                                      only when a second module earns it)
├── tests/
│   ├── conftest.py
│   ├── unit/                       <- no R required
│   └── validation_against_r/       <- GeoLift parity via rpy2
├── docs/
│   ├── methodology.md              <- maps code to Abadie 2021 §5 / GeoLift docs
│   ├── releasing.md                <- PyPI release runbook
│   └── geoexp-bootstrap-spec.md    <- the bootstrap spec (relocated here)
└── .github/workflows/
    ├── ci.yml                      <- lint, mypy, unit tests, min-deps job
    ├── validation.yml              <- R + GeoLift parity
    └── release.yml                 <- tag-triggered PyPI trusted publishing
```

## Code and testing conventions

Identical to augsynth-py's — type hints everywhere in `src/` with
`mypy --strict`; NumPy docstrings citing paper + section for known methods;
no `print()` in library code; no global mutable state; every stochastic
function takes `rng: np.random.Generator`; `ValueError` for bad input and
`geoexp.exceptions` subclasses for domain errors; `tests/unit/` runs without
R (`pytest.importorskip("rpy2")` guards the validation suite); `ruff` for
lint + format. When in doubt, match augsynth-py.

## Out of scope (inherited wholesale from augsynth-py)

Estimators/inference/power (import them), Bayesian variants, GPU
acceleration, Rust/C extensions.

## Things to confirm with the maintainer before doing

- Adding a dependency.
- Changing the public API of anything exported from `__init__.py`.
- Relaxing a validation tolerance.
- Starting multi-cell work (project v0.6).
