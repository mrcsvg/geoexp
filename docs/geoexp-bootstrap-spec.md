# `geoexp` bootstrap specification

> **Historical.** This is the bootstrap spec, written before the package
> existed. It is kept for the record of why the repository is shaped the way
> it is. The public API it sketches in §4 was superseded by
> `docs/design-2026-09-05-v0.5-api.md`, and the scope in §7 by the version map
> in `CLAUDE.md` — enumeration and `cpic` landed in 0.1.x, not 0.2.x. Where
> this document and those disagree, they win.

**Date**: 2026-08-28, relocated here at repo creation (2026-08-30).
**Status**: accepted and being executed — this repository *is* the skeleton
of §3, and the maintainer accepted the recommended (full) bundle of the
power-contract review, so the dependency floor in §5 resolves to
**augsynth-py 0.5.0**.

Companion document: `docs/power-api-contract-review.md` **in the
augsynth-py repository** — the pre-freeze review of `augsynth_py.power`,
whose accepted bundle determined the dependency floor in §5.

## 1. Identity

- **Name**: `geoexp` — settled 2026-08-12 (see `CLAUDE.md`, package-boundary
  section, including why the `geo`+word namespace is a minefield and why
  this name survived). PyPI name free at the time of the decision;
  re-verify at registration time.
- **Repository**: `mrcsvg/geoexp`. **License**: MIT.
- **One-line description**: *Geo-experiment design on top of augsynth-py:
  market selection, power-based design ranking, and experiment reporting.*
  No GeoLift association is implied anywhere — the working name `geolift-py`
  was dropped for exactly that reason.
- **Clean-room rule carries over verbatim**: implemented from published
  material (Abadie 2021 §5; the GeoLift methodology papers/blog posts; the
  CWZ 2021 inference already wrapped by augsynth-py). The R `GeoLift`
  package is a **parity oracle only**, driven via `rpy2` in
  `tests/validation_against_r/`; its source is never translated. Do not copy
  R code into the repo.

## 2. Scope of the first release

The line from `CLAUDE.md` is *"given a design" (augsynth-py) vs. "choose a
design" (geoexp)*; `geoexp` consumes augsynth-py **through the public API
only**.

**geoexp 0.1.0 = project milestone v0.5 — single-cell market selection:**

- Rank candidate treated sets × experiment durations by what each design can
  detect: per candidate, call `augsynth_py.simulate_power` and read power
  curve, empirical size, MDE, and pre-period fit quality.
- Candidates supplied explicitly by the caller in 0.1.0 (a list of unit
  sets). Automatic candidate enumeration (GeoLift's
  `include_markets`/`exclude_markets`/ combinatorial search) is 0.2.0
  material — enumeration strategy deserves its own design doc.
- Parity oracle: `GeoLiftMarketSelection` (`model="none"` first, mirroring
  how the power parity test isolates the harness from the estimator).
- Budget / cost-per-incremental-conversion / investment outputs
  (`GeoLiftMarketSelection`'s `cpic` path): **stretch, not blocking** —
  include in 0.1.0 only if it falls out as a thin post-processing of the
  ranking frame; otherwise 0.2.0.

**Not in geoexp 0.1.0** (and mostly not ever):

- Multi-cell designs — project milestone v0.6+.
- Estimators, inference, or power simulation — those live in augsynth-py;
  `geoexp` imports them. If market selection appears to need something the
  augsynth-py public API cannot express, the fix is a proposal against
  augsynth-py's API, never a private-attribute reach-in and never a fork of
  the logic.
- Bayesian variants, GPU, Rust/C extensions — the augsynth-py exclusions
  are inherited wholesale.

## 3. Repository skeleton

Mirror augsynth-py's layout — same build backend, same conventions, so a
contributor moves between the repos without relearning anything:

```
geoexp/
├── CLAUDE.md                       <- §7 below
├── README.md
├── CHANGELOG.md                    <- Keep a Changelog
├── LICENSE                         <- MIT
├── pyproject.toml                  <- hatchling, PEP 621, src layout
├── src/geoexp/
│   ├── __init__.py                 <- public exports
│   ├── _version.py                 <- single source of truth
│   ├── py.typed
│   ├── exceptions.py
│   └── selection.py                <- v0.5 module (split into a package
│                                      only when a second module earns it)
├── tests/
│   ├── conftest.py
│   ├── unit/                       <- no R required
│   └── validation_against_r/       <- GeoLift parity via rpy2
├── docs/
│   ├── methodology.md              <- maps code to Abadie 2021 §5 / GeoLift docs
│   ├── releasing.md                <- copy of augsynth-py's, names swapped
│   └── geoexp-bootstrap-spec.md    <- this file, relocated
├── notebooks/                      <- not shipped
└── .github/workflows/
    ├── ci.yml                      <- lint, mypy, unit tests, min-deps job
    ├── validation.yml              <- R + GeoLift parity
    └── release.yml                 <- tag-triggered PyPI trusted publishing
```

## 4. Preliminary public surface (to be fixed in a design doc, not here)

Sketch only — enough to shape the skeleton; the real API design is the first
work item after bootstrap and follows augsynth-py's design-doc habit:

```python
from geoexp import rank_designs, DesignRanking

res = rank_designs(
    panel,  # same long-format contract as augsynth-py
    unit=...,
    time=...,
    outcome=...,
    candidates=[{"NY"}, {"NY", "SF"}, ...],  # explicit treated sets (0.1.0)
    durations=[15, 30],
    estimator=AugSynth(lambda_=...),  # prototype, deep-copied per fit
    effect_sizes=...,
    alpha=0.1,  # forwarded to simulate_power
    rng=np.random.default_rng(7),
    n_jobs=-1,  # candidate-level parallelism
)
res.ranking  # pl.DataFrame: candidate, duration, mde, size, rank, ...
res.power(candidate, duration)  # -> augsynth_py.PowerResults for one cell
```

Contract-usage rules (from the power review, binding on `geoexp` code):

- One child generator per candidate via `rng.spawn(len(candidates))` —
  order-independent reproducibility under candidate-level parallelism.
- Parallelize across candidates; pass `n_jobs=1` down to `simulate_power`
  (joblib does not nest workers).
- Surface augsynth-py's guidance about freezing CV: recommend
  `AugSynth(lambda_=...)` for large grids in `geoexp` docs, since market
  selection multiplies the per-simulation CV cost by the candidate count.
- Nothing in `geoexp` may import an underscore-prefixed name from
  `augsynth_py`.

## 5. `pyproject.toml` decisions

- **Build**: `hatchling`, `dynamic = ["version"]` from `src/geoexp/_version.py`.
- **Python**: `>=3.11` (same as augsynth-py).
- **Dependencies** — only what `geoexp` imports directly, floors low,
  no caps (augsynth-py's compatibility policy applies verbatim):
  - `augsynth-py>=X` where **X = the release carrying the accepted bundle**
    of the power-contract review (0.4.1 or 0.5.0 — see that document's
    sequencing table). Do not start `geoexp` against 0.4.0 and absorb the
    contract change mid-flight.
  - `polars>=1.0`, `numpy>=1.26`, `joblib>=1.3`.
  - **No direct `cvxpy`/`scipy`** — they arrive transitively and `geoexp`
    never calls them.
- **Extras**: `dev` (ruff, mypy, pytest) and `validation` (rpy2) mirroring
  augsynth-py. No `numpy1` extra of its own: environments that need the
  numpy 1.x ABI install `augsynth-py[numpy1]` alongside; document that
  recipe in a `docs/compatibility.md` stub instead of duplicating pins.

## 6. CI and release engineering

Copy augsynth-py's three workflows with names swapped; the deltas that
matter:

- **ci.yml**: lint (`ruff check`, `ruff format --check`), `mypy --strict`
  on `src/`, `pytest tests/unit/`, plus the `unit-tests-min-deps` job
  pinned to the declared floors — including `augsynth-py==X`, so the floor
  is exercised, not aspirational.
- **validation.yml**: R 4.5 via `r-lib/actions/setup-r`, install `remotes`
  then `remotes::install_github("facebookincubator/GeoLift", upgrade="never")`
  with `GITHUB_PAT: ${{ secrets.GITHUB_TOKEN }}` (augsynth-py's workflow
  documents why: anonymous GitHub API rate limits). `geoexp` does not need
  the R `augsynth`/`Synth` installs unless a test compares estimator
  internals — it should not.
- **release.yml**: identical trusted-publishing pipeline — lint+unit gate,
  tag ↔ `_version.py` agreement check, `python -m build`, `twine check
  --strict`, publish gated on the `pypi` environment.

**One-time setup checklist** (mirrors `docs/releasing.md` here):

1. Create `mrcsvg/geoexp` (private is fine during bootstrap; the 0.3.0-era
   public-repo prep checklist here shows what to sweep before flipping).
2. Re-verify the `geoexp` name on PyPI, then register the **pending
   publisher**: project `geoexp`, owner `mrcsvg`, repo `geoexp`, workflow
   `release.yml`, environment `pypi`.
3. Create the `pypi` GitHub environment (required reviewer optional but
   recommended — it made the v0.4.0 publish a deliberate click).
4. First release: `0.1.0`, tagged `v0.1.0`, once the parity test passes.

## 7. `CLAUDE.md` for the new repo

Same skeleton as this repo's, with these sections rewritten:

- **Project goal**: choose-a-design layer; explicit pointer to the
  package-boundary section of augsynth-py's `CLAUDE.md` as the authoritative
  statement of what belongs on which side.
- **Canonical references**: Abadie 2021 (§5 in-time placebos); the GeoLift
  methodology publications; CWZ 2021 via augsynth-py. Same "cite paper and
  section in docstrings" rule.
- **Validation rule**: every public ranking/selection feature gets a parity
  test against `GeoLiftMarketSelection` via rpy2 before it ships — same
  non-negotiable framing, same default tolerances, same
  `GeoLift_PreTest` fixture.
- **Consumption rule** (new, the one rule this repo doesn't need): public
  augsynth-py API only; API gaps become upstream proposals, not reach-ins.
- **Version map** (avoids the v0.5/0.5.0 trap): project milestone v0.5 →
  `geoexp 0.1.x`; v0.6 multi-cell → `geoexp 0.3.x`+; augsynth-py's own
  releases number independently.

## 8. Bootstrap sequence

1. Land the accepted power-contract bundle in augsynth-py; release it
   (0.4.1 or 0.5.0 per the review's table).
2. Create the repo from §3; port tooling configs verbatim from augsynth-py
   (`ruff`/`mypy`/`pytest` sections of `pyproject.toml`).
3. PyPI pending publisher + `pypi` environment (§6 checklist).
4. Write the v0.5 API design doc (the real version of §4), get sign-off.
5. Implement `selection.py` with unit tests; then the
   `GeoLiftMarketSelection` parity test; then README + methodology docs;
   then `v0.1.0`.
