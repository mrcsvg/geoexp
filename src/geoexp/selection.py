"""Market selection for single-cell geo-experiments.

Ranks candidate treated sets x experiment durations by what each design can
detect: :func:`enumerate_candidates` builds candidates by similarity to the
donor pool, and :func:`rank_designs` calls ``augsynth_py.simulate_power`` once
per candidate and reduces each power curve to a ranking row --- interpolated
MDE, power at a target effect, empirical size, pre-period fit, bias
diagnostics, and treated share of the outcome.

The public surface is fixed by ``docs/design-2026-09-05-v0.5-api.md``; what
each column means and where it comes from is in ``docs/methodology.md``.

Contract rules binding on this module: consume augsynth-py through its public
API only; one child generator per candidate via ``rng.spawn``; parallelize
across candidates with ``n_jobs=1`` passed down to ``simulate_power``.

Parity oracle: R ``GeoLiftMarketSelection``, driven via rpy2 in
``tests/validation_against_r/`` --- clean-room rule, never translated.
"""

from __future__ import annotations

import itertools
import warnings
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import polars as pl
from augsynth_py import (
    DEFAULT_EFFECT_SIZES,
    EffectType,
    PowerEstimator,
    PowerResults,
    simulate_power,
)
from joblib import Parallel, delayed

# augsynth_py exports EffectType but not these two: they live in
# augsynth_py.inference and are absent from augsynth_py.__all__. The
# consumption rule forbids reaching into a submodule, so they are mirrored
# here. An upstream proposal to export them is open; when it lands these
# become re-exports.
PermutationType = Literal["block", "iid"]
Side = Literal["two-sided", "left", "right"]


def _candidate_key(units: Iterable[Any]) -> str:
    """Canonical, stable key for a candidate treated set.

    Members are sorted by their string form and joined with ``", "``. This
    reproduces the convention R ``GeoLiftMarketSelection`` uses for
    ``BestMarkets$location``, so parity joins are a plain string match with no
    translation layer.

    Sorting on ``str(unit)`` rather than on the unit itself gives a total order
    for any unit type. ``augsynth_py.PowerParams.treated`` is typed
    ``tuple[Any, ...]``, so units are not guaranteed to be mutually orderable.

    Parameters
    ----------
    units : iterable of object
        The treated units making up one candidate.

    Returns
    -------
    str
        The canonical key, e.g. ``"nashville, san diego"``.
    """
    return ", ".join(sorted(str(u) for u in units))


def _candidate_keys(candidates: Sequence[Iterable[Any]]) -> list[str]:
    """Canonical keys for a candidate list, rejecting collisions.

    Parameters
    ----------
    candidates : sequence of iterable of object
        Candidate treated sets.

    Returns
    -------
    list of str
        One key per candidate, in the order given.

    Raises
    ------
    ValueError
        If two distinct candidates map to the same canonical key (e.g. ``{1}``
        and ``{"1"}``). Collapsing them would silently drop a row from the
        ranking frame.
    """
    keys = [_candidate_key(c) for c in candidates]
    seen: dict[str, int] = {}
    for i, key in enumerate(keys):
        if key in seen:
            raise ValueError(
                f"candidates {seen[key]} and {i} collide on the canonical key {key!r}; "
                "give them distinct units or normalize the unit type"
            )
        seen[key] = i
    return keys


def _interp_mde(
    effect_sizes: Sequence[float],
    powers: Sequence[float],
    *,
    target_power: float,
) -> float | None:
    """Minimum detectable effect, interpolated off the power curve.

    ``augsynth_py.PowerResults.mde`` returns a grid point, which leaves most
    candidates tied on a coarse grid and makes a ranking degenerate. This
    interpolates the crossing instead. It is a geoexp construct and is **not**
    R GeoLift's ``Average_MDE`` — see ``docs/methodology.md``.

    Rows at ``effect_size == 0`` are dropped: that row is the empirical size,
    not a detectable effect. The curve is monotonized by running maximum before
    interpolating, because simulated power curves dip.

    Parameters
    ----------
    effect_sizes : sequence of float
        Grid of effect sizes, ascending.
    powers : sequence of float
        Power at each effect size.
    target_power : float
        Power the design must reach.

    Returns
    -------
    float or None
        The interpolated crossing, or ``None`` if the curve never reaches
        ``target_power`` on the evaluated grid. Never extrapolated.
    """
    pairs = sorted((e, p) for e, p in zip(effect_sizes, powers, strict=True) if e > 0)
    if not pairs:
        return None
    xs = np.asarray([e for e, _ in pairs], dtype=float)
    ys = np.maximum.accumulate(np.asarray([p for _, p in pairs], dtype=float))
    if ys[-1] < target_power:
        return None
    i = int(np.argmax(ys >= target_power))
    if i == 0:
        return float(xs[0])
    x0, x1, y0, y1 = xs[i - 1], xs[i], ys[i - 1], ys[i]
    if y1 == y0:
        return float(x1)
    return float(x0 + (target_power - y0) * (x1 - x0) / (y1 - y0))


#: Substantive ranking criteria, in priority order. ``candidate`` is appended
#: for a deterministic display order but is deliberately not a rank key: two
#: designs identical on every criterion below share a rank.
_RANK_KEYS = ("mde", "empirical_size", "rmspe_pre")


def _apply_ranking(ranking: pl.DataFrame) -> pl.DataFrame:
    """Sort a ranking frame and attach a competition rank.

    Ordering is ``mde`` ascending, then ``empirical_size`` ascending, then
    ``rmspe_pre`` ascending, then ``candidate`` lexicographic. The reasoning:
    MDE answers the question the package exists to ask; empirical size guards
    against a design that "detects" because it over-rejects; ``rmspe_pre``
    guards against a poor pre-period fit; the candidate key makes the order
    fully deterministic so two runs never disagree on ties.

    A null ``mde`` (the curve never reached target power) sorts last — such a
    design is the worst available, not a missing value.

    Parameters
    ----------
    ranking : polars.DataFrame
        Must carry ``candidate`` and the columns in :data:`_RANK_KEYS`.

    Returns
    -------
    polars.DataFrame
        Sorted, with an integer ``rank`` column. Ties on every substantive
        criterion share a rank and consume the positions they span, matching
        the oracle's competition ranking.
    """
    ordered = ranking.sort([*_RANK_KEYS, "candidate"], nulls_last=True)
    return (
        ordered.with_row_index("_pos", offset=1)
        .with_columns(pl.col("_pos").min().over(_RANK_KEYS).cast(pl.Int64).alias("rank"))
        .drop("_pos")
    )


@dataclass(frozen=True)
class DesignRanking:
    """Result of :func:`rank_designs`.

    Attributes
    ----------
    ranking : polars.DataFrame
        One row per candidate x duration, sorted best-first, with a ``rank``
        column. See :func:`rank_designs` for the column meanings.
    candidates : dict of str to tuple
        Canonical key -> the original treated units, so the caller can recover
        the objects they passed in.
    """

    ranking: pl.DataFrame
    candidates: dict[str, tuple[Any, ...]]
    _results: dict[str, PowerResults] = field(repr=False, default_factory=dict)

    def power(self, candidate: str | Iterable[Any], duration: int) -> PowerResults:
        """Return the power results behind one candidate.

        Parameters
        ----------
        candidate : str or iterable of object
            Either the canonical key from the ranking frame or the original
            treated set.
        duration : int
            A duration that was evaluated. Validated, not used to subset: one
            ``simulate_power`` call covers all durations for a candidate, per
            the contract pattern of one child generator per *candidate*.

        Returns
        -------
        augsynth_py.PowerResults
            The candidate's results, spanning every evaluated duration.

        Raises
        ------
        KeyError
            If the candidate was not evaluated.
        ValueError
            If ``duration`` was not among the evaluated durations.
        """
        key = candidate if isinstance(candidate, str) else _candidate_key(candidate)
        if key not in self._results:
            raise KeyError(f"no results for candidate {key!r}")
        results = self._results[key]
        durations = results.params.durations if results.params is not None else ()
        if duration not in durations:
            raise ValueError(f"duration {duration} was not evaluated; got {list(durations)}")
        return results


def rank_designs(
    panel: pl.DataFrame,
    *,
    estimator: PowerEstimator,
    unit: str,
    time: str,
    outcome: str,
    candidates: Sequence[Iterable[Any]],
    durations: int | Sequence[int],
    rng: np.random.Generator,
    target_power: float = 0.8,
    target_effect_size: float = 0.10,
    cpic: float | None = None,
    n_jobs: int = 1,
    effect_sizes: Sequence[float] = DEFAULT_EFFECT_SIZES,
    effect_type: EffectType = "multiplicative",
    lookback_window: int = 1,
    alpha: float = 0.1,
    permutation_type: PermutationType = "block",
    block_size: int | None = None,
    side: Side = "two-sided",
    ns: int = 1000,
    on_error: Literal["raise", "record"] = "raise",
) -> DesignRanking:
    """Rank candidate geo-experiment designs by what each one can detect.

    Calls :func:`augsynth_py.simulate_power` once per candidate and reduces each
    power curve to a row: interpolated MDE, power at a target effect, empirical
    size, pre-period fit, and treated share of the outcome.

    Parameters
    ----------
    panel : polars.DataFrame
        Long-format panel, same contract as ``augsynth_py.simulate_power``.
    estimator : augsynth_py.PowerEstimator
        Prototype estimator. Freeze the penalty with ``AugSynth(lambda_=...)``
        for large candidate grids: market selection multiplies the
        per-simulation CV cost by the candidate count.
    candidates : sequence of iterable of object
        Explicit treated sets. Automatic enumeration is out of scope for this
        release.
    durations : int or sequence of int
        Treatment durations to evaluate; fanned out across every candidate.
    rng : numpy.random.Generator
        Required, with no default. One child generator per candidate via
        ``rng.spawn`` makes the result order-independent under candidate-level
        parallelism and restartable.
    target_power : float, default 0.8
        Power the MDE is defined against.
    target_effect_size : float, default 0.10
        Effect size reported in ``power_at_target``. Need not be on the grid.
    cpic : float, optional
        Cost per incremental conversion. When given, an ``investment`` column
        is added; when omitted, no cost is reported rather than assumed.
    n_jobs : int, default 1
        Parallelism **across candidates**. ``n_jobs=1`` is always passed down to
        ``simulate_power`` — joblib does not nest workers.

    Returns
    -------
    DesignRanking
        Ranking frame plus the underlying power results.

    Raises
    ------
    ValueError
        If two candidates collide on the canonical key, or ``candidates`` is
        empty.

    Notes
    -----
    ``mde`` is interpolated off the power curve and is a geoexp construct: it
    is **not** R GeoLift's ``Average_MDE``, which cannot be reproduced from
    published material. ``mde_grid`` carries the unmodified
    ``PowerResults.mde`` grid point alongside it.

    Rank order by ``mde`` is stable within an implementation but not across
    them: small differences near ties are not meaningful. See
    ``docs/geolift-oracle-characterization.md`` §9.
    """
    if not candidates:
        raise ValueError("candidates must not be empty")
    keys = _candidate_keys(candidates)
    members = [tuple(c) for c in candidates]
    duration_list = [durations] if isinstance(durations, int) else list(durations)

    if lookback_window < 10:
        warnings.warn(
            f"lookback_window={lookback_window} makes empirical_size a coarse rate over "
            f"{lookback_window} placebo test(s); it participates in the ranking order. "
            "Use a larger lookback_window for a meaningful false-positive rate.",
            UserWarning,
            stacklevel=2,
        )

    children = rng.spawn(len(keys))

    def _one(treated: tuple[Any, ...], child: np.random.Generator) -> PowerResults:
        return simulate_power(
            panel,
            estimator=estimator,
            unit=unit,
            time=time,
            outcome=outcome,
            treated=list(treated),
            durations=duration_list,
            effect_sizes=effect_sizes,
            effect_type=effect_type,
            lookback_window=lookback_window,
            alpha=alpha,
            permutation_type=permutation_type,
            block_size=block_size,
            side=side,
            ns=ns,
            rng=child,
            n_jobs=1,
            on_error=on_error,
        )

    computed: list[PowerResults] = Parallel(n_jobs=n_jobs)(
        delayed(_one)(m, c) for m, c in zip(members, children, strict=True)
    )
    results = dict(zip(keys, computed, strict=True))

    rows = [
        _summarize(
            key=key,
            treated=treated,
            results=results[key],
            duration=duration,
            panel=panel,
            unit=unit,
            time=time,
            outcome=outcome,
            target_power=target_power,
            target_effect_size=target_effect_size,
            alpha=alpha,
            effect_type=effect_type,
            cpic=cpic,
        )
        for key, treated in zip(keys, members, strict=True)
        for duration in duration_list
    ]
    return DesignRanking(
        ranking=_apply_ranking(pl.DataFrame(rows, schema=_ranking_schema(cpic))),
        candidates=dict(zip(keys, members, strict=True)),
        _results=results,
    )


def _ranking_schema(cpic: float | None) -> dict[str, Any]:
    """Column order and dtypes of the ranking frame."""
    schema: dict[str, Any] = {
        "candidate": pl.String,
        "duration": pl.Int64,
        "n_units": pl.Int64,
        "mde": pl.Float64,
        "mde_grid": pl.Float64,
        "power_at_target": pl.Float64,
        "empirical_size": pl.Float64,
        "rmspe_pre": pl.Float64,
        "null_bias": pl.Float64,
        "h1_calibration_error": pl.Float64,
        "proportion_total_y": pl.Float64,
        "holdout": pl.Float64,
    }
    if cpic is not None:
        schema["investment"] = pl.Float64
    return schema


def _power_at(curve: pl.DataFrame, effect_size: float) -> float | None:
    """Power at ``effect_size``, interpolated when it is off the grid."""
    exact = curve.filter(pl.col("effect_size") == effect_size)
    if exact.height:
        return float(exact.get_column("power").item())
    xs = curve.get_column("effect_size").to_numpy()
    ys = np.maximum.accumulate(curve.get_column("power").to_numpy())
    if effect_size < xs.min() or effect_size > xs.max():
        return None
    return float(np.interp(effect_size, xs, ys))


def _mean_or_none(frame: pl.DataFrame, column: str) -> float | None:
    """Mean of ``column``, or ``None`` when the frame is empty or all-null."""
    if not frame.height:
        return None
    value = frame.select(pl.col(column).mean()).item()
    return None if value is None else float(value)


def _calibration_error(
    sims: pl.DataFrame, grid_mde: float | None, effect_type: EffectType
) -> float | None:
    """How far the recovered effect sits from the injected one, at the MDE.

    Catches a design that detects with high power but recovers the wrong
    magnitude --- power alone cannot see that. Adapted from the
    ``h1_calibration_error`` diagnostic in mrcsvg/augsynth-py#22.

    Defined for multiplicative effects only: ``att_pct`` and a multiplicative
    effect size are the same quantity, while an additive effect size is on the
    outcome's scale and the comparison would be meaningless. Returns ``None``
    rather than a wrong number.
    """
    if effect_type != "multiplicative" or grid_mde is None:
        return None
    recovered = _mean_or_none(sims.filter(pl.col("effect_size") == grid_mde), "att_pct")
    return None if recovered is None else abs(recovered - grid_mde)


def _summarize(
    *,
    key: str,
    treated: tuple[Any, ...],
    results: PowerResults,
    duration: int,
    panel: pl.DataFrame,
    unit: str,
    time: str,
    outcome: str,
    target_power: float,
    target_effect_size: float,
    alpha: float,
    effect_type: EffectType,
    cpic: float | None,
) -> dict[str, Any]:
    """Reduce one candidate x duration cell to a ranking row."""
    curve = (
        results.power_curve(alpha=alpha).filter(pl.col("duration") == duration).sort("effect_size")
    )
    sims = results.simulations.filter(pl.col("duration") == duration)
    zero = curve.filter(pl.col("effect_size") == 0.0)

    # Treated share of the outcome over the treatment window (the last
    # `duration` periods). GeoLift reports a near-identical quantity under a
    # slightly different window convention; see the characterization doc.
    window = panel.filter(pl.col(time) > pl.col(time).max() - duration)
    totals = window.select(
        total=pl.col(outcome).sum(),
        treated=pl.col(outcome).filter(pl.col(unit).is_in(list(treated))).sum(),
    ).row(0)
    total, treated_y = float(totals[0] or 0.0), float(totals[1] or 0.0)
    share = treated_y / total if total else None

    grid_mde = results.mde(target_power=target_power, alpha=alpha, duration=duration)
    row: dict[str, Any] = {
        "candidate": key,
        "duration": duration,
        "n_units": len(treated),
        "mde": _interp_mde(
            curve.get_column("effect_size").to_list(),
            curve.get_column("power").to_list(),
            target_power=target_power,
        ),
        "mde_grid": grid_mde,
        "power_at_target": _power_at(curve, target_effect_size),
        "empirical_size": float(zero.get_column("power").item()) if zero.height else None,
        "rmspe_pre": _mean_or_none(sims, "rmspe_pre"),
        "null_bias": _mean_or_none(sims.filter(pl.col("effect_size") == 0.0), "att_pct"),
        "h1_calibration_error": _calibration_error(sims, grid_mde, effect_type),
        "proportion_total_y": share,
        "holdout": None if share is None else 1.0 - share,
    }
    if cpic is not None:
        row["investment"] = cpic * target_effect_size * treated_y
    return row


def _rank_by_similarity(
    panel: pl.DataFrame, *, unit: str, time: str, outcome: str, units: Sequence[Any]
) -> list[Any]:
    """Order units by how well the rest of the panel explains them.

    Each unit's series is correlated with the mean of every *other* unit's
    series; higher correlation first. A test market that moves with the donor
    pool is one a synthetic control can reconstruct, which is the published
    GeoLift guidance ("test and control areas should resemble each other
    historically"). Correlation is GeoLift's own documented default criterion
    (``dtw_emphasis = 0``).

    Ties and non-finite correlations fall back to the canonical key, so the
    order is total and reproducible.
    """
    wide = panel.pivot(on=unit, index=time, values=outcome).sort(time)
    matrix = np.column_stack([wide.get_column(str(u)).to_numpy() for u in units]).astype(float)
    scores: dict[Any, float] = {}
    for i, u in enumerate(units):
        others = np.delete(matrix, i, axis=1)
        if others.shape[1] == 0:
            scores[u] = 0.0
            continue
        pool = others.mean(axis=1)
        series = matrix[:, i]
        if series.std() == 0 or pool.std() == 0:
            scores[u] = -np.inf
            continue
        corr = float(np.corrcoef(series, pool)[0, 1])
        scores[u] = corr if np.isfinite(corr) else -np.inf
    return sorted(units, key=lambda u: (-scores[u], _candidate_key([u])))


def enumerate_candidates(
    panel: pl.DataFrame,
    *,
    unit: str,
    time: str,
    outcome: str,
    sizes: Sequence[int],
    include: Iterable[Any] = (),
    exclude: Iterable[Any] = (),
    max_per_size: int = 50,
    rng: np.random.Generator | None = None,
) -> list[frozenset[Any]]:
    """Build candidate treated sets for :func:`rank_designs`.

    The combination space is large --- C(40, 3) is 9880 --- so candidates are
    drawn from units the donor pool explains well, ranked by
    :func:`_rank_by_similarity`, and capped per size.

    This is a geoexp construct. It is **not** a reimplementation of R GeoLift's
    selection step, which uses its own (unpublished) combination of correlation
    and dynamic time warping; no parity test compares the two candidate lists.
    Its output is a plain list of sets, so a caller who wants different
    candidates can build them any way they like and pass them straight to
    ``rank_designs``.

    Parameters
    ----------
    panel : polars.DataFrame
        Long-format panel.
    unit, time, outcome : str
        Column names.
    sizes : sequence of int
        Treated-set sizes to generate, e.g. ``[1, 2, 3]``.
    include : iterable of object, optional
        Units that must appear in every candidate.
    exclude : iterable of object, optional
        Units that must never be treated. They stay in the donor pool.
    max_per_size : int, default 50
        Cap on candidates generated per size.
    rng : numpy.random.Generator, optional
        When given, candidates are sampled at random from the eligible pool
        instead of taken in similarity order. Sampling is reproducible for a
        given generator. Random sampling risks interpolation bias unless the
        units are already similar; prefer the deterministic default until you
        have checked that.

    Returns
    -------
    list of frozenset
        Candidate treated sets, ready to pass as ``candidates``.

    Raises
    ------
    ValueError
        If a size is not positive, exceeds the eligible pool, or is smaller
        than ``include``; or if ``include``/``exclude`` name unknown units.
    """
    all_units = list(panel.get_column(unit).unique().sort())
    required = list(dict.fromkeys(include))
    excluded = set(map(_candidate_key, ([e] for e in exclude)))
    for name, given in (("include", required), ("exclude", list(exclude))):
        unknown = [u for u in given if u not in all_units]
        if unknown:
            raise ValueError(f"{name} names units absent from the panel: {unknown}")

    eligible = [u for u in all_units if _candidate_key([u]) not in excluded and u not in required]
    ranked = _rank_by_similarity(panel, unit=unit, time=time, outcome=outcome, units=eligible)

    out: list[frozenset[Any]] = []
    for size in sizes:
        if size <= 0:
            raise ValueError(f"sizes must be positive, got {size}")
        free = size - len(required)
        if free < 0:
            raise ValueError(f"size {size} is smaller than include ({len(required)} units)")
        if free > len(ranked):
            raise ValueError(
                f"size {size} exceeds the eligible pool ({len(ranked)} units after include/exclude)"
            )
        out.extend(_combinations(ranked, required, free, max_per_size, rng))
    return out


def _combinations(
    ranked: Sequence[Any],
    required: Sequence[Any],
    free: int,
    max_per_size: int,
    rng: np.random.Generator | None,
) -> list[frozenset[Any]]:
    """Pick up to ``max_per_size`` combinations of ``free`` units from ``ranked``."""
    if rng is None:
        picked = itertools.islice(itertools.combinations(ranked, free), max_per_size)
        return [frozenset([*required, *combo]) for combo in picked]

    seen: set[tuple[Any, ...]] = set()
    for _ in range(max_per_size * 50):
        if len(seen) >= max_per_size:
            break
        idx = rng.choice(len(ranked), size=free, replace=False)
        seen.add(tuple(sorted((ranked[int(i)] for i in idx), key=lambda u: _candidate_key([u]))))
    ordered = sorted(seen, key=lambda combo: _candidate_key(combo))
    return [frozenset([*required, *combo]) for combo in ordered]
