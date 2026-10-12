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

import copy
import itertools
import math
import warnings
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Literal

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

from geoexp import _plotting

if TYPE_CHECKING:
    from matplotlib.axes import Axes

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
    powers: Sequence[float | None],
    *,
    target_power: float,
) -> float | None:
    """Minimum detectable effect, interpolated off the power curve.

    ``augsynth_py.PowerResults.mde`` returns a grid point, which leaves most
    candidates tied on a coarse grid and makes a ranking degenerate. This
    interpolates the crossing instead. It is a geoexp construct and is **not**
    R GeoLift's ``Average_MDE`` — see ``docs/methodology.md``.

    Rows at ``effect_size == 0`` are dropped: that row is the empirical size,
    not a detectable effect. Rows whose power is ``None`` --- every simulation
    in the cell failed under ``on_error="record"`` --- are dropped too: they
    carry no evidence, and reading them as a number would turn an unknown
    cell into the smallest grid effect. ``PowerResults.mde`` skips them the
    same way. The curve is monotonized by running maximum before
    interpolating, because simulated power curves dip.

    Parameters
    ----------
    effect_sizes : sequence of float
        Grid of effect sizes, ascending.
    powers : sequence of float or None
        Power at each effect size; ``None`` where it is unknown.
    target_power : float
        Power the design must reach.

    Returns
    -------
    float or None
        The interpolated crossing, or ``None`` if the curve never reaches
        ``target_power`` on the evaluated grid. Never extrapolated.
    """
    pairs = sorted(
        (e, p) for e, p in zip(effect_sizes, powers, strict=True) if e > 0 and p is not None
    )
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


def _apply_ranking(ranking: pl.DataFrame, *, by: Sequence[str] = ()) -> pl.DataFrame:
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
    by : sequence of str, optional
        Group columns; the rank restarts at 1 inside each group. The
        stability bootstrap uses it to rank every draw with this same rule.

    Returns
    -------
    polars.DataFrame
        Sorted, with an integer ``rank`` column. Ties on every substantive
        criterion share a rank and consume the positions they span, matching
        the oracle's competition ranking.
    """
    group = list(by)
    ordered = ranking.sort([*group, *_RANK_KEYS, "candidate"], nulls_last=True)
    positioned = ordered.with_row_index("_pos", offset=1)
    if group:
        positioned = positioned.with_columns(pl.col("_pos") - pl.col("_pos").min().over(group) + 1)
    return positioned.with_columns(
        pl.col("_pos").min().over([*group, *_RANK_KEYS]).cast(pl.Int64).alias("rank")
    ).drop("_pos")


@dataclass(frozen=True)
class _Context:
    """What :func:`rank_designs` was called with, kept for the plot methods."""

    panel: pl.DataFrame
    estimator: PowerEstimator
    unit: str
    time: str
    outcome: str
    target_power: float
    alpha: float


@dataclass(frozen=True)
class RankStability:
    """How much a ranking depends on the placebo tests that produced it.

    Returned by :meth:`DesignRanking.stability`. One paired bootstrap over the
    simulated ``(replicate, window)`` evidence, re-ranked per draw with the
    production rule. See ``docs/methodology.md`` for what it measures and
    what it does not: the intervals are optimistic, because consecutive
    placebo windows overlap.

    Attributes
    ----------
    frame : polars.DataFrame
        One row per row of the ranking it was built from, same order:
        ``candidate``, ``duration``, ``rank`` (the point ranking),
        ``rank_low`` / ``rank_high`` (quantiles of the rank across draws at
        ``level``), ``p_top`` (share of draws ranked ``<= top``),
        ``p_undetectable`` (share of draws where the curve never reached
        target power), ``mde_low`` / ``mde_high`` (quantiles of the
        interpolated MDE over the draws where it exists; null when more than
        ``1 - level`` of the draws have none). Both intervals use the
        nearest-rank quantile, so every endpoint is a value some draw actually
        produced; nothing is interpolated between draws.
    n_draws : int
        Bootstrap draws taken.
    level : float
        Coverage of the rank and MDE intervals.
    top : int
        The position ``p_top`` is measured against.
    pool_size : int
        ``replicates x lookback_window`` --- the evidence resampled per draw.
    """

    frame: pl.DataFrame
    n_draws: int
    level: float
    top: int
    pool_size: int


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
    budget: float | None = None
    _results: dict[str, PowerResults] = field(repr=False, default_factory=dict)
    _unfiltered: pl.DataFrame | None = field(repr=False, default=None)
    _context: _Context | None = field(repr=False, default=None)

    def _key(self, candidate: str | Iterable[Any]) -> str:
        return candidate if isinstance(candidate, str) else _candidate_key(candidate)

    def _row(self, key: str, duration: int) -> dict[str, Any] | None:
        full = self.ranking if self._unfiltered is None else self._unfiltered
        rows = full.filter((pl.col("candidate") == key) & (pl.col("duration") == duration))
        return rows.row(0, named=True) if rows.height else None

    def _require_context(self) -> _Context:
        if self._context is None:
            raise ValueError(
                "this DesignRanking was not built by rank_designs; it carries no simulation results"
            )
        return self._context

    def fit_path(self, candidate: str | Iterable[Any], duration: int) -> pl.DataFrame:
        """Actual vs synthetic outcome for one design, as a frame.

        Fits a **copy** of the prototype estimator --- the one passed to
        :func:`rank_designs` is never mutated --- with the simulated test in the
        last ``duration`` periods of the panel, the most recent of the
        windows the power simulation evaluated.

        Parameters
        ----------
        candidate : str or iterable of object
            Canonical key or the original treated set.
        duration : int
            An evaluated duration.

        Returns
        -------
        polars.DataFrame
            ``[time, actual, synthetic, gap, window]``, one row per period;
            ``window`` is ``"pre"`` or ``"test"``.

        Raises
        ------
        KeyError, ValueError
            As :meth:`power`, or if the ranking has no fitting context.
        TypeError
            If the estimator does not expose ``periods_``, ``actual_`` and
            ``synthetic_`` after fitting (``Synth`` and ``AugSynth`` do).
        """
        self.power(candidate, duration)
        ctx = self._require_context()
        key = self._key(candidate)
        periods = ctx.panel.get_column(ctx.time).unique().sort()
        model = copy.deepcopy(ctx.estimator).fit(
            ctx.panel,
            unit=ctx.unit,
            time=ctx.time,
            outcome=ctx.outcome,
            treated=list(self.candidates[key]),
            treatment_time=periods[-duration],
        )
        missing = [a for a in ("periods_", "actual_", "synthetic_") if not hasattr(model, a)]
        if missing:
            raise TypeError(
                f"{type(model).__name__} does not expose {missing} after fit; fit_path "
                "needs the fitted actual and synthetic paths"
            )
        actual = np.asarray(getattr(model, "actual_"), dtype=np.float64)  # noqa: B009
        synthetic = np.asarray(getattr(model, "synthetic_"), dtype=np.float64)  # noqa: B009
        if len(actual) != periods.len():
            raise ValueError("fitted path length does not match the panel's periods")
        return pl.DataFrame(
            {
                ctx.time: periods,
                "actual": actual,
                "synthetic": synthetic,
                "gap": actual - synthetic,
                "window": np.where(np.asarray(model.pre_mask_), "pre", "test"),
            }
        )

    def plot_power(
        self,
        candidate: str | Iterable[Any],
        *,
        durations: Sequence[int] | None = None,
        ax: Axes | None = None,
    ) -> Axes:
        """Plot power against effect size, one line per duration.

        Requires the ``plot`` extra (matplotlib). A dot on each line marks the
        interpolated ``mde`` at the target power.

        Parameters
        ----------
        candidate : str or iterable of object
            Canonical key or the original treated set.
        durations : sequence of int, optional
            Durations to draw; all evaluated ones by default (at most eight).
        ax : matplotlib.axes.Axes, optional
            Axes to draw on; a new figure otherwise.

        Returns
        -------
        matplotlib.axes.Axes
        """
        key = self._key(candidate)
        if key not in self._results:
            raise KeyError(f"no results for candidate {key!r}")
        results = self._results[key]
        evaluated = list(results.params.durations) if results.params is not None else []
        chosen = evaluated if durations is None else list(durations)
        for d in chosen:
            self.power(key, d)
        ctx = self._require_context()
        mdes = {d: (row or {}).get("mde") for d in chosen for row in [self._row(key, d)]}
        return _plotting.power(
            results.power_curve(alpha=ctx.alpha),
            durations=chosen,
            mdes=mdes,
            target_power=ctx.target_power,
            title=key,
            ax=ax,
        )

    def plot_fit(
        self, candidate: str | Iterable[Any], duration: int, *, ax: Axes | None = None
    ) -> Axes:
        """Plot actual vs synthetic outcome, with the test window shaded.

        Requires the ``plot`` extra (matplotlib). The data is
        :meth:`fit_path`.

        Parameters
        ----------
        candidate : str or iterable of object
            Canonical key or the original treated set.
        duration : int
            An evaluated duration.
        ax : matplotlib.axes.Axes, optional
            Axes to draw on; a new figure otherwise.

        Returns
        -------
        matplotlib.axes.Axes
        """
        key = self._key(candidate)
        path = self.fit_path(key, duration)
        ctx = self._require_context()
        row = self._row(key, duration)
        return _plotting.fit(
            path,
            time=ctx.time,
            outcome=ctx.outcome,
            rmspe_pre=None if row is None else row.get("rmspe_pre"),
            title=f"{key}, {duration} periods",
            ax=ax,
        )

    def within_budget(self, budget: float) -> DesignRanking:
        """Keep the designs whose cost at their own MDE fits ``budget``.

        Cost is ``investment_mde`` --- ``cpic x mde_grid x sum(Y_treated)``, which
        is R GeoLift's ``Investment``. Because ``mde_grid >= mde``, it is an upper
        bound: a design that passes can afford the lift it is able to detect.
        Designs with no MDE on the evaluated grid are dropped.

        Filters from the unfiltered ranking, so a budget can be loosened as well
        as tightened without recomputing anything. Survivors are re-ranked.

        Parameters
        ----------
        budget : float
            Maximum spend, in the currency of ``cpic``.

        Returns
        -------
        DesignRanking
            A new result; the power results stay reachable for every candidate.

        Raises
        ------
        ValueError
            If the ranking was built without ``cpic``, or ``budget`` is not a
            finite, non-negative number.

        Warns
        -----
        UserWarning
            If no design fits, naming the cheapest cost available.
        """
        if not (math.isfinite(budget) and budget >= 0):
            raise ValueError(f"budget must be a finite, non-negative number; got {budget!r}")
        full = self.ranking if self._unfiltered is None else self._unfiltered
        if "investment_mde" not in full.columns:
            raise ValueError("within_budget needs a ranking built with cpic")
        kept = full.filter(pl.col("investment_mde") <= budget)
        if not kept.height:
            cheapest = full.get_column("investment_mde").min()
            warnings.warn(
                f"no design fits budget={budget:g}; the cheapest investment_mde is "
                f"{cheapest if cheapest is None else format(cheapest, 'g')}",
                UserWarning,
                stacklevel=2,
            )
        return replace(
            self,
            ranking=_apply_ranking(kept.drop("rank")),
            budget=budget,
            _unfiltered=full,
        )

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
        key = self._key(candidate)
        if key not in self._results:
            raise KeyError(f"no results for candidate {key!r}")
        results = self._results[key]
        durations = results.params.durations if results.params is not None else ()
        if duration not in durations:
            raise ValueError(f"duration {duration} was not evaluated; got {list(durations)}")
        return results

    def stability(
        self,
        *,
        rng: np.random.Generator,
        n_draws: int = 500,
        level: float = 0.90,
        top: int = 3,
    ) -> RankStability:
        """Rank intervals from a paired bootstrap over the simulated evidence.

        Each draw resamples the ``replicates x lookback_window`` placebo tests
        with replacement --- the same indices for every design, so a calendar
        window that hurts every synthetic control hurts every design in the
        same draw --- recomputes ``mde``, ``empirical_size`` and ``rmspe_pre``
        from the resampled rows, and ranks the draw with the rule that produced
        :attr:`ranking`. No model is refit.

        The designs ranked are those in :attr:`ranking`; a ``budget`` is not
        re-applied per draw. What the result measures, and what it does not,
        is in ``docs/methodology.md``: in particular the intervals are
        optimistic, because consecutive placebo windows overlap.

        Parameters
        ----------
        rng : numpy.random.Generator
            Required, no default.
        n_draws : int, default 500
            Bootstrap draws.
        level : float, default 0.90
            Coverage of the rank and MDE intervals.
        top : int, default 3
            ``p_top`` is the share of draws in which the design ranks at or
            above this position.

        Returns
        -------
        RankStability

        Raises
        ------
        ValueError
            Bad arguments; a ranking not built by :func:`rank_designs`; a pool
            of one window (``lookback_window=1`` with ``replicates=1``).

        Warns
        -----
        UserWarning
            When the pool has fewer than 10 placebo tests.
        """
        if n_draws < 1:
            raise ValueError(f"n_draws must be >= 1, got {n_draws}")
        if not 0.0 < level < 1.0:
            raise ValueError(f"level must be in (0, 1), got {level}")
        if top < 1:
            raise ValueError(f"top must be >= 1, got {top}")
        context = self._require_context()
        if not self.ranking.height:
            raise ValueError("the ranking has no designs to resample (did a budget drop them all?)")
        pool, pool_size = _simulation_pool(self.ranking, self._results, alpha=context.alpha)
        if pool_size < 2:
            raise ValueError(
                "stability needs more than one window to resample: increase "
                "lookback_window or replicates"
            )
        if pool_size < 10:
            warnings.warn(
                f"the simulation pool has {pool_size} placebo tests; the bootstrap is "
                "coarse on a small pool. Increase lookback_window or replicates.",
                UserWarning,
                stacklevel=2,
            )
        draws = _bootstrap_ranking(
            pool,
            pool_size=pool_size,
            n_draws=n_draws,
            rng=rng,
            target_power=context.target_power,
        )
        lo, hi = (1.0 - level) / 2.0, 1.0 - (1.0 - level) / 2.0
        defined = pl.col("p_undetectable") <= 1.0 - level
        summary = (
            draws.group_by(["candidate", "duration"])
            .agg(
                rank_low=pl.col("rank").quantile(lo, interpolation="nearest").cast(pl.Int64),
                rank_high=pl.col("rank").quantile(hi, interpolation="nearest").cast(pl.Int64),
                p_top=(pl.col("rank") <= top).mean(),
                p_undetectable=pl.col("mde").is_null().mean(),
                mde_low=pl.col("mde").drop_nulls().quantile(lo, interpolation="nearest"),
                mde_high=pl.col("mde").drop_nulls().quantile(hi, interpolation="nearest"),
            )
            .with_columns(pl.when(defined).then(pl.col("mde_low", "mde_high")))
        )
        frame = (
            self.ranking.select("candidate", "duration", "rank")
            .with_row_index("_i")
            .join(summary, on=["candidate", "duration"], how="left")
            .sort("_i")
            .drop("_i")
        )
        return RankStability(
            frame=frame, n_draws=n_draws, level=level, top=top, pool_size=pool_size
        )


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
    budget: float | None = None,
    n_jobs: int = 1,
    replicates: int = 1,
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
        is added (cost at ``target_effect_size``), plus ``investment_mde`` (cost
        at the design's own ``mde_grid``); when omitted, no cost is reported
        rather than assumed.
    budget : float, optional
        Keep only designs whose ``investment_mde`` fits; survivors are
        re-ranked. Requires ``cpic``. See :meth:`DesignRanking.within_budget`.
    n_jobs : int, default 1
        Parallelism **across candidates**. ``n_jobs=1`` is always passed down to
        ``simulate_power`` — joblib does not nest workers.
    replicates : int, default 1
        Independent ``simulate_power`` runs per candidate, each with its own
        child generator, pooled into one ``PowerResults`` (a ``replicate``
        column tags the rows). Every ranking column is then a rate or mean
        over ``replicates x lookback_window`` simulations: a row at
        ``replicates=3`` is a better estimate of the same quantity, not the
        same number. Pooling re-draws the permutations for the *same* placebo
        windows, so it sharpens cells near the decision boundary and cannot
        add windows; see :meth:`DesignRanking.stability`.

    Returns
    -------
    DesignRanking
        Ranking frame plus the underlying power results.

    Raises
    ------
    ValueError
        If two candidates collide on the canonical key, ``candidates`` is
        empty, ``budget`` is given without ``cpic``, or ``replicates < 1``.

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
    if budget is not None and cpic is None:
        raise ValueError("budget needs cpic: the cost of a design is cpic x mde_grid x sum(Y)")
    if replicates < 1:
        raise ValueError(f"replicates must be >= 1, got {replicates}")
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

    children = rng.spawn(len(keys) * replicates)

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

    tasks = [member for member in members for _ in range(replicates)]
    computed: list[PowerResults] = Parallel(n_jobs=n_jobs)(
        delayed(_one)(m, c) for m, c in zip(tasks, children, strict=True)
    )
    results = {
        key: _pool_results(computed[i * replicates : (i + 1) * replicates])
        for i, key in enumerate(keys)
    }

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
            lookback_window=lookback_window,
        )
        for key, treated in zip(keys, members, strict=True)
        for duration in duration_list
    ]
    ranked = DesignRanking(
        ranking=_apply_ranking(pl.DataFrame(rows, schema=_ranking_schema(cpic))),
        candidates=dict(zip(keys, members, strict=True)),
        _results=results,
        _context=_Context(
            panel=panel,
            estimator=estimator,
            unit=unit,
            time=time,
            outcome=outcome,
            target_power=target_power,
            alpha=alpha,
        ),
    )
    return ranked if budget is None else ranked.within_budget(budget)


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
        schema["investment_mde"] = pl.Float64
    return schema


def _pool_results(runs: Sequence[PowerResults]) -> PowerResults:
    """Concatenate replicate runs of one candidate into one ``PowerResults``.

    Rows get a ``replicate`` column; ``power_curve`` and ``mde`` group by
    duration and effect size and so read the pool as one larger sample.
    Rebuilding ``PowerResults`` from a ``simulations`` frame is a documented
    use of augsynth-py's public API; ``params`` is shared, since only the
    generator differed between runs.
    """
    first = runs[0]
    frames = [
        run.simulations.with_columns(pl.lit(i, dtype=pl.Int64).alias("replicate"))
        for i, run in enumerate(runs)
    ]
    return PowerResults(
        simulations=pl.concat(frames),
        alpha=first.alpha,
        effect_type=first.effect_type,
        params=first.params,
    )


def _simulation_pool(
    ranking: pl.DataFrame, results: dict[str, PowerResults], *, alpha: float
) -> tuple[pl.DataFrame, int]:
    """Long frame of every simulation behind the ranked designs, pool-indexed.

    One row per ``(candidate, duration, effect_size, pool_idx)``, where
    ``pool_idx`` enumerates the distinct ``(replicate, window)`` pairs ---
    shared across candidates, which is what makes the bootstrap paired.
    ``detected`` is null for a failed simulation (``on_error="record"``), so
    it drops out of the denominator exactly as in ``PowerResults.power_curve``.
    """
    designs = ranking.select("candidate", "duration")
    frames = []
    for key, run in results.items():
        sims = run.simulations
        if "replicate" not in sims.columns:
            sims = sims.with_columns(pl.lit(0, dtype=pl.Int64).alias("replicate"))
        frames.append(
            sims.select(
                pl.lit(key).alias("candidate"),
                pl.col("duration").cast(pl.Int64),
                "effect_size",
                pl.col("replicate").cast(pl.Int64),
                pl.col("window").cast(pl.Int64),
                detected=pl.col("pvalue") <= alpha,
                rmspe_pre=pl.col("rmspe_pre").cast(pl.Float64),
            )
        )
    long = pl.concat(frames).join(designs, on=["candidate", "duration"], how="semi")
    pairs = (
        long.select("replicate", "window")
        .unique()
        .sort(["replicate", "window"])
        .with_row_index("pool_idx")
        .with_columns(pl.col("pool_idx").cast(pl.Int64))
    )
    pool = long.join(pairs, on=["replicate", "window"]).drop("replicate", "window")
    per_cell = pool.group_by(["candidate", "duration", "effect_size"]).len().get_column("len")
    if (per_cell != pairs.height).any():
        raise ValueError("designs were simulated on different placebo windows; cannot pair draws")
    return pool, pairs.height


def _bootstrap_ranking(
    pool: pl.DataFrame,
    *,
    pool_size: int,
    n_draws: int,
    rng: np.random.Generator,
    target_power: float,
) -> pl.DataFrame:
    """Rank every design in every draw; one row per ``(draw, candidate, duration)``.

    A draw is a multinomial count over the pool --- sampling ``pool_size``
    placebo tests with replacement --- applied as weights to every design, so
    the curves, the interpolated MDE and the rank are recomputed from the same
    resampled evidence for all of them.
    """
    counts = rng.multinomial(pool_size, np.full(pool_size, 1.0 / pool_size), size=n_draws)
    weights = pl.DataFrame(
        {
            "draw": np.repeat(np.arange(n_draws), pool_size).astype(np.int64),
            "pool_idx": np.tile(np.arange(pool_size), n_draws).astype(np.int64),
            "w": counts.ravel().astype(np.int64),
        }
    ).filter(pl.col("w") > 0)
    return _rank_weighted_draws(weights, pool, target_power=target_power)


def _rank_weighted_draws(
    weights: pl.DataFrame, pool: pl.DataFrame, *, target_power: float
) -> pl.DataFrame:
    """Recompute the ranking keys and the rank for every weighted draw.

    ``weights`` has one row per ``(draw, pool_idx)`` with a positive integer
    ``w``. A draw with every weight equal to one is the point ranking: the
    columns here are, by construction, the quantities ``_summarize`` reports
    --- a test asserts it.
    """
    w = pl.col("w")
    cells = (
        pool.join(weights, on="pool_idx")
        .group_by(["draw", "candidate", "duration", "effect_size"])
        .agg(
            n_detected=(w * pl.col("detected").cast(pl.Int64)).sum(),
            n_valid=w.filter(pl.col("detected").is_not_null()).sum(),
            rmspe_w=(w * pl.col("rmspe_pre")).sum(),
            rmspe_n=w.filter(pl.col("rmspe_pre").is_not_null()).sum().cast(pl.Float64),
        )
        .with_columns(
            power=pl.when(pl.col("n_valid") > 0)
            .then(pl.col("n_detected") / pl.col("n_valid"))
            .otherwise(None)
        )
        .sort(["draw", "candidate", "duration", "effect_size"])
    )
    per_design = cells.group_by(["draw", "candidate", "duration"], maintain_order=True).agg(
        effect_sizes=pl.col("effect_size"),
        powers=pl.col("power"),
        empirical_size=pl.col("power").filter(pl.col("effect_size") == 0.0).first(),
        rmspe_pre=pl.when(pl.col("rmspe_n").sum() > 0)
        .then(pl.col("rmspe_w").sum() / pl.col("rmspe_n").sum())
        .otherwise(None),
    )
    mdes = [
        _interp_mde(effects, powers, target_power=target_power)
        for effects, powers in zip(
            per_design.get_column("effect_sizes").to_list(),
            per_design.get_column("powers").to_list(),
            strict=True,
        )
    ]
    draws = per_design.drop("effect_sizes", "powers").with_columns(
        pl.Series("mde", mdes, dtype=pl.Float64)
    )
    return _apply_ranking(draws, by=("draw",))


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


def _lookback_treated_y(
    panel: pl.DataFrame,
    periods: pl.Series,
    *,
    unit: str,
    time: str,
    outcome: str,
    treated: tuple[Any, ...],
    duration: int,
    lookback_window: int,
) -> float:
    """Treated outcome per test window, averaged over the lookback windows.

    The cost basis of ``investment`` and ``investment_mde``: the simulated test
    windows end at the last period, the one before, and so on for
    ``lookback_window`` windows, and a design is priced at their mean. This is
    R GeoLift's ``Investment`` exactly --- measured to the cent, see the
    characterization doc; with ``lookback_window = 1`` it reduces to the last
    window.
    """
    per_period = (
        panel.filter(pl.col(unit).is_in(list(treated))).group_by(time).agg(pl.col(outcome).sum())
    )
    series = (
        pl.DataFrame({time: periods})
        .join(per_period, on=time, how="left")
        .get_column(outcome)
        .fill_null(0.0)
        .to_numpy()
        .astype(float)
    )
    n = len(series)
    sums = [
        series[n - k - duration : n - k].sum()
        for k in range(lookback_window)
        if n - k - duration >= 0
    ]
    return float(np.mean(sums)) if sums else 0.0


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
    lookback_window: int,
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
    # Counted in distinct periods, not time arithmetic, so dates work too.
    periods = panel.get_column(time).unique().sort()
    window = panel.filter(pl.col(time).is_in(periods.tail(duration).to_list()))
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
        priced_y = _lookback_treated_y(
            panel,
            periods,
            unit=unit,
            time=time,
            outcome=outcome,
            treated=treated,
            duration=duration,
            lookback_window=lookback_window,
        )
        row["investment"] = cpic * target_effect_size * priced_y
        row["investment_mde"] = None if grid_mde is None else cpic * grid_mde * priced_y
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
