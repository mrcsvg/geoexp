"""Budget parity against R ``GeoLiftMarketSelection(budget = ...)``.

Measured semantics (``docs/geolift-oracle-characterization.md``, budget
section): ``BestMarkets.EffectSize`` is the smallest grid effect reaching power
0.8 --- geoexp's ``mde_grid`` --- and ``Investment = cpic x EffectSize x
sum(Y_treated)`` over the treatment window. ``budget`` keeps rows with
``Investment <= budget`` and re-ranks them.

The grid MDE itself sits on a power curve that both implementations estimate
by resampling, so the two sides can land on neighbouring grid points for a
candidate near the 0.8 boundary. The assertions are therefore conditioned on
the rows where the grid MDEs coincide: there, cost and the keep/drop decision
must agree exactly. A floor on how many rows that is guards against the
condition quietly emptying the test.
"""

from __future__ import annotations

import warnings

import numpy as np
import polars as pl
import pytest
from augsynth_py import Synth

from geoexp import rank_designs

from ._r_frames import from_r_data_frame
from .conftest import ALPHA, DURATION, EFFECT_SIZES, LOOKBACK_WINDOW, NS, child_safe_env

pytestmark = [pytest.mark.requires_r, pytest.mark.slow]

#: Rows where both sides pick the same grid MDE. Measured on the power-curve
#: sweep, ~90% of cells agree; half the rows is a loose floor that still fails
#: if the conditioning ever hides a systematic disagreement.
MIN_AGREEING_SHARE = 0.5


def _best_markets(budget: float | None) -> pl.DataFrame:
    from rpy2 import robjects

    extra = "" if budget is None else f", budget = {budget!r}"
    robjects.r(f"""
        set.seed(1)
        .geoexp_bm <- GeoLiftMarketSelection(
            data = .geoexp_panel, treatment_periods = c({DURATION}), N = c(2),
            effect_size = c({", ".join(str(e) for e in EFFECT_SIZES)}),
            lookback_window = {LOOKBACK_WINDOW}, model = "none", cpic = 1,
            alpha = {ALPHA}, conformal_type = "iid", ns = {NS},
            side_of_test = "two_sided", parallel = FALSE,
            ProgressBar = FALSE, print = FALSE{extra})$BestMarkets
    """)
    return from_r_data_frame(robjects.r(".geoexp_bm")).select(
        candidate=pl.col("location").cast(pl.String),
        r_effect=pl.col("EffectSize").cast(pl.Float64),
        r_investment=pl.col("Investment").cast(pl.Float64),
    )


@pytest.fixture(scope="module")
def r_full(geolift_panel: pl.DataFrame) -> pl.DataFrame:
    return _best_markets(None)


@pytest.fixture(scope="module")
def budget(r_full: pl.DataFrame) -> float:
    return float(r_full.get_column("r_investment").median())


@pytest.fixture(scope="module")
def r_kept(budget: float) -> pl.DataFrame:
    return _best_markets(budget)


@pytest.fixture(scope="module")
def joined(geolift_panel: pl.DataFrame, r_full: pl.DataFrame, budget: float) -> pl.DataFrame:
    """geoexp priced and budget-filtered on R's candidates, joined to R."""
    candidates = [set(c.split(", ")) for c in r_full.get_column("candidate")]
    with warnings.catch_warnings(), child_safe_env():
        warnings.simplefilter("ignore", UserWarning)
        res = rank_designs(
            geolift_panel,
            estimator=Synth(),
            unit="location",
            time="time",
            outcome="Y",
            candidates=candidates,
            durations=[DURATION],
            effect_sizes=EFFECT_SIZES,
            lookback_window=LOOKBACK_WINDOW,
            alpha=ALPHA,
            permutation_type="iid",
            ns=NS,
            side="two-sided",
            cpic=1.0,
            rng=np.random.default_rng(7),
            n_jobs=-1,
        )
    kept = set(res.within_budget(budget).ranking.get_column("candidate"))
    return (
        res.ranking.select("candidate", "mde_grid", "investment_mde")
        .join(r_full, on="candidate", how="inner")
        .with_columns(geoexp_kept=pl.col("candidate").is_in(list(kept)))
    )


def test_most_rows_agree_on_the_grid_mde(joined: pl.DataFrame) -> None:
    agree = joined.filter(pl.col("mde_grid") == pl.col("r_effect")).height
    assert agree >= MIN_AGREEING_SHARE * joined.height, f"{agree}/{joined.height}"


def test_investment_mde_equals_geolift_investment(joined: pl.DataFrame) -> None:
    same = joined.filter(pl.col("mde_grid") == pl.col("r_effect"))
    np.testing.assert_allclose(
        same.get_column("investment_mde").to_numpy(),
        same.get_column("r_investment").to_numpy(),
        rtol=1e-9,
    )


def test_budget_keeps_the_same_designs_as_geolift(
    joined: pl.DataFrame, r_kept: pl.DataFrame
) -> None:
    same = joined.filter(pl.col("mde_grid") == pl.col("r_effect")).with_columns(
        r_kept=pl.col("candidate").is_in(r_kept.get_column("candidate").to_list())
    )
    mismatched = same.filter(pl.col("geoexp_kept") != pl.col("r_kept"))
    assert mismatched.height == 0, mismatched
