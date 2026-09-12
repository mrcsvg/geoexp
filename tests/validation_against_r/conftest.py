"""Fixtures bridging the R GeoLift oracle via rpy2.

The suite is guarded so developers without R can run tests/unit/ freely.
Requires R >= 4.5 and the GeoLift R package; see
.github/workflows/validation.yml.
"""

from __future__ import annotations

import polars as pl
import pytest

pytest.importorskip("rpy2")

from ._r_frames import from_r_data_frame

#: Design shared by both sides of every parity assertion. lookback_window is
#: deliberately well above 1: at 1 the power of a cell is a single 0/1 outcome
#: and the comparison loses nearly all of its resolution.
DURATION = 15
LOOKBACK_WINDOW = 10
EFFECT_SIZES = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25)
NS = 200
ALPHA = 0.1


@pytest.fixture(scope="session")
def geolift_panel() -> pl.DataFrame:
    """``GeoLift_PreTest`` after ``GeoDataRead``, taken from R.

    The fixture ships raw (``location``, ``Y``, ``date``) and has no ``time``
    column, which GeoLift requires. Taking the converted panel *from R* means
    both sides see byte-identical input and geoexp never reimplements
    ``GeoDataRead`` — which the clean-room rule would forbid anyway.
    """
    from rpy2 import robjects

    robjects.r("""
        suppressMessages(library(GeoLift)); data(GeoLift_PreTest)
        .geoexp_panel <- GeoDataRead(GeoLift_PreTest, date_id = "date",
            location_id = "location", Y_id = "Y", X = c(),
            format = "yyyy-mm-dd", summary = FALSE)
    """)
    return from_r_data_frame(robjects.r(".geoexp_panel")).with_columns(
        pl.col("time").cast(pl.Int64), pl.col("Y").cast(pl.Float64)
    )


@pytest.fixture(scope="session")
def geolift_power_curves(geolift_panel: pl.DataFrame) -> pl.DataFrame:
    """``GeoLiftMarketSelection`` power curves for every single-market candidate.

    Read from ``PowerCurves``, not ``BestMarkets``: the latter is filtered by
    GeoLift's own selection step and silently drops candidates.
    """
    from rpy2 import robjects

    robjects.r(f"""
        set.seed(1)
        .geoexp_ms <- GeoLiftMarketSelection(
            data = .geoexp_panel, treatment_periods = c({DURATION}), N = c(1),
            effect_size = c({", ".join(str(e) for e in EFFECT_SIZES)}),
            lookback_window = {LOOKBACK_WINDOW}, model = "none", cpic = 1,
            alpha = {ALPHA}, conformal_type = "iid", ns = {NS},
            side_of_test = "two_sided", parallel = FALSE,
            ProgressBar = FALSE, print = FALSE)
    """)
    return from_r_data_frame(robjects.r(".geoexp_ms$PowerCurves")).select(
        candidate=pl.col("location").cast(pl.String),
        duration=pl.col("duration").cast(pl.Int64),
        effect_size=pl.col("EffectSize").cast(pl.Float64),
        r_power=pl.col("power").cast(pl.Float64),
    )
