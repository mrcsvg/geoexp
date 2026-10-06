"""``prepare_panel`` parity against R ``GeoDataRead``.

The dirty fixture carries the four defects whose ``GeoDataRead`` behaviour was
measured in the characterization: a mixed-case unit name (lower-cased), a
duplicate unit x date row (summed), a missing date for one unit and a missing
outcome for another (both units dropped). It is built once, in R, and the same
raw frame goes to both sides.
"""

from __future__ import annotations

import warnings

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from geoexp import prepare_panel

from ._r_frames import from_r_data_frame

pytestmark = [pytest.mark.requires_r]


@pytest.fixture(scope="module")
def dirty_and_r() -> tuple[pl.DataFrame, pl.DataFrame]:
    from rpy2 import robjects

    robjects.r("""
        suppressMessages(library(GeoLift)); data(GeoLift_PreTest)
        d <- GeoLift_PreTest
        d$Y <- as.numeric(d$Y)
        d$location[d$location == "atlanta"] <- "Atlanta"
        d <- d[!(d$location == "boston" & d$date == "2021-01-15"), ]
        d <- rbind(d, d[d$location == "chicago" & d$date == "2021-01-10", ])
        d$Y[d$location == "denver" & d$date == "2021-01-20"] <- NA
        .geoexp_dirty <- d
        .geoexp_read <- GeoDataRead(data = d, date_id = "date", location_id = "location",
            Y_id = "Y", X = c(), format = "yyyy-mm-dd", summary = FALSE)
    """)
    dirty = from_r_data_frame(robjects.r(".geoexp_dirty")).with_columns(
        # R's NA_real_ arrives as NaN; a missing value is a null in Polars.
        pl.col("Y").cast(pl.Float64).fill_nan(None),
        pl.col("location").cast(pl.String),
        pl.col("date").cast(pl.String),
    )
    r_side = from_r_data_frame(robjects.r(".geoexp_read")).select(
        location=pl.col("location").cast(pl.String),
        time=pl.col("time").cast(pl.Int64),
        Y=pl.col("Y").cast(pl.Float64),
    )
    return dirty, r_side


def test_prepare_panel_matches_geodataread(dirty_and_r) -> None:
    dirty, r_side = dirty_and_r
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ours = prepare_panel(
            dirty.rename({"date": "time"}),
            unit="location",
            time="time",
            outcome="Y",
            date_format="%Y-%m-%d",
            time_index=True,
        )

    assert_frame_equal(ours, r_side.sort("location", "time"))
    # Same result as GeoDataRead, but the drops are said out loud.
    dropped = " ".join(str(w.message) for w in caught)
    assert "boston" in dropped
    assert "denver" in dropped
