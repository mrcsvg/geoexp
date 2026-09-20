"""Fixtures bridging the R GeoLift oracle via rpy2.

The suite is guarded so developers without R can run tests/unit/ freely.
Requires R >= 4.5 and the GeoLift R package; see
.github/workflows/validation.yml.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator

import polars as pl
import pytest

pytest.importorskip("rpy2")

from ._r_frames import from_r_data_frame

#: ``LD_LIBRARY_PATH`` as it was before embedded R had a chance to rewrite it.
#: Captured at import time: ``pytest.importorskip`` above loads the ``rpy2``
#: package but not ``rpy2.robjects``, so R is still uninitialized here.
_LD_LIBRARY_PATH_BEFORE_R = os.environ.get("LD_LIBRARY_PATH")


@contextlib.contextmanager
def child_safe_env() -> Iterator[None]:
    """Restore the pre-R ``LD_LIBRARY_PATH`` while Python subprocesses are spawned.

    Initializing embedded R sources ``$R_HOME/etc/ldpaths``, which rewrites
    ``LD_LIBRARY_PATH`` in this process to put R's own library directories --
    and, on a Linux runner, ``/usr/lib/x86_64-linux-gnu`` -- *ahead* of the
    Python installation's ``lib``. Every Python child spawned afterwards
    inherits that, so the dynamic loader hands it the system
    ``libpython3.12.so.1.0`` instead of the one belonging to the interpreter
    being launched. The child then computes a different set of site directories
    and comes up without ``site-packages``: joblib's loky workers die at
    bootstrap with ``ModuleNotFoundError: No module named 'joblib'`` and
    ``rank_designs(n_jobs=-1)`` fails with ``TerminatedWorkerError``.

    Measured on ubuntu-latest: under R's value ``ldd`` resolves the child's
    ``libpython3.12.so.1.0`` to ``/usr/lib/x86_64-linux-gnu``; restoring the
    original value puts ``site-packages`` back and ``Parallel`` runs.

    The restore is scoped to the spawn rather than made permanent because R
    still needs its own value to ``dlopen`` package libraries: nothing calls
    into R inside this block.
    """
    during_r = os.environ.get("LD_LIBRARY_PATH")
    _set_ld_library_path(_LD_LIBRARY_PATH_BEFORE_R)
    try:
        yield
    finally:
        _set_ld_library_path(during_r)


def _set_ld_library_path(value: str | None) -> None:
    if value is None:
        os.environ.pop("LD_LIBRARY_PATH", None)
    else:
        os.environ["LD_LIBRARY_PATH"] = value


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
