"""Numerical parity against R ``GeoLiftMarketSelection``.

Scope is deliberate. The assertions target the **power curve**, which both
packages compute from the same definition, at high resolution: 40 candidates x
6 effect sizes. Quantities with no reproducible counterpart --- ``Average_MDE``,
``abs_lift_in_zero``, ``rank``, ``AvgScaledL2Imbalance`` --- are excluded, with
the reason recorded in ``docs/geolift-oracle-characterization.md`` §6. Do not
add assertions on ranking *order*: it is not stable across implementations
(§9 of that document), and a test built on it would fail on noise.

**How the tolerances are set.** Both sides estimate a conformal p-value by
resampling and then reduce it to a rate over ``LOOKBACK_WINDOW`` placebo tests,
so cell-level power is genuinely stochastic: a cell whose p-value sits near
``alpha`` flips between runs *of the same implementation*. Measured, each
implementation disagrees with itself about as much as it disagrees with the
other (~223-225 of 240 cells identical within one implementation, ~215-219
across them; mean absolute difference 0.0067 and 0.0079 within, 0.0063-0.0075
across). Fixed thresholds calibrated on one draw would therefore flake.

So the assertions below are either **self-calibrating** --- comparing the
cross-implementation difference against geoexp's own run-to-run difference
measured in the same session --- or **bias-based**, which is nearly noise-free
and is what actually moves when an implementation breaks.
"""

from __future__ import annotations

import warnings

import numpy as np
import polars as pl
import pytest
from augsynth_py import Synth

from geoexp import rank_designs

from .conftest import ALPHA, DURATION, EFFECT_SIZES, LOOKBACK_WINDOW, NS, child_safe_env

pytestmark = [pytest.mark.requires_r, pytest.mark.slow]

#: A real implementation error --- wrong window, wrong alpha, wrong effect
#: injection --- shifts the mean *signed* difference. Measured |bias| across
#: every seed pairing is <= 0.0008; 0.02 is 25x that and still a fifth of the
#: 1/LOOKBACK_WINDOW step size, so a systematic error cannot hide under it.
MAX_BIAS = 0.02

#: Cross-implementation disagreement must not exceed geoexp's own run-to-run
#: disagreement by more than this factor. Measured ratio is ~0.9-1.1 (the two
#: packages agree with each other as well as geoexp agrees with itself); 3.0
#: leaves generous headroom for seed variation while catching real drift.
MAX_NOISE_RATIO = 3.0


def _curves(panel: pl.DataFrame, candidates: list[str], seed: int) -> pl.DataFrame:
    """geoexp power curves for the given candidates at one seed."""
    # child_safe_env: n_jobs=-1 spawns Python subprocesses, and embedded R has
    # rewritten LD_LIBRARY_PATH by now -- see the context manager's docstring.
    with warnings.catch_warnings(), child_safe_env():
        # lookback_window is 10 here, above the warning threshold; guard anyway
        # so an unrelated warning cannot fail the suite under -W error.
        warnings.simplefilter("ignore", UserWarning)
        res = rank_designs(
            panel,
            estimator=Synth(),
            unit="location",
            time="time",
            outcome="Y",
            candidates=[{c} for c in candidates],
            durations=[DURATION],
            effect_sizes=EFFECT_SIZES,
            lookback_window=LOOKBACK_WINDOW,
            alpha=ALPHA,
            permutation_type="iid",
            ns=NS,
            side="two-sided",
            rng=np.random.default_rng(seed),
            n_jobs=-1,
        )
    return pl.concat(
        res.power(key, DURATION)
        .power_curve(alpha=ALPHA)
        .with_columns(candidate=pl.lit(key))
        .select("candidate", "effect_size", power=pl.col("power"))
        for key in res.candidates
    )


def _diff(a: pl.DataFrame, b: pl.DataFrame) -> pl.Series:
    """Signed per-cell power difference over the inner join."""
    joined = a.join(b, on=["candidate", "effect_size"], how="inner", suffix="_b")
    assert joined.height == 240, f"expected 40 candidates x 6 effect sizes, got {joined.height}"
    return joined.get_column("power") - joined.get_column("power_b")


@pytest.fixture(scope="session")
def candidates(geolift_power_curves: pl.DataFrame) -> list[str]:
    return sorted(geolift_power_curves.get_column("candidate").unique())


@pytest.fixture(scope="session")
def cross(geolift_panel, candidates, geolift_power_curves) -> pl.Series:
    """geoexp vs the R oracle."""
    r_side = geolift_power_curves.select("candidate", "effect_size", power=pl.col("r_power"))
    return _diff(_curves(geolift_panel, candidates, seed=7), r_side)


@pytest.fixture(scope="session")
def self_noise(geolift_panel, candidates) -> pl.Series:
    """geoexp against itself at a different seed: the noise floor."""
    return _diff(
        _curves(geolift_panel, candidates, seed=7),
        _curves(geolift_panel, candidates, seed=99),
    )


def test_power_shows_no_systematic_drift_against_the_oracle(cross: pl.Series) -> None:
    # The sharp regression detector: noise is symmetric, an implementation
    # error is not.
    assert abs(float(cross.mean())) <= MAX_BIAS


def test_power_agrees_with_the_oracle_as_well_as_geoexp_agrees_with_itself(
    cross: pl.Series, self_noise: pl.Series
) -> None:
    # The parity claim proper, stated so it cannot flake on a lucky draw.
    cross_mad = float(cross.abs().mean())
    floor = float(self_noise.abs().mean())
    assert cross_mad <= max(floor, 1e-6) * MAX_NOISE_RATIO, (
        f"cross-implementation MAD {cross_mad:.4f} exceeds {MAX_NOISE_RATIO}x "
        f"geoexp's own run-to-run MAD {floor:.4f}"
    )


def test_power_agrees_closely_in_absolute_terms(cross: pl.Series) -> None:
    # An absolute floor, so the test still means something if the noise floor
    # itself regresses. 0.02 is a fifth of the 1/LOOKBACK_WINDOW step size;
    # measured 0.0063-0.0075.
    assert float(cross.abs().mean()) <= 0.02


def test_most_cells_are_identical(cross: pl.Series) -> None:
    # Loose by design: the exact-match count is a random variable (measured
    # 215-226 of 240 across seed pairings, and 223-225 within one
    # implementation). 200 sits below every observed draw, cross or self.
    assert int((cross.abs() == 0.0).sum()) >= 200
