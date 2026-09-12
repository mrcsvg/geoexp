"""Unit tests for geoexp.selection (v0.5 design doc)."""

from __future__ import annotations

import pytest


def test_candidate_key_sorts_members_and_joins_with_comma_space() -> None:
    # Matches the oracle's convention so parity joins are a plain string match:
    # GeoLift emits BestMarkets$location as sorted, ", "-joined member names.
    from geoexp.selection import _candidate_key

    assert _candidate_key({"san diego", "nashville"}) == "nashville, san diego"


def test_candidate_key_orders_non_string_units_by_string_form() -> None:
    # PowerParams.treated is tuple[Any, ...]: units need not be mutually
    # orderable, so the total order comes from str(), not from the unit.
    from geoexp.selection import _candidate_key

    assert _candidate_key({10, 9}) == "10, 9"


def test_candidate_keys_rejects_distinct_candidates_that_collide() -> None:
    # 1 and "1" are different units but produce the same canonical key;
    # silently collapsing them would corrupt the ranking frame.
    from geoexp.selection import _candidate_keys

    with pytest.raises(ValueError, match="collide"):
        _candidate_keys([{1}, {"1"}])


def test_candidate_keys_returns_one_key_per_candidate_in_order() -> None:
    from geoexp.selection import _candidate_keys

    assert _candidate_keys([{"b", "a"}, {"c"}]) == ["a, b", "c"]


def test_interp_mde_interpolates_between_grid_points() -> None:
    # Grid MDE would return 0.10; interpolation places the crossing where the
    # curve actually reaches target power. This is what breaks ranking ties.
    from geoexp.selection import _interp_mde

    mde = _interp_mde([0.0, 0.05, 0.10], [0.0, 0.6, 1.0], target_power=0.8)

    assert mde == pytest.approx(0.075)


def test_interp_mde_returns_none_when_curve_never_reaches_target() -> None:
    # Never extrapolate past the grid: a design that cannot reach the target
    # power on the evaluated grid has no MDE to report.
    from geoexp.selection import _interp_mde

    assert _interp_mde([0.0, 0.05, 0.10], [0.0, 0.2, 0.5], target_power=0.8) is None


def test_interp_mde_monotonizes_before_interpolating() -> None:
    # Simulated power curves are noisy and can dip. Running maximum first, so a
    # dip does not move the reported crossing.
    from geoexp.selection import _interp_mde

    mde = _interp_mde([0.05, 0.10, 0.15], [0.3, 0.2, 0.9], target_power=0.8)

    # running max -> [0.3, 0.3, 0.9]; crossing interpolated from y0 = 0.3
    assert mde == pytest.approx(0.10 + (0.8 - 0.3) / (0.9 - 0.3) * 0.05)


def test_interp_mde_ignores_the_zero_effect_row() -> None:
    # effect_size == 0 is the empirical size, not a detectable effect.
    from geoexp.selection import _interp_mde

    assert _interp_mde([0.0, 0.05], [1.0, 1.0], target_power=0.8) == pytest.approx(0.05)


def _ranking_frame(rows: list[dict[str, object]]):
    import polars as pl

    return pl.DataFrame(
        rows,
        schema={
            "candidate": pl.String,
            "mde": pl.Float64,
            "empirical_size": pl.Float64,
            "rmspe_pre": pl.Float64,
        },
    )


def test_apply_ranking_orders_by_mde_then_size_then_fit() -> None:
    from geoexp.selection import _apply_ranking

    out = _apply_ranking(
        _ranking_frame(
            [
                {"candidate": "c", "mde": 0.10, "empirical_size": 0.0, "rmspe_pre": 0.01},
                {"candidate": "a", "mde": 0.05, "empirical_size": 0.3, "rmspe_pre": 0.01},
                {"candidate": "b", "mde": 0.05, "empirical_size": 0.1, "rmspe_pre": 0.01},
            ]
        )
    )

    assert out.get_column("candidate").to_list() == ["b", "a", "c"]


def test_apply_ranking_gives_equal_rank_to_substantively_tied_designs() -> None:
    # candidate breaks display order only. Two designs identical on every
    # substantive criterion must share a rank, as the oracle's does.
    from geoexp.selection import _apply_ranking

    out = _apply_ranking(
        _ranking_frame(
            [
                {"candidate": "a", "mde": 0.05, "empirical_size": 0.1, "rmspe_pre": 0.02},
                {"candidate": "b", "mde": 0.05, "empirical_size": 0.1, "rmspe_pre": 0.02},
                {"candidate": "c", "mde": 0.10, "empirical_size": 0.1, "rmspe_pre": 0.02},
            ]
        )
    )

    # competition ranking: the tie consumes rank 2, so the next row is 3
    assert out.get_column("rank").to_list() == [1, 1, 3]


def test_apply_ranking_puts_designs_without_an_mde_last() -> None:
    # A design whose curve never reaches target power is the worst design,
    # not a missing value to be sorted first.
    from geoexp.selection import _apply_ranking

    out = _apply_ranking(
        _ranking_frame(
            [
                {"candidate": "a", "mde": None, "empirical_size": 0.0, "rmspe_pre": 0.01},
                {"candidate": "b", "mde": 0.20, "empirical_size": 0.0, "rmspe_pre": 0.01},
            ]
        )
    )

    assert out.get_column("candidate").to_list() == ["b", "a"]


def _synthetic_panel():
    """Small deterministic panel: 6 units x 30 periods, one shared trend."""
    import numpy as np
    import polars as pl

    rng = np.random.default_rng(0)
    units = ["u1", "u2", "u3", "u4", "u5", "u6"]
    times = range(1, 31)
    trend = {t: 100.0 + 2.0 * t for t in times}
    rows = [
        {"loc": u, "t": t, "y": trend[t] * (1.0 + 0.1 * i) + rng.normal(0, 2.0)}
        for i, u in enumerate(units)
        for t in times
    ]
    return pl.DataFrame(rows, schema={"loc": pl.String, "t": pl.Int64, "y": pl.Float64})


def test_rank_designs_returns_one_row_per_candidate_and_duration() -> None:
    import numpy as np
    from augsynth_py import Synth

    from geoexp.selection import rank_designs

    with pytest.warns(UserWarning, match="lookback_window"):
        res = rank_designs(
            _synthetic_panel(),
            estimator=Synth(),
            unit="loc",
            time="t",
            outcome="y",
            candidates=[{"u1"}, {"u2", "u3"}],
            durations=[3, 5],
            effect_sizes=(0.0, 0.1, 0.2),
            lookback_window=2,
            ns=50,
            permutation_type="iid",
            rng=np.random.default_rng(11),
        )

    assert res.ranking.height == 4  # 2 candidates x 2 durations
    assert set(res.ranking.get_column("candidate")) == {"u1", "u2, u3"}
    for col in (
        "candidate",
        "duration",
        "n_units",
        "mde",
        "mde_grid",
        "power_at_target",
        "empirical_size",
        "rmspe_pre",
        "rank",
    ):
        assert col in res.ranking.columns


def test_rank_designs_exposes_the_underlying_power_results() -> None:
    import numpy as np
    from augsynth_py import PowerResults, Synth

    from geoexp.selection import rank_designs

    with pytest.warns(UserWarning, match="lookback_window"):
        res = rank_designs(
            _synthetic_panel(),
            estimator=Synth(),
            unit="loc",
            time="t",
            outcome="y",
            candidates=[{"u2", "u3"}],
            durations=[3],
            effect_sizes=(0.0, 0.1),
            lookback_window=2,
            ns=50,
            permutation_type="iid",
            rng=np.random.default_rng(11),
        )

    # accepts the canonical key and the original set interchangeably
    assert isinstance(res.power("u2, u3", 3), PowerResults)
    assert isinstance(res.power({"u3", "u2"}, 3), PowerResults)


def test_rank_designs_warns_when_lookback_window_is_too_small_for_empirical_size() -> None:
    # empirical_size is a rate over the lookback window and participates in the
    # ranking order; at a small window it is a coin flip, not a rate.
    import numpy as np
    from augsynth_py import Synth

    from geoexp.selection import rank_designs

    with pytest.warns(UserWarning, match="empirical_size"):
        rank_designs(
            _synthetic_panel(),
            estimator=Synth(),
            unit="loc",
            time="t",
            outcome="y",
            candidates=[{"u1"}],
            durations=[3],
            effect_sizes=(0.0, 0.1),
            lookback_window=1,
            ns=20,
            permutation_type="iid",
            rng=np.random.default_rng(3),
        )


def _rank(**overrides):
    import numpy as np
    from augsynth_py import Synth

    from geoexp.selection import rank_designs

    kwargs = dict(
        estimator=Synth(),
        unit="loc",
        time="t",
        outcome="y",
        candidates=[{"u1"}, {"u2", "u3"}],
        durations=[3],
        effect_sizes=(0.0, 0.1, 0.2),
        lookback_window=2,
        ns=50,
        permutation_type="iid",
        rng=np.random.default_rng(11),
    )
    kwargs.update(overrides)
    with pytest.warns(UserWarning, match="lookback_window"):
        return rank_designs(_synthetic_panel(), **kwargs)


def test_rank_designs_is_reproducible_across_worker_counts() -> None:
    # One child generator per candidate via rng.spawn: results must not depend
    # on how joblib schedules the candidates.
    from polars.testing import assert_frame_equal

    assert_frame_equal(_rank(n_jobs=1).ranking, _rank(n_jobs=2).ranking)


def test_rank_designs_adds_investment_only_when_cpic_is_given() -> None:
    # No fabricated costs: without cpic the column is absent entirely.
    assert "investment" not in _rank().ranking.columns
    assert "investment" in _rank(cpic=2.0).ranking.columns


def test_rank_designs_rejects_an_empty_candidate_list() -> None:
    import numpy as np
    from augsynth_py import Synth

    from geoexp.selection import rank_designs

    with pytest.raises(ValueError, match="must not be empty"):
        rank_designs(
            _synthetic_panel(),
            estimator=Synth(),
            unit="loc",
            time="t",
            outcome="y",
            candidates=[],
            durations=[3],
            rng=np.random.default_rng(1),
        )


# --- candidate enumeration -------------------------------------------------


def test_enumerate_candidates_returns_sets_of_each_requested_size() -> None:
    from geoexp.selection import enumerate_candidates

    cands = enumerate_candidates(
        _synthetic_panel(), unit="loc", time="t", outcome="y", sizes=[1, 2]
    )

    assert {len(c) for c in cands} == {1, 2}
    assert all(isinstance(c, frozenset) for c in cands)


def test_enumerate_candidates_caps_the_number_generated_per_size() -> None:
    # C(6, 3) = 20; without a cap this explodes on real panels.
    from geoexp.selection import enumerate_candidates

    cands = enumerate_candidates(
        _synthetic_panel(), unit="loc", time="t", outcome="y", sizes=[3], max_per_size=4
    )

    assert len(cands) == 4


def test_enumerate_candidates_always_includes_required_units() -> None:
    from geoexp.selection import enumerate_candidates

    cands = enumerate_candidates(
        _synthetic_panel(), unit="loc", time="t", outcome="y", sizes=[2], include=["u5"]
    )

    assert all("u5" in c for c in cands)


def test_enumerate_candidates_never_treats_excluded_units() -> None:
    # Excluded units stay in the donor pool; they are just never treated.
    from geoexp.selection import enumerate_candidates

    cands = enumerate_candidates(
        _synthetic_panel(), unit="loc", time="t", outcome="y", sizes=[2], exclude=["u1"]
    )

    assert all("u1" not in c for c in cands)


def test_enumerate_candidates_is_deterministic_without_an_rng() -> None:
    from geoexp.selection import enumerate_candidates

    kw = dict(unit="loc", time="t", outcome="y", sizes=[2], max_per_size=5)
    first = enumerate_candidates(_synthetic_panel(), **kw)
    second = enumerate_candidates(_synthetic_panel(), **kw)

    assert first == second


def test_enumerate_candidates_samples_reproducibly_when_given_an_rng() -> None:
    import numpy as np

    from geoexp.selection import enumerate_candidates

    kw = dict(unit="loc", time="t", outcome="y", sizes=[3], max_per_size=4)
    a = enumerate_candidates(_synthetic_panel(), rng=np.random.default_rng(4), **kw)
    b = enumerate_candidates(_synthetic_panel(), rng=np.random.default_rng(4), **kw)
    c = enumerate_candidates(_synthetic_panel(), rng=np.random.default_rng(5), **kw)

    assert a == b
    assert a != c  # different draw, not a fixed list


def test_enumerate_candidates_rejects_a_size_larger_than_the_eligible_pool() -> None:
    from geoexp.selection import enumerate_candidates

    with pytest.raises(ValueError, match="eligible"):
        enumerate_candidates(_synthetic_panel(), unit="loc", time="t", outcome="y", sizes=[9])


# --- diagnostics adapted from augsynth-py#22 -------------------------------


def test_null_bias_is_the_mean_recovered_lift_under_a_zero_effect() -> None:
    # The honest analog of GeoLift's abs_lift_in_zero: rather than reproducing
    # their number (whose definition is not published), report the estimator's
    # own recovered lift when nothing was injected. Signed, so direction shows.
    import polars as pl

    res = _rank()
    row = res.ranking.filter(pl.col("candidate") == "u1").row(0, named=True)

    sims = res.power("u1", 3).simulations
    expected = (
        sims.filter((pl.col("duration") == 3) & (pl.col("effect_size") == 0.0))
        .get_column("att_pct")
        .mean()
    )

    assert row["null_bias"] == pytest.approx(expected)


def test_h1_calibration_error_measures_recovery_of_the_injected_effect() -> None:
    # Catches a design with high power but a biased estimate: power says
    # "detected", calibration says "detected the wrong magnitude".
    import polars as pl

    res = _rank()
    row = res.ranking.filter(pl.col("candidate") == "u1").row(0, named=True)
    grid_mde = row["mde_grid"]
    if grid_mde is None:
        pytest.skip("candidate never reaches target power on this grid")

    recovered = (
        res.power("u1", 3)
        .simulations.filter((pl.col("duration") == 3) & (pl.col("effect_size") == grid_mde))
        .get_column("att_pct")
        .mean()
    )

    assert row["h1_calibration_error"] == pytest.approx(abs(recovered - grid_mde))


def test_h1_calibration_error_is_null_for_additive_effects() -> None:
    # att_pct and an additive effect size are not on the same scale, so the
    # comparison is not defined. Report nothing rather than a wrong number.
    res = _rank(effect_type="additive", effect_sizes=(0.0, 50.0, 100.0))

    assert res.ranking.get_column("h1_calibration_error").null_count() == res.ranking.height
