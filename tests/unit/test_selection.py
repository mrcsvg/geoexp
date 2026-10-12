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


def test_interp_mde_ignores_cells_whose_power_is_unknown() -> None:
    # A cell where every simulation failed has null power. It carries no
    # evidence and must not be read as "power 0" nor turn into the smallest
    # grid effect; it is skipped, as PowerResults.mde skips it.
    from geoexp.selection import _interp_mde

    assert _interp_mde([0.05, 0.1, 0.2], [None, 1.0, 1.0], target_power=0.8) == pytest.approx(0.1)
    # an unknown middle cell is treated as if the grid did not have it: the
    # crossing is interpolated between the evaluated neighbours
    assert _interp_mde([0.05, 0.1, 0.2], [0.2, None, 1.0], target_power=0.8) == pytest.approx(
        0.05 + (0.8 - 0.2) * (0.2 - 0.05) / (1.0 - 0.2)
    )
    assert _interp_mde([0.05, 0.1], [None, None], target_power=0.8) is None


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


def test_apply_ranking_can_rank_within_groups() -> None:
    # The stability bootstrap ranks every draw with the production rule in one call.
    import polars as pl

    from geoexp.selection import _apply_ranking

    frame = pl.DataFrame(
        {
            "draw": [0, 0, 1, 1],
            "candidate": ["a", "b", "a", "b"],
            "mde": [0.10, 0.20, 0.30, 0.20],
            "empirical_size": [0.1, 0.1, 0.1, 0.1],
            "rmspe_pre": [1.0, 1.0, 1.0, 1.0],
        }
    )
    out = _apply_ranking(frame, by=("draw",)).sort(["draw", "candidate"])
    assert out.get_column("rank").to_list() == [1, 2, 2, 1]


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


def test_replicates_pool_every_run_into_one_power_result() -> None:
    import polars as pl

    res = _rank(replicates=2)
    sims = res.power("u1", 3).simulations
    assert sims.get_column("replicate").unique().sort().to_list() == [0, 1]
    # lookback_window=2 windows per replicate -> 4 rows per cell
    per_cell = sims.group_by(["duration", "effect_size"]).len().get_column("len")
    assert per_cell.unique().to_list() == [4]
    # pooled power is the mean of the per-replicate powers
    pooled = res.power("u1", 3).power_curve().sort("effect_size").get_column("power")
    by_rep = (
        sims.group_by(["replicate", "effect_size"])
        .agg(power=(pl.col("pvalue") <= 0.1).mean())
        .group_by("effect_size")
        .agg(pl.col("power").mean())
        .sort("effect_size")
        .get_column("power")
    )
    assert pooled.to_list() == pytest.approx(by_rep.to_list())


def test_replicates_keep_the_design_parameters() -> None:
    res = _rank(replicates=2)
    params = res.power("u1", 3).params
    assert params is not None
    assert list(params.durations) == [3]


def test_replicates_are_reproducible_across_worker_counts() -> None:
    from polars.testing import assert_frame_equal

    assert_frame_equal(_rank(replicates=2, n_jobs=1).ranking, _rank(replicates=2, n_jobs=2).ranking)


def test_replicates_must_be_positive() -> None:
    import numpy as np
    from augsynth_py import Synth

    from geoexp.selection import rank_designs

    with pytest.raises(ValueError, match="replicates"):
        rank_designs(
            _synthetic_panel(),
            estimator=Synth(),
            unit="loc",
            time="t",
            outcome="y",
            candidates=[{"u1"}],
            durations=[3],
            lookback_window=2,
            ns=20,
            permutation_type="iid",
            rng=np.random.default_rng(3),
            replicates=0,
        )


def test_one_replicate_is_the_default_and_tags_replicate_zero() -> None:
    from polars.testing import assert_frame_equal

    default = _rank()
    explicit = _rank(replicates=1)
    assert_frame_equal(default.ranking, explicit.ranking)
    assert default.power("u1", 3).simulations.get_column("replicate").unique().to_list() == [0]


def _hand_built(pvalues, *, rmspe=None, effects=(0.0, 0.1, 0.2)):
    """DesignRanking from chosen p-values: no fit, exact expectations.

    pvalues[candidate][e][w] is the p-value at effect index e, window w;
    rmspe has the same shape or is None (all 1.0). One duration (3).
    """
    import polars as pl
    from augsynth_py import PowerResults, Synth

    from geoexp.selection import DesignRanking, _apply_ranking, _Context, _interp_mde

    results, rows = {}, []
    for key, grid in pvalues.items():
        recs = [
            {
                "duration": 3,
                "effect_size": effects[e],
                "window": w,
                "replicate": 0,
                "pvalue": grid[e][w],
                "rmspe_pre": 1.0 if rmspe is None else rmspe[key][e][w],
            }
            for e in range(len(effects))
            for w in range(len(grid[e]))
        ]
        sims = pl.DataFrame(recs)
        results[key] = PowerResults(simulations=sims, alpha=0.1, effect_type="multiplicative")
        curve = results[key].power_curve().sort("effect_size")
        rows.append(
            {
                "candidate": key,
                "duration": 3,
                "mde": _interp_mde(
                    curve.get_column("effect_size").to_list(),
                    curve.get_column("power").to_list(),
                    target_power=0.8,
                ),
                "empirical_size": float(curve.get_column("power")[0]),
                "rmspe_pre": float(sims.get_column("rmspe_pre").mean()),
            }
        )
    return DesignRanking(
        ranking=_apply_ranking(pl.DataFrame(rows)),
        candidates={k: (k,) for k in pvalues},
        _results=results,
        _context=_Context(
            panel=pl.DataFrame(),
            estimator=Synth(),
            unit="u",
            time="t",
            outcome="y",
            target_power=0.8,
            alpha=0.1,
        ),
    )


def _grid(detect_at_0_1: int, windows: int = 10) -> list[list[float]]:
    """p-values: never at effect 0, `detect_at_0_1` of `windows` at 0.1, always at 0.2."""
    at_0_1 = [0.01] * detect_at_0_1 + [0.5] * (windows - detect_at_0_1)
    return [[0.5] * windows, at_0_1, [0.01] * windows]


def test_stability_has_one_row_per_ranking_row_in_the_same_order() -> None:
    import numpy as np

    res = _hand_built({"a": _grid(9), "b": _grid(5)})
    stab = res.stability(rng=np.random.default_rng(0), n_draws=50)
    assert (
        stab.frame.get_column("candidate").to_list()
        == res.ranking.get_column("candidate").to_list()
    )
    assert stab.frame.get_column("rank").to_list() == res.ranking.get_column("rank").to_list()
    for col in ("rank_low", "rank_high", "p_top", "p_undetectable", "mde_low", "mde_high"):
        assert col in stab.frame.columns
    assert stab.n_draws == 50
    assert stab.pool_size == 10


def test_stability_is_certain_about_a_dominant_design() -> None:
    import numpy as np
    import polars as pl

    # a detects always at 0.1; b never reaches target power at any effect
    res = _hand_built({"a": _grid(10), "b": [[0.5] * 10, [0.5] * 10, [0.5] * 10]})
    stab = res.stability(rng=np.random.default_rng(0), n_draws=100, top=1)
    a = stab.frame.filter(pl.col("candidate") == "a").row(0, named=True)
    b = stab.frame.filter(pl.col("candidate") == "b").row(0, named=True)
    assert (a["rank_low"], a["rank_high"], a["p_top"], a["p_undetectable"]) == (1, 1, 1.0, 0.0)
    assert a["mde_low"] == a["mde_high"] == pytest.approx(0.1)
    assert b["p_undetectable"] == 1.0
    assert b["mde_low"] is None
    assert b["mde_high"] is None


def test_stability_intervals_have_zero_width_when_every_window_agrees() -> None:
    import numpy as np

    res = _hand_built({"a": _grid(10), "b": _grid(0)})
    stab = res.stability(rng=np.random.default_rng(0), n_draws=50)
    assert (stab.frame.get_column("rank_low") == stab.frame.get_column("rank")).all()
    assert (stab.frame.get_column("rank_high") == stab.frame.get_column("rank")).all()


def test_stability_recomputes_every_rank_key_per_draw() -> None:
    import numpy as np
    import polars as pl

    # identical p-values; b carries one bad-fit window, so b ranks 2 on the
    # point estimate but ties a (shared rank 1) in draws that skip that window.
    bad = [[1.0] * 10, [1.0] * 10, [1.0] * 10]
    bad[0][0] = 50.0
    res = _hand_built({"a": _grid(9), "b": _grid(9)}, rmspe={"a": [[1.0] * 10] * 3, "b": bad})
    assert res.ranking.filter(pl.col("candidate") == "b").get_column("rank").item() == 2
    stab = res.stability(rng=np.random.default_rng(0), n_draws=200, top=1)
    p_top_b = stab.frame.filter(pl.col("candidate") == "b").get_column("p_top").item()
    assert 0.0 < p_top_b < 1.0  # ~ (0.9)**10 = 0.35


def test_stability_is_reproducible_for_a_given_rng() -> None:
    import numpy as np
    from polars.testing import assert_frame_equal

    res = _hand_built({"a": _grid(8), "b": _grid(7), "c": _grid(6)})
    one = res.stability(rng=np.random.default_rng(5), n_draws=100).frame
    two = res.stability(rng=np.random.default_rng(5), n_draws=100).frame
    assert_frame_equal(one, two)


def test_stability_rejects_a_pool_of_one_and_warns_on_a_small_pool() -> None:
    import numpy as np

    with pytest.raises(ValueError, match="one window"):
        _hand_built({"a": _grid(1, windows=1)}).stability(rng=np.random.default_rng(0))
    with pytest.warns(UserWarning, match="pool"):
        _hand_built({"a": _grid(3, windows=3)}).stability(rng=np.random.default_rng(0), n_draws=5)


@pytest.mark.parametrize("kwargs", [{"n_draws": 0}, {"level": 1.0}, {"level": 0.0}, {"top": 0}])
def test_stability_validates_its_arguments(kwargs) -> None:
    import numpy as np

    with pytest.raises(ValueError):
        _hand_built({"a": _grid(9)}).stability(rng=np.random.default_rng(0), **kwargs)


def test_stability_needs_a_ranking_built_by_rank_designs() -> None:
    import numpy as np
    import polars as pl

    from geoexp.selection import DesignRanking

    bare = DesignRanking(
        ranking=pl.DataFrame({"candidate": ["a"], "duration": [3], "rank": [1]}),
        candidates={"a": ("a",)},
    )
    with pytest.raises(ValueError, match="rank_designs"):
        bare.stability(rng=np.random.default_rng(0))


def test_stability_refuses_an_empty_ranking_with_a_clear_message() -> None:
    import dataclasses

    import numpy as np

    res = _hand_built({"a": _grid(9)})
    empty = dataclasses.replace(res, ranking=res.ranking.head(0))
    with pytest.raises(ValueError, match="no designs"):
        empty.stability(rng=np.random.default_rng(0))


def test_bootstrap_with_unit_weights_reproduces_the_point_ranking() -> None:
    # The per-draw columns must be the same quantities _summarize reports:
    # a draw that takes every placebo test exactly once is the point ranking.
    import numpy as np
    import polars as pl
    from polars.testing import assert_frame_equal

    from geoexp.selection import _rank_weighted_draws, _simulation_pool

    res = _rank(candidates=[{"u1"}, {"u2", "u3"}, {"u4"}], lookback_window=3, durations=[3, 4])
    pool, pool_size = _simulation_pool(res.ranking, res._results, alpha=0.1)
    unit = pl.DataFrame(
        {
            "draw": np.zeros(pool_size, dtype=np.int64),
            "pool_idx": np.arange(pool_size, dtype=np.int64),
            "w": np.ones(pool_size, dtype=np.int64),
        }
    )
    draw = _rank_weighted_draws(unit, pool, target_power=0.8)
    cols = ["candidate", "duration", "mde", "empirical_size", "rmspe_pre", "rank"]
    assert_frame_equal(
        draw.select(cols).sort(["candidate", "duration"]),
        res.ranking.select(cols).sort(["candidate", "duration"]),
        check_dtypes=False,
    )


def test_stability_ranks_only_the_budgeted_set() -> None:
    import numpy as np

    res = _rank(cpic=2.0, lookback_window=3)
    full = res.ranking
    cut = res.within_budget(float(full.get_column("investment_mde").min()))
    assert cut.ranking.height < full.height
    with pytest.warns(UserWarning, match="pool"):
        stab = cut.stability(rng=np.random.default_rng(0), n_draws=20)
    assert stab.frame.height == cut.ranking.height
    assert set(stab.frame.get_column("candidate")) == set(cut.ranking.get_column("candidate"))


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


def test_rank_designs_treatment_window_works_on_a_date_time_column() -> None:
    # prepare_panel keeps real dates by default and augsynth-py accepts them, so
    # the treatment window must be "the last `duration` periods", not integer
    # arithmetic on the time column.
    import datetime as dt

    import polars as pl

    base = _rank(cpic=1.0).ranking
    dated = _synthetic_panel().with_columns(
        pl.lit(dt.date(2021, 1, 1)).add(pl.duration(days=pl.col("t"))).alias("t")
    )
    import numpy as np
    from augsynth_py import Synth

    from geoexp.selection import rank_designs

    with pytest.warns(UserWarning, match="lookback_window"):
        res = rank_designs(
            dated,
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
            cpic=1.0,
        )

    cols = ["candidate", "proportion_total_y", "investment"]
    assert res.ranking.sort("candidate").select(cols).equals(base.sort("candidate").select(cols))


def test_investment_mde_prices_the_grid_mde_on_the_treatment_window() -> None:
    # investment is cpic x target_effect_size x sum(Y_treated); investment_mde
    # swaps target_effect_size for mde_grid --- exactly GeoLift's Investment.
    import polars as pl

    r = _rank(cpic=2.0, target_effect_size=0.1).ranking
    treated_y = pl.col("investment") / (2.0 * 0.1)
    expected = (
        pl.when(pl.col("mde_grid").is_null())
        .then(None)
        .otherwise(2.0 * pl.col("mde_grid") * treated_y)
    )
    diff = r.select((pl.col("investment_mde") - expected).abs().max()).item()
    assert diff is None or diff < 1e-9
    assert r.get_column("investment_mde").null_count() == r.get_column("mde_grid").null_count()


def test_investment_mde_is_absent_without_cpic() -> None:
    assert "investment_mde" not in _rank().ranking.columns


def test_budget_requires_cpic() -> None:
    import numpy as np
    from augsynth_py import Synth

    from geoexp.selection import rank_designs

    with pytest.raises(ValueError, match="cpic"):
        rank_designs(
            _synthetic_panel(),
            estimator=Synth(),
            unit="loc",
            time="t",
            outcome="y",
            candidates=[{"u1"}],
            durations=[3],
            rng=np.random.default_rng(0),
            budget=100.0,
        )


def _budgets():
    """Full ranking with cpic, plus a budget that keeps some rows but not all."""
    full = _rank(cpic=1.0, durations=[3, 5])
    costs = sorted(c for c in full.ranking.get_column("investment_mde").to_list() if c is not None)
    assert len(costs) >= 2, "fixture must price at least two designs"
    return full, costs


def test_budget_keeps_only_designs_whose_mde_cost_fits_and_reranks() -> None:
    _full, costs = _budgets()
    budget = costs[0]
    res = _rank(cpic=1.0, durations=[3, 5], budget=budget)

    kept = res.ranking
    assert kept.height == sum(c <= budget for c in costs)
    assert (kept.get_column("investment_mde") <= budget).all()
    assert kept.get_column("rank").min() == 1
    assert res.budget == budget


def test_within_budget_can_loosen_as_well_as_tighten() -> None:
    full, costs = _budgets()
    tight = full.within_budget(costs[0])
    loose = tight.within_budget(costs[-1])

    assert tight.ranking.height < loose.ranking.height
    assert loose.ranking.height == len(costs)


def test_within_budget_warns_with_the_cheapest_cost_when_nothing_fits() -> None:
    full, costs = _budgets()
    with pytest.warns(UserWarning, match="cheapest"):
        res = full.within_budget(costs[0] / 2)
    assert res.ranking.height == 0


def test_power_results_survive_the_budget_filter() -> None:
    from augsynth_py import PowerResults

    full, costs = _budgets()
    with pytest.warns(UserWarning, match="cheapest"):
        res = full.within_budget(costs[0] / 2)
    assert isinstance(res.power("u1", 3), PowerResults)


def test_within_budget_requires_a_ranking_priced_with_cpic() -> None:
    with pytest.raises(ValueError, match="cpic"):
        _rank().within_budget(100.0)


def test_investment_averages_treated_outcome_over_the_lookback_windows() -> None:
    # GeoLift prices a design at the mean, over the lookback windows, of the
    # treated outcome summed over each simulated test window (windows ending at
    # T, T-1, ..., T-L+1). Measured exact to the cent in the characterization.
    import polars as pl

    panel = _synthetic_panel()
    duration, lookback = 3, 2
    treated = panel.filter(pl.col("loc") == "u1")
    sums = [
        treated.filter(pl.col("t").is_between(end - duration + 1, end)).get_column("y").sum()
        for end in (30, 29)
    ]
    expected_y = sum(sums) / lookback

    row = (
        _rank(cpic=1.0, target_effect_size=0.1, lookback_window=lookback)
        .ranking.filter(pl.col("candidate") == "u1")
        .row(0, named=True)
    )
    assert row["investment"] == pytest.approx(0.1 * expected_y)
    if row["mde_grid"] is not None:
        assert row["investment_mde"] == pytest.approx(row["mde_grid"] * expected_y)


@pytest.mark.parametrize("budget", [float("nan"), float("inf"), -1.0])
def test_within_budget_rejects_a_budget_that_is_not_a_finite_non_negative_number(budget) -> None:
    full, _costs = _budgets()
    with pytest.raises(ValueError, match="budget"):
        full.within_budget(budget)
