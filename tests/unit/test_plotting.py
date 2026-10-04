"""Unit tests for the design plots on DesignRanking (fit_path, plot_power, plot_fit)."""

from __future__ import annotations

import sys

import numpy as np
import polars as pl
import pytest
from augsynth_py import Synth
from test_selection import _synthetic_panel

from geoexp import rank_designs


def _ranked():
    with pytest.warns(UserWarning, match="lookback_window"):
        return rank_designs(
            _synthetic_panel(),
            estimator=Synth(),
            unit="loc",
            time="t",
            outcome="y",
            candidates=[{"u2", "u3"}],
            durations=[3, 5],
            effect_sizes=(0.0, 0.1, 0.2),
            lookback_window=2,
            ns=50,
            permutation_type="iid",
            rng=np.random.default_rng(11),
        )


def test_fit_path_marks_the_last_duration_periods_as_the_test_window() -> None:
    path = _ranked().fit_path("u2, u3", 5)

    assert path.columns == ["t", "actual", "synthetic", "gap", "window"]
    assert path.height == 30
    assert path.filter(pl.col("window") == "test").get_column("t").to_list() == [26, 27, 28, 29, 30]
    diff = path.select((pl.col("actual") - pl.col("synthetic") - pl.col("gap")).abs().max())
    assert diff.item() < 1e-9


def test_fit_path_does_not_mutate_the_prototype_estimator() -> None:
    proto = Synth()
    with pytest.warns(UserWarning, match="lookback_window"):
        res = rank_designs(
            _synthetic_panel(),
            estimator=proto,
            unit="loc",
            time="t",
            outcome="y",
            candidates=[{"u1"}],
            durations=[3],
            effect_sizes=(0.0, 0.1),
            lookback_window=2,
            ns=20,
            permutation_type="iid",
            rng=np.random.default_rng(1),
        )
    res.fit_path("u1", 3)
    assert not hasattr(proto, "synthetic_")


def test_fit_path_rejects_an_unevaluated_duration() -> None:
    with pytest.raises(ValueError, match="duration"):
        _ranked().fit_path("u2, u3", 7)


@pytest.fixture
def plt():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    yield plt
    plt.close("all")


def test_plot_power_draws_one_curve_per_duration_from_the_power_results(plt) -> None:
    res = _ranked()
    ax = res.plot_power("u2, u3")

    curve = res.power("u2, u3", 3).power_curve()
    labelled = {line.get_label(): line for line in ax.get_lines()}
    for duration in (3, 5):
        line = labelled[f"{duration} periods"]
        expected = curve.filter(pl.col("duration") == duration).sort("effect_size")
        np.testing.assert_allclose(line.get_xdata(), expected.get_column("effect_size"))
        np.testing.assert_allclose(line.get_ydata(), expected.get_column("power"))
    assert ax.get_legend() is not None


def test_plot_power_draws_on_a_given_axes(plt) -> None:
    _, ax = plt.subplots()
    assert _ranked().plot_power("u2, u3", ax=ax) is ax


def test_plot_fit_draws_actual_and_synthetic_from_fit_path(plt) -> None:
    res = _ranked()
    ax = res.plot_fit("u2, u3", 5)

    path = res.fit_path("u2, u3", 5)
    labelled = {line.get_label(): line for line in ax.get_lines()}
    np.testing.assert_allclose(labelled["actual"].get_ydata(), path.get_column("actual"))
    np.testing.assert_allclose(labelled["synthetic"].get_ydata(), path.get_column("synthetic"))


def test_plots_explain_how_to_install_matplotlib(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", None)
    with pytest.raises(ImportError, match=r"geoexp\[plot\]"):
        _ranked().plot_power("u2, u3")
