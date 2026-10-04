"""Matplotlib drawing for :class:`geoexp.DesignRanking` --- optional extra.

matplotlib is imported inside each call, never at module import, so the core
package stays usable without it. Install with ``pip install 'geoexp[plot]'``.

Colours are the first categorical slots of a colour-vision-deficiency-checked
palette, assigned in fixed order and never cycled; identity is always carried
by a legend as well, and the numbers behind every plot are available as frames
(``DesignRanking.fit_path``, ``PowerResults.power_curve``).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import polars as pl

if TYPE_CHECKING:
    from matplotlib.axes import Axes

#: Categorical slots in fixed order (blue, orange, aqua, yellow, magenta,
#: green, violet, red). Validated for CVD separation on a light surface.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
#: Recessive ink for references and annotations; fill for the test window.
MUTED = "#6b6a63"
WINDOW_FILL = "#e9e8e2"


def _axes(ax: Axes | None) -> Axes:
    try:
        import matplotlib.pyplot as plt
    except ImportError as err:
        raise ImportError(
            "geoexp plots need matplotlib; install it with: pip install 'geoexp[plot]'"
        ) from err
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    return ax


def _style(ax: Axes) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(True, color=WINDOW_FILL, linewidth=0.8)
    ax.set_axisbelow(True)


def power(
    curve: pl.DataFrame,
    *,
    durations: Sequence[int],
    mdes: dict[int, float | None],
    target_power: float,
    title: str,
    ax: Axes | None,
) -> Axes:
    """Power vs effect size, one line per duration."""
    if len(durations) > len(SERIES):
        raise ValueError(
            f"plot_power draws at most {len(SERIES)} durations; pass durations= to choose"
        )
    ax = _axes(ax)
    _style(ax)
    ax.axhline(target_power, color=MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
    ax.annotate(
        f"target power {target_power:g}",
        xy=(0, target_power),
        xycoords=("axes fraction", "data"),
        xytext=(2, 3),
        textcoords="offset points",
        color=MUTED,
        fontsize=8,
    )
    for color, duration in zip(SERIES, durations, strict=False):
        rows = curve.filter(pl.col("duration") == duration).sort("effect_size")
        ax.plot(
            rows.get_column("effect_size").to_numpy(),
            rows.get_column("power").to_numpy(),
            color=color,
            linewidth=2,
            marker="o",
            markersize=5,
            label=f"{duration} periods",
            zorder=3,
        )
        mde = mdes.get(duration)
        if mde is not None:
            ax.scatter(
                [mde],
                [target_power],
                s=64,
                color=color,
                edgecolors="white",
                linewidths=2,
                zorder=4,
            )
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("effect size")
    ax.set_ylabel("power")
    ax.set_title(title, loc="left")
    ax.legend(frameon=False, title="duration (dot = MDE)", title_fontsize=8, fontsize=8)
    return ax


def fit(
    path: pl.DataFrame,
    *,
    time: str,
    outcome: str,
    rmspe_pre: float | None,
    title: str,
    ax: Axes | None,
) -> Axes:
    """Actual vs synthetic outcome, with the simulated test window shaded."""
    ax = _axes(ax)
    _style(ax)
    times: Any = path.get_column(time).to_numpy()
    test = path.filter(pl.col("window") == "test").get_column(time)
    if test.len():
        start: Any = test.min()
        end: Any = test.max()
        ax.axvspan(start, end, color=WINDOW_FILL, zorder=0, label="test window")
    ax.plot(
        times,
        path.get_column("actual").to_numpy(),
        color=SERIES[0],
        linewidth=2,
        label="actual",
        zorder=3,
    )
    ax.plot(
        times,
        path.get_column("synthetic").to_numpy(),
        color=SERIES[1],
        linewidth=2,
        linestyle=(0, (5, 2)),
        label="synthetic",
        zorder=3,
    )
    if path.schema[time].is_temporal():
        import matplotlib.dates as mdates

        locator = mdates.AutoDateLocator()  # type: ignore[no-untyped-call]
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))  # type: ignore[no-untyped-call]
    ax.set_xlabel(time)
    ax.set_ylabel(outcome)
    subtitle = "" if rmspe_pre is None else f"   rmspe_pre {rmspe_pre:.4g}"
    ax.set_title(f"{title}{subtitle}", loc="left")
    # One row under the axes: series cross the whole plot, so no corner is free.
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper left", bbox_to_anchor=(0, -0.16))
    return ax
