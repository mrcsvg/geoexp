"""Polars <-> R data.frame conversion for the parity suite.

Adapted from the helper in mrcsvg/augsynth-py#22 (author: GuiMarthe). Keeping
the bridge Polars-native means the validation extra needs neither pandas nor
pyarrow: rpy2 alone is enough.
"""

from __future__ import annotations

from typing import Any

import polars as pl


def from_r_data_frame(frame: Any) -> pl.DataFrame:
    """Convert an R ``data.frame`` to Polars without routing through pandas."""
    import rpy2.robjects as ro

    columns: dict[str, list[object]] = {}
    for name in frame.names:
        column = frame.rx2(name)
        if set(column.rclass) & {"factor", "Date", "POSIXct"}:
            column = ro.r["as.character"](column)
        columns[str(name)] = list(column)
    return pl.DataFrame(columns)
