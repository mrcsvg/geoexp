"""Panel preparation: from a raw export to the long panel ``rank_designs`` takes.

:func:`prepare_panel` does what R GeoLift's ``GeoDataRead`` does --- parse
dates, lower-case unit names, sum duplicate rows, drop units that are not
observed in every period --- plus temporal aggregation, and says out loud what
it dropped. The ``GeoDataRead`` behaviour was measured from its outputs, not
read from its source (clean-room rule); see
``docs/geolift-oracle-characterization.md`` and the parity test in
``tests/validation_against_r/test_panel.py``.
"""

from __future__ import annotations

import datetime as dt
import warnings
from typing import Literal

import polars as pl

OnDuplicate = Literal["sum", "raise"]
OnIncomplete = Literal["drop", "fill_zero", "raise"]

#: How many offending units an error or warning names before summarizing.
_SHOW = 10


def prepare_panel(
    data: pl.DataFrame,
    *,
    unit: str,
    time: str,
    outcome: str,
    date_format: str | None = None,
    normalize_units: bool = True,
    on_duplicate: OnDuplicate = "sum",
    on_incomplete: OnIncomplete = "drop",
    every: str | None = None,
    time_index: bool = False,
) -> pl.DataFrame:
    """Turn a raw unit x time export into a balanced long panel.

    Steps, in order: parse the time column; normalize unit names; resolve
    duplicate unit x period rows; aggregate over time (``every``); handle units
    not observed in every period; optionally replace dates by an integer index.

    Parameters
    ----------
    data : polars.DataFrame
        Raw data, one row per unit x period (duplicates allowed).
    unit, time, outcome : str
        Column names. Other columns are discarded.
    date_format : str, optional
        ``strftime`` format for a string time column, e.g. ``"%Y-%m-%d"``.
        When omitted, polars infers it. Non-string time columns are kept.
    normalize_units : bool, default True
        Strip whitespace and lower-case string unit names, as ``GeoDataRead``
        does. Names that collapse together are merged, with a warning.
    on_duplicate : {"sum", "raise"}, default "sum"
        Duplicate unit x period rows are summed (``GeoDataRead`` behaviour) or
        rejected. A null among duplicates keeps the sum null.
    on_incomplete : {"drop", "fill_zero", "raise"}, default "drop"
        A unit missing a period, or with a null outcome, is dropped with a
        warning naming it and the reason (``GeoDataRead`` drops it silently);
        or completed with zeros --- only when absence genuinely means zero; or
        rejected. The periods expected are every period observed for any unit.
    every : str, optional
        Aggregate over time by summing, with a polars duration string such as
        ``"1w"`` or ``"1mo"``; periods are labelled by their start. Needs a date
        or datetime time column. Partial periods at either edge are dropped
        with a warning: a truncated last week would bias the last point.
    time_index : bool, default False
        Replace the time column by an ``Int64`` index ``1..T``, the shape
        ``GeoDataRead`` returns. Real dates are otherwise kept; ``rank_designs``
        accepts them.

    Returns
    -------
    polars.DataFrame
        Exactly ``[unit, time, outcome]``, sorted by unit then time.

    Raises
    ------
    ValueError
        On a missing column, unparseable dates, duplicates under
        ``on_duplicate="raise"``, incomplete units under
        ``on_incomplete="raise"``, ``every`` on a non-temporal column, or when
        no complete unit is left.

    Warns
    -----
    UserWarning
        When unit names merge, partial edge periods are dropped, or incomplete
        units are dropped.

    Notes
    -----
    Parity with ``GeoDataRead`` covers everything except ``every`` and
    ``on_incomplete="fill_zero"``, which it has no counterpart for.
    """
    missing = [c for c in (unit, time, outcome) if c not in data.columns]
    if missing:
        raise ValueError(f"columns not found in data: {missing}")
    frame = data.select(unit, time, outcome)

    frame = _parse_time(frame, time=time, date_format=date_format)
    null_time = frame.get_column(time).null_count()
    if null_time:
        raise ValueError(
            f"{null_time} row(s) have a null {time!r}; drop or date them first --- a row "
            "with no period cannot be placed in the panel"
        )
    if normalize_units and frame.schema[unit] == pl.String:
        frame = _normalize_units(frame, unit=unit)
    frame = _resolve_duplicates(frame, unit=unit, time=time, outcome=outcome, how=on_duplicate)
    if every is not None:
        frame = _aggregate_time(frame, unit=unit, time=time, outcome=outcome, every=every)
    frame = _complete(frame, unit=unit, time=time, outcome=outcome, how=on_incomplete)
    if time_index:
        frame = frame.with_columns(pl.col(time).rank("dense").cast(pl.Int64))
    return frame.sort(unit, time)


#: strftime directives that carry a time of day.
_TIME_DIRECTIVES = ("%H", "%I", "%M", "%S", "%f", "%p", "%T", "%R", "%X", "%c", "%z", "%Z", "%s")


def _parse_time(frame: pl.DataFrame, *, time: str, date_format: str | None) -> pl.DataFrame:
    """Parse a string time column to ``Date``, or ``Datetime`` if it has a time of day.

    Never truncates: timestamps parsed as dates would turn distinct hours into
    duplicates and get summed. Without ``date_format``, values are parsed as
    datetimes and narrowed to dates only when every one of them is midnight.
    """
    if frame.schema[time] != pl.String:
        return frame
    shown = date_format if date_format is not None else "an inferred format"
    try:
        if date_format is not None:
            if any(d in date_format for d in _TIME_DIRECTIVES):
                return frame.with_columns(pl.col(time).str.to_datetime(date_format, strict=True))
            return frame.with_columns(pl.col(time).str.to_date(date_format, strict=True))
        parsed = frame.with_columns(pl.col(time).str.to_datetime(strict=True))
    except pl.exceptions.PolarsError as err:
        raise ValueError(f"could not parse {time!r} with {shown}: {err}") from None
    column = parsed.get_column(time)
    if (column.dt.truncate("1d") == column).all():
        return parsed.with_columns(pl.col(time).dt.date())
    return parsed


def _normalize_units(frame: pl.DataFrame, *, unit: str) -> pl.DataFrame:
    normalized = pl.col(unit).str.strip_chars().str.to_lowercase()
    variants = (
        frame.select(original=pl.col(unit), normalized=normalized)
        .unique()
        .group_by("normalized")
        .agg(pl.col("original").sort())
        .filter(pl.col("original").list.len() > 1)
        .sort("normalized")
    )
    if variants.height:
        shown = "; ".join(
            f"{sorted(row['original'])} -> {row['normalized']!r}"
            for row in variants.head(_SHOW).iter_rows(named=True)
        )
        warnings.warn(
            f"{variants.height} unit name(s) merged by normalization: {shown}. "
            "Their rows are summed; pass normalize_units=False to keep them apart.",
            UserWarning,
            stacklevel=3,
        )
    return frame.with_columns(normalized)


def _null_propagating_sum(outcome: str) -> pl.Expr:
    return (
        pl.when(pl.col(outcome).null_count() > 0)
        .then(None)
        .otherwise(pl.col(outcome).sum())
        .alias(outcome)
    )


def _resolve_duplicates(
    frame: pl.DataFrame, *, unit: str, time: str, outcome: str, how: OnDuplicate
) -> pl.DataFrame:
    dupes = frame.group_by(unit, time).len().filter(pl.col("len") > 1)
    if not dupes.height:
        return frame
    if how == "raise":
        shown = dupes.sort(unit, time).head(_SHOW).select(unit, time).rows()
        raise ValueError(f"{dupes.height} duplicate {unit} x {time} key(s), e.g. {shown}")
    return frame.group_by(unit, time).agg(_null_propagating_sum(outcome))


def _aggregate_time(
    frame: pl.DataFrame, *, unit: str, time: str, outcome: str, every: str
) -> pl.DataFrame:
    if not frame.schema[time].is_temporal():
        raise ValueError(f"every={every!r} needs a date or datetime {time!r} column")
    bucket = pl.col(time).dt.truncate(every).alias("_bucket")
    # Edge periods are judged against calendar bounds, not against each other:
    # a whole February is shorter than January without being partial. The
    # first period is partial when the data's own frequency leaves room for an
    # earlier observation inside it; the last, when it leaves room for a later
    # one. That needs the base frequency, inferred from the data.
    periods = frame.get_column(time).unique().sort()
    base = _base_frequency(periods)
    first, last = periods[0], periods[-1]
    bounds = pl.select(
        first_start=pl.lit(first).dt.truncate(every),
        last_start=pl.lit(last).dt.truncate(every),
        next_start=pl.lit(last).dt.truncate(every).dt.offset_by(every),
        before_first=pl.lit(first).dt.offset_by(f"-{base}"),
        after_last=pl.lit(last).dt.offset_by(base),
    ).row(0, named=True)
    partial: list[object] = []
    if bounds["before_first"] >= bounds["first_start"]:
        partial.append(bounds["first_start"])
    if bounds["after_last"] < bounds["next_start"] and bounds["last_start"] not in partial:
        partial.append(bounds["last_start"])
    if partial:
        warnings.warn(
            f"dropped {len(partial)} partial {every} period(s) at the edges starting "
            f"{[str(p) for p in partial]}: summing an incomplete period biases it low",
            UserWarning,
            stacklevel=3,
        )
    out = (
        frame.with_columns(bucket)
        .filter(~pl.col("_bucket").is_in(partial))
        .group_by(unit, "_bucket")
        .agg(_null_propagating_sum(outcome))
        .rename({"_bucket": time})
    )
    if not out.height:
        raise ValueError(
            f"every={every!r} leaves no complete period: the data covers only partial ones"
        )
    return out


#: Candidate base frequencies, finest first.
_FREQUENCIES = ("1h", "1d", "1w", "1mo", "1q", "1y")


def _base_frequency(periods: pl.Series) -> str:
    """Infer the calendar frequency most consecutive periods are one step apart at.

    Calendar-aware on purpose: monthly data is 28 to 31 days apart, and only
    ``"1mo"`` describes every gap. Falls back to the smallest gap, as a fixed
    duration, for irregular data or a single period (``"1d"``).
    """
    if periods.len() < 2:
        return "1d"
    pairs = pl.DataFrame({"a": periods[:-1], "b": periods[1:]})
    for freq in _FREQUENCIES:
        hits = pairs.select((pl.col("a").dt.offset_by(freq) == pl.col("b")).mean()).item()
        if hits is not None and hits >= 0.5:
            return freq
    gap = periods.diff().min()
    if not isinstance(gap, dt.timedelta):
        return "1d"
    return f"{int(gap.total_seconds() * 1_000_000)}us"


def _complete(
    frame: pl.DataFrame, *, unit: str, time: str, outcome: str, how: OnIncomplete
) -> pl.DataFrame:
    n_periods = frame.get_column(time).n_unique()
    status = frame.group_by(unit).agg(
        n_missing=n_periods - pl.col(time).n_unique(),
        n_null=pl.col(outcome).null_count(),
    )
    bad = status.filter((pl.col("n_missing") > 0) | (pl.col("n_null") > 0)).sort(unit)
    if not bad.height:
        return frame

    reasons = [
        f"{row[unit]} ("
        + ", ".join(
            part
            for part in (
                f"{row['n_missing']} missing period(s)" if row["n_missing"] else "",
                f"{row['n_null']} null outcome(s)" if row["n_null"] else "",
            )
            if part
        )
        + ")"
        for row in bad.head(_SHOW).iter_rows(named=True)
    ]
    summary = "; ".join(reasons) + (
        f"; and {bad.height - _SHOW} more" if bad.height > _SHOW else ""
    )

    if how == "raise":
        raise ValueError(f"{bad.height} incomplete unit(s): {summary}")
    if how == "fill_zero":
        grid = frame.select(unit).unique().join(frame.select(time).unique(), how="cross")
        return grid.join(frame, on=[unit, time], how="left").with_columns(
            pl.col(outcome).fill_null(0)
        )
    if bad.height == status.height:
        raise ValueError(f"no complete unit is left after dropping incomplete ones: {summary}")
    warnings.warn(
        f"dropped {bad.height} incomplete unit(s): {summary}. "
        "Pass on_incomplete='fill_zero' if absence means zero.",
        UserWarning,
        stacklevel=3,
    )
    return frame.filter(~pl.col(unit).is_in(bad.get_column(unit).to_list()))
