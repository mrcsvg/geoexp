"""Unit tests for geoexp.panel.prepare_panel."""

from __future__ import annotations

import datetime as dt

import polars as pl
import pytest

from geoexp import prepare_panel


def _raw(**overrides: object) -> pl.DataFrame:
    """Two units x four daily periods, dates as strings, like a CSV export."""
    frame = pl.DataFrame(
        {
            "city": ["a", "a", "a", "a", "b", "b", "b", "b"],
            "date": ["2021-01-0" + str(d) for d in (1, 2, 3, 4)] * 2,
            "sales": [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0],
        }
    )
    return frame.with_columns(**overrides) if overrides else frame


def _prep(frame: pl.DataFrame, **kwargs: object) -> pl.DataFrame:
    return prepare_panel(frame, unit="city", time="date", outcome="sales", **kwargs)


def test_parses_string_dates_and_returns_exactly_the_three_columns_sorted() -> None:
    out = _prep(_raw().reverse(), date_format="%Y-%m-%d")

    assert out.columns == ["city", "date", "sales"]
    assert out.schema["date"] == pl.Date
    assert out.get_column("date").to_list()[:2] == [dt.date(2021, 1, 1), dt.date(2021, 1, 2)]
    assert out.get_column("city").to_list() == ["a"] * 4 + ["b"] * 4


def test_rejects_missing_columns() -> None:
    with pytest.raises(ValueError, match="sales"):
        prepare_panel(_raw().drop("sales"), unit="city", time="date", outcome="sales")


def test_rejects_unparseable_dates_with_the_format_in_the_message() -> None:
    with pytest.raises(ValueError, match="%d/%m/%Y"):
        _prep(_raw(), date_format="%d/%m/%Y")


def test_keeps_integer_time_as_is() -> None:
    frame = _raw().with_columns(date=pl.Series([1, 2, 3, 4] * 2))
    out = _prep(frame)
    assert out.get_column("date").to_list()[:4] == [1, 2, 3, 4]


def test_normalizes_unit_names_and_warns_when_names_merge() -> None:
    frame = _raw(city=pl.Series(["A ", "A ", "A ", "A ", "b", "b", "b", "b"]))
    out = _prep(frame, date_format="%Y-%m-%d")
    assert set(out.get_column("city")) == {"a", "b"}

    merged = pl.concat(
        [_raw(), _raw().filter(pl.col("city") == "a").with_columns(city=pl.lit("A"))]
    )
    with pytest.warns(UserWarning, match="merged"):
        out = _prep(merged, date_format="%Y-%m-%d")
    assert out.filter(pl.col("city") == "a").get_column("sales").to_list() == [2.0, 4.0, 6.0, 8.0]


def test_normalize_units_can_be_turned_off() -> None:
    frame = _raw(city=pl.Series(["A"] * 4 + ["b"] * 4))
    out = _prep(frame, date_format="%Y-%m-%d", normalize_units=False)
    assert set(out.get_column("city")) == {"A", "b"}


def test_sums_duplicate_rows_by_default() -> None:
    frame = pl.concat([_raw(), _raw().head(1)])
    out = _prep(frame, date_format="%Y-%m-%d")
    assert out.height == 8
    assert out.get_column("sales")[0] == 2.0


def test_on_duplicate_raise_names_the_duplicated_key() -> None:
    frame = pl.concat([_raw(), _raw().head(1)])
    with pytest.raises(ValueError, match="duplicate"):
        _prep(frame, date_format="%Y-%m-%d", on_duplicate="raise")


def test_drops_a_unit_missing_a_period_and_says_why() -> None:
    frame = _raw().filter(~((pl.col("city") == "b") & (pl.col("date") == "2021-01-03")))
    with pytest.warns(UserWarning, match=r"b.*missing"):
        out = _prep(frame, date_format="%Y-%m-%d")
    assert set(out.get_column("city")) == {"a"}


def test_drops_a_unit_with_a_null_outcome_even_when_a_duplicate_has_a_value() -> None:
    # A null must not vanish into a sum: GeoDataRead drops the unit.
    frame = pl.concat(
        [
            _raw().with_columns(
                sales=pl.when((pl.col("city") == "b") & (pl.col("date") == "2021-01-02"))
                .then(None)
                .otherwise(pl.col("sales"))
            ),
            _raw().filter((pl.col("city") == "b") & (pl.col("date") == "2021-01-02")),
        ]
    )
    with pytest.warns(UserWarning, match=r"b.*null"):
        out = _prep(frame, date_format="%Y-%m-%d")
    assert set(out.get_column("city")) == {"a"}


def test_fill_zero_completes_the_panel_instead() -> None:
    frame = _raw().filter(~((pl.col("city") == "b") & (pl.col("date") == "2021-01-03")))
    out = _prep(frame, date_format="%Y-%m-%d", on_incomplete="fill_zero")
    assert out.height == 8
    row = out.filter((pl.col("city") == "b") & (pl.col("date") == dt.date(2021, 1, 3)))
    assert row.get_column("sales").item() == 0.0


def test_on_incomplete_raise_lists_the_units() -> None:
    frame = _raw().filter(~((pl.col("city") == "b") & (pl.col("date") == "2021-01-03")))
    with pytest.raises(ValueError, match="b"):
        _prep(frame, date_format="%Y-%m-%d", on_incomplete="raise")


def test_raises_when_no_complete_unit_is_left() -> None:
    frame = (
        _raw()
        .filter(pl.col("date") != "2021-01-03")
        .with_columns(
            date=pl.when(pl.col("city") == "a").then(pl.col("date")).otherwise(pl.lit("2021-01-03"))
        )
    )
    with pytest.raises(ValueError, match="no complete unit"):
        _prep(frame, date_format="%Y-%m-%d")


def test_time_index_replaces_dates_with_integers_from_one() -> None:
    out = _prep(_raw(), date_format="%Y-%m-%d", time_index=True)
    assert out.schema["date"] == pl.Int64
    assert out.get_column("date").to_list()[:4] == [1, 2, 3, 4]


def _daily(days: int, start: dt.date) -> pl.DataFrame:
    dates = [start + dt.timedelta(days=d) for d in range(days)]
    return pl.DataFrame(
        {"city": ["a"] * days + ["b"] * days, "date": dates * 2, "sales": [1.0] * (2 * days)}
    )


def test_every_sums_into_periods_and_drops_partial_edge_periods() -> None:
    # 2021-01-04 is a Monday: 14 days = two whole ISO weeks, plus two stray
    # days on each side that must not become biased half-weeks.
    frame = _daily(18, dt.date(2021, 1, 2))
    with pytest.warns(UserWarning, match="partial"):
        out = _prep(frame, every="1w")
    a = out.filter(pl.col("city") == "a")
    assert a.get_column("date").to_list() == [dt.date(2021, 1, 4), dt.date(2021, 1, 11)]
    assert a.get_column("sales").to_list() == [7.0, 7.0]


def test_every_needs_a_date_column() -> None:
    frame = _raw().with_columns(date=pl.Series([1, 2, 3, 4] * 2))
    with pytest.raises(ValueError, match="every"):
        _prep(frame, every="1w")


def test_every_keeps_whole_calendar_months_of_unequal_length() -> None:
    # A complete February has fewer days than January; that is not partial.
    import warnings

    frame = _daily(59, dt.date(2021, 1, 1))  # Jan 1 .. Feb 28
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = _prep(frame, every="1mo")
    a = out.filter(pl.col("city") == "a")
    assert a.get_column("sales").to_list() == [31.0, 28.0]


def _monthly(months: int) -> pl.DataFrame:
    dates = [dt.date(2024, m, 1) for m in range(1, months + 1)]
    return pl.DataFrame(
        {"city": ["a"] * months + ["b"] * months, "date": dates * 2, "sales": [1.0] * (2 * months)}
    )


@pytest.mark.parametrize(("every", "expected"), [("1mo", [1.0] * 12), ("1q", [3.0] * 4)])
def test_every_keeps_the_last_complete_period_of_monthly_data(every, expected) -> None:
    # Month-start data: the smallest gap (28 days) must not make December, or
    # a whole Q4, look partial.
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = _prep(_monthly(12), every=every)
    assert out.filter(pl.col("city") == "a").get_column("sales").to_list() == expected


def test_every_raises_when_every_period_is_partial() -> None:
    frame = _daily(2, dt.date(2024, 1, 3))  # Wednesday and Thursday
    with pytest.raises(ValueError, match="partial"), pytest.warns(UserWarning):
        _prep(frame, every="1w")


def test_string_timestamps_keep_their_time_of_day() -> None:
    frame = pl.DataFrame(
        {
            "city": ["a"] * 4,
            "date": [
                "2024-01-01 00:00",
                "2024-01-01 01:00",
                "2024-01-02 00:00",
                "2024-01-02 01:00",
            ],
            "sales": [1.0, 2.0, 3.0, 4.0],
        }
    )
    out = _prep(frame, date_format="%Y-%m-%d %H:%M")
    assert out.height == 4
    assert out.schema["date"] == pl.Datetime("us")
    inferred = _prep(frame)
    assert inferred.height == 4


def test_inferred_plain_dates_stay_dates() -> None:
    assert _prep(_raw()).schema["date"] == pl.Date


def test_rejects_rows_with_a_null_time() -> None:
    frame = pl.concat(
        [
            _raw(),
            pl.DataFrame({"city": ["c"], "date": [None], "sales": [5.0]}, schema=_raw().schema),
        ]
    )
    with pytest.raises(ValueError, match="null"):
        _prep(frame, date_format="%Y-%m-%d")
