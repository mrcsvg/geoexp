"""Unit tests for geoexp.hierarchy (Hierarchy, aggregate_panel)."""

from __future__ import annotations

import polars as pl
import pytest

from geoexp import Hierarchy, aggregate_panel
from geoexp.exceptions import HierarchyError


def _stores() -> pl.DataFrame:
    """Four stores -> two districts -> one region, with labels."""
    return pl.DataFrame(
        {
            "store": [1, 2, 3, 4],
            "district": [10, 10, 20, 20],
            "district_name": ["north", "north", "south", "south"],
            "region": ["R1", "R1", "R1", "R1"],
        }
    )


def _hierarchy(**kwargs: object) -> Hierarchy:
    params: dict[str, object] = {
        "levels": ["store", "district", "region"],
        "labels": {"district": "district_name"},
        "name": "stores",
    }
    params.update(kwargs)
    return Hierarchy.from_frame(_stores(), **params)  # type: ignore[arg-type]


def _panel() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "store": [1, 1, 2, 2, 3, 3, 4, 4],
            "t": [1, 2] * 4,
            "sales": [1.0, 2.0, 10.0, 20.0, 100.0, 200.0, 1000.0, 2000.0],
            "visits": [1.0] * 8,
        }
    )


def _agg(panel: pl.DataFrame | None = None, **kwargs: object) -> pl.DataFrame:
    params: dict[str, object] = {
        "unit": "store",
        "time": "t",
        "outcome": "sales",
        "hierarchy": _hierarchy(),
        "to": "district",
    }
    params.update(kwargs)
    return aggregate_panel(_panel() if panel is None else panel, **params)  # type: ignore[arg-type]


# --- Hierarchy -------------------------------------------------------------


def test_from_frame_keeps_levels_and_labels() -> None:
    h = _hierarchy(source="test fixture")
    assert h.levels == ("store", "district", "region")
    assert h.labels == {"district": "district_name"}
    assert h.name == "stores"
    assert h.source == "test fixture"


def test_from_frame_rejects_a_missing_level_or_label_column() -> None:
    with pytest.raises(ValueError, match="zone"):
        _hierarchy(levels=["store", "zone"])
    with pytest.raises(ValueError, match="district_label"):
        _hierarchy(labels={"district": "district_label"})


def test_from_frame_rejects_a_label_for_an_unknown_level() -> None:
    with pytest.raises(ValueError, match="state"):
        _hierarchy(labels={"state": "district_name"})


def test_from_frame_rejects_nulls_in_levels() -> None:
    frame = _stores().with_columns(district=pl.Series([10, None, 20, 20]))
    with pytest.raises(HierarchyError, match="district"):
        Hierarchy.from_frame(frame, levels=["store", "district"], name="x")


def test_from_frame_rejects_a_repeated_finest_unit() -> None:
    frame = pl.concat([_stores(), _stores().head(1)])
    with pytest.raises(HierarchyError, match="store"):
        Hierarchy.from_frame(frame, levels=["store", "district"], name="x")


def test_mapping_is_one_row_per_source_value() -> None:
    m = _hierarchy().mapping("district", "region")
    assert m.sort("district").rows() == [(10, "R1"), (20, "R1")]


def test_mapping_rejects_a_non_functional_pair() -> None:
    # Media markets cross state lines: a market does not map to one state. The
    # hierarchy is still valid; only that direction of aggregation is not.
    frame = pl.DataFrame({"zip": [1, 2, 3], "dma": ["a", "a", "b"], "state": ["x", "y", "y"]})
    h = Hierarchy.from_frame(frame, levels=["zip", "dma", "state"], name="us")
    assert h.mapping("zip", "state").height == 3
    with pytest.raises(HierarchyError, match="dma"):
        h.mapping("dma", "state")


def test_mapping_can_return_labels_for_the_target_level() -> None:
    m = _hierarchy().mapping("store", "district", use_labels=True)
    assert dict(m.rows()) == {1: "north", 2: "north", 3: "south", 4: "south"}


def test_mapping_rejects_unknown_levels() -> None:
    with pytest.raises(ValueError, match="nope"):
        _hierarchy().mapping("store", "nope")


# --- aggregate_panel -------------------------------------------------------


def test_aggregate_panel_sums_to_the_target_level() -> None:
    out = _agg()
    assert out.columns == ["district", "t", "sales"]
    assert out.rows() == [(10, 1, 11.0), (10, 2, 22.0), (20, 1, 1100.0), (20, 2, 2200.0)]


def test_aggregate_panel_sums_several_outcomes() -> None:
    # Rates aggregate as numerator and denominator, never as a mean of rates.
    out = _agg(outcome=["sales", "visits"])
    assert out.columns == ["district", "t", "sales", "visits"]
    assert out.get_column("visits").to_list() == [2.0] * 4


def test_aggregate_panel_uses_labels_when_asked() -> None:
    out = _agg(use_labels=True)
    assert out.get_column("district").unique().sort().to_list() == ["north", "south"]


def test_aggregate_panel_starts_from_any_level() -> None:
    panel = _agg().rename({"district": "d"})
    out = aggregate_panel(
        panel,
        unit="d",
        time="t",
        outcome="sales",
        hierarchy=_hierarchy(),
        from_level="district",
        to="region",
    )
    assert out.rows() == [("R1", 1, 1111.0), ("R1", 2, 2222.0)]


def test_aggregate_panel_raises_on_unmapped_units_by_default() -> None:
    panel = pl.concat([_panel(), _panel().head(1).with_columns(store=pl.lit(99, dtype=pl.Int64))])
    with pytest.raises(ValueError, match="99"):
        _agg(panel)


def test_aggregate_panel_can_drop_unmapped_units_with_a_warning() -> None:
    panel = pl.concat([_panel(), _panel().head(1).with_columns(store=pl.lit(99, dtype=pl.Int64))])
    with pytest.warns(UserWarning, match="99"):
        out = _agg(panel, on_unmapped="drop")
    assert out.height == 4


def test_aggregate_panel_explains_a_key_dtype_mismatch() -> None:
    # Codes read as strings on one side and integers on the other would
    # otherwise join to nothing, silently.
    panel = _panel().with_columns(pl.col("store").cast(pl.String))
    with pytest.raises(ValueError, match="dtype"):
        _agg(panel)


def test_aggregate_panel_rejects_a_non_functional_aggregation() -> None:
    frame = pl.DataFrame({"zip": [1, 2, 3], "dma": ["a", "a", "b"], "state": ["x", "y", "y"]})
    h = Hierarchy.from_frame(frame, levels=["zip", "dma", "state"], name="us")
    panel = pl.DataFrame({"dma": ["a", "b"], "t": [1, 1], "y": [1.0, 2.0]})
    with pytest.raises(HierarchyError):
        aggregate_panel(
            panel, unit="dma", time="t", outcome="y", hierarchy=h, from_level="dma", to="state"
        )


def test_aggregate_panel_rejects_a_target_named_like_a_panel_column() -> None:
    with pytest.raises(ValueError, match="collides"):
        _agg(_panel().rename({"sales": "district"}), outcome="district")


def test_labels_must_identify_units_uniquely() -> None:
    # Homonyms: "Itabaiana" is a regiao imediata in PB and in SE. Aggregating
    # by label would silently add the two together.
    frame = pl.DataFrame({"m": [1, 2, 3], "r": [10, 20, 30], "r_name": ["x", "x", "y"]})
    h = Hierarchy.from_frame(frame, levels=["m", "r"], labels={"r": "r_name"}, name="h")
    assert h.mapping("m", "r").height == 3
    with pytest.raises(HierarchyError, match="x"):
        h.mapping("m", "r", use_labels=True)


def test_null_labels_are_refused_when_labels_are_used() -> None:
    frame = pl.DataFrame({"m": [1, 2], "r": [10, 20], "r_name": ["x", None]})
    h = Hierarchy.from_frame(frame, levels=["m", "r"], labels={"r": "r_name"}, name="h")
    with pytest.raises(HierarchyError, match="null"):
        h.mapping("m", "r", use_labels=True)


def test_aggregate_panel_rejects_a_panel_column_named_like_the_target() -> None:
    panel = _panel().with_columns(district=pl.lit(99))
    with pytest.raises(ValueError, match="collides"):
        _agg(panel)


def test_aggregate_panel_is_not_confused_by_a_column_named_like_its_internals() -> None:
    panel = _panel().with_columns(_mapped=pl.lit(None, dtype=pl.Boolean))
    out = _agg(panel)
    assert out.height == 4
