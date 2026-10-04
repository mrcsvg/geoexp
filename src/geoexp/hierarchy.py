"""Unit hierarchies and aggregation to coarser units.

A :class:`Hierarchy` says which coarser units each fine unit belongs to ---
municipality -> state, store -> sales district, zip -> media market. Build one
from any frame with :meth:`Hierarchy.from_frame`; the companion package
``geoexp-units`` ships ready-made ones (IBGE first) built the same way, so
geoexp itself never changes when a new hierarchy is added.

:func:`aggregate_panel` uses a hierarchy to roll a long panel up to the level
the experiment runs at. This is a geoexp construct with no R GeoLift
counterpart; see ``docs/methodology.md``.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

import polars as pl

from geoexp.exceptions import HierarchyError

OnUnmapped = Literal["raise", "drop"]

#: How many offending values an error or warning names.
_SHOW = 10


@dataclass(frozen=True)
class Hierarchy:
    """Membership of fine units in coarser ones.

    One row per unit of the finest level, one column per level. Levels are
    listed finest first, but nesting is **not** required: whether a level can
    be aggregated into another is checked per direction, in :meth:`mapping`.
    That admits hierarchies that do not nest, such as media markets crossing
    state lines.

    Build instances with :meth:`from_frame`, which validates them.

    Attributes
    ----------
    name : str
        Short identifier, e.g. ``"br.ibge"``.
    levels : tuple of str
        Level column names, finest first.
    frame : polars.DataFrame
        The level and label columns.
    labels : dict of str to str
        Level -> column holding a human-readable label for its values.
    source : str or None
        Provenance: where the data came from and when.
    """

    name: str
    levels: tuple[str, ...]
    frame: pl.DataFrame = field(repr=False)
    labels: Mapping[str, str] = field(default_factory=dict)
    source: str | None = None

    @classmethod
    def from_frame(
        cls,
        frame: pl.DataFrame,
        *,
        levels: Sequence[str],
        name: str,
        labels: Mapping[str, str] | None = None,
        source: str | None = None,
    ) -> Hierarchy:
        """Build and validate a hierarchy from a frame.

        Parameters
        ----------
        frame : polars.DataFrame
            One row per finest unit; other columns are ignored.
        levels : sequence of str
            Level columns, finest first.
        name : str
            Short identifier.
        labels : mapping of str to str, optional
            Level -> label column, e.g. ``{"uf": "uf_sigla"}``.
        source : str, optional
            Provenance.

        Returns
        -------
        Hierarchy

        Raises
        ------
        ValueError
            If ``levels`` is empty or repeats a name, a level or label column is
            missing, or a label names an unknown level.
        HierarchyError
            If a level column has nulls, or the finest level repeats a unit.
        """
        levels = tuple(levels)
        labels = dict(labels or {})
        if not levels:
            raise ValueError("levels must not be empty")
        if len(set(levels)) != len(levels):
            raise ValueError(f"levels repeat a name: {list(levels)}")
        unknown = [lvl for lvl in labels if lvl not in levels]
        if unknown:
            raise ValueError(f"labels name levels not in {list(levels)}: {unknown}")
        missing = [c for c in (*levels, *labels.values()) if c not in frame.columns]
        if missing:
            raise ValueError(f"columns not found in frame: {missing}")

        nulls = [lvl for lvl in levels if frame.get_column(lvl).null_count()]
        if nulls:
            raise HierarchyError(f"level column(s) with nulls: {nulls}")
        finest = levels[0]
        repeated = frame.get_column(finest).is_duplicated()
        if repeated.any():
            shown = frame.filter(repeated).get_column(finest).unique().sort().head(_SHOW)
            raise HierarchyError(
                f"finest level {finest!r} repeats units, e.g. {shown.to_list()}: "
                "each finest unit must appear on exactly one row"
            )
        kept = frame.select(*dict.fromkeys((*levels, *labels.values())))
        return cls(name=name, levels=levels, frame=kept, labels=labels, source=source)

    def mapping(self, from_level: str, to_level: str, *, use_labels: bool = False) -> pl.DataFrame:
        """Two-column frame mapping ``from_level`` values to ``to_level`` values.

        Parameters
        ----------
        from_level, to_level : str
            Levels of this hierarchy.
        use_labels : bool, default False
            Return ``to_level``'s label instead of its value, under the
            ``to_level`` column name.

        Returns
        -------
        polars.DataFrame
            Columns ``[from_level, to_level]``, one row per ``from_level`` value.

        Raises
        ------
        ValueError
            If a level is unknown, the two are the same, or ``use_labels`` is
            asked for a level without labels.
        HierarchyError
            If a ``from_level`` value maps to more than one ``to_level`` value,
            or (with ``use_labels``) a value has more than one label, a label
            names more than one value, or a label is null.
        """
        for lvl in (from_level, to_level):
            if lvl not in self.levels:
                raise ValueError(f"unknown level {lvl!r}; levels are {list(self.levels)}")
        if from_level == to_level:
            raise ValueError(f"from_level and to_level are both {from_level!r}")
        pairs = self.frame.select(from_level, to_level).unique()
        _require_function(pairs, from_level, to_level)
        if not use_labels:
            return pairs
        if to_level not in self.labels:
            raise ValueError(f"level {to_level!r} has no labels; labelled: {list(self.labels)}")
        label = self.labels[to_level]
        names = self.frame.select(to_level, label).unique()
        if names.get_column(label).null_count():
            raise HierarchyError(
                f"level {to_level!r} has null labels in {label!r}; use codes instead"
            )
        _require_function(names, to_level, label)
        # Labels are only usable if they identify units: homonyms (the same
        # name for two codes) would be summed together under one label.
        _require_function(names, label, to_level)
        return pairs.join(names, on=to_level, how="left").drop(to_level).rename({label: to_level})


def _require_function(pairs: pl.DataFrame, source: str, target: str) -> None:
    """Raise unless every ``source`` value has exactly one ``target`` value."""
    counts = pairs.group_by(source).len().filter(pl.col("len") > 1)
    if counts.height:
        shown = counts.get_column(source).sort().head(_SHOW).to_list()
        raise HierarchyError(
            f"{source!r} does not map to a single {target!r}: {counts.height} value(s) "
            f"map to several, e.g. {shown}. Aggregating in that direction would "
            "double-count them."
        )


def aggregate_panel(
    panel: pl.DataFrame,
    *,
    unit: str,
    time: str,
    outcome: str | Sequence[str],
    hierarchy: Hierarchy,
    to: str,
    from_level: str | None = None,
    on_unmapped: OnUnmapped = "raise",
    use_labels: bool = False,
) -> pl.DataFrame:
    """Roll a long panel up to a coarser level of a hierarchy, by summing.

    Parameters
    ----------
    panel : polars.DataFrame
        Long panel, e.g. from :func:`geoexp.prepare_panel`.
    unit, time : str
        Column names; ``unit`` holds values of ``from_level``.
    outcome : str or sequence of str
        Column(s) to sum. Rates are not summable: aggregate their numerator and
        denominator as two outcomes and divide afterwards.
    hierarchy : Hierarchy
        Membership data.
    to : str
        Target level.
    from_level : str, optional
        The level ``unit`` holds; defaults to the hierarchy's finest level.
    on_unmapped : {"raise", "drop"}, default "raise"
        Panel units absent from the hierarchy are rejected, or dropped with a
        warning naming them.
    use_labels : bool, default False
        Name target units by their label instead of their code.

    Returns
    -------
    polars.DataFrame
        ``[to, time, *outcome]``, sorted by ``to`` then ``time``. A null outcome
        in any member keeps the aggregate null.

    Raises
    ------
    ValueError
        On unmapped units under ``on_unmapped="raise"``, or when the panel's
        unit dtype differs from the hierarchy's.
    HierarchyError
        If ``from_level`` does not map to a single ``to`` value.
    """
    outcomes = [outcome] if isinstance(outcome, str) else list(outcome)
    if to in panel.columns:
        raise ValueError(f"target level {to!r} collides with a panel column name; rename it first")
    source = hierarchy.levels[0] if from_level is None else from_level
    mapping = hierarchy.mapping(source, to, use_labels=use_labels)

    ours, theirs = panel.schema[unit], mapping.schema[source]
    if ours != theirs:
        raise ValueError(
            f"panel column {unit!r} has dtype {ours} but hierarchy level {source!r} has "
            f"dtype {theirs}; cast one to the other first, e.g. "
            f"pl.col({unit!r}).cast(pl.{theirs})"
        )

    flag = "_mapped"
    while flag in panel.columns:
        flag = f"_{flag}"
    joined = panel.join(
        mapping.rename({source: unit}).with_columns(pl.lit(True).alias(flag)), on=unit, how="left"
    )
    unmapped = joined.filter(pl.col(flag).is_null()).get_column(unit).unique().sort()
    if unmapped.len():
        message = (
            f"{unmapped.len()} unit(s) in {unit!r} are not in hierarchy "
            f"{hierarchy.name!r} at level {source!r}, e.g. {unmapped.head(_SHOW).to_list()}"
        )
        if on_unmapped == "raise":
            raise ValueError(message)
        warnings.warn(f"{message}; dropped", UserWarning, stacklevel=2)
        joined = joined.filter(pl.col(flag).is_not_null())

    return (
        joined.group_by(to, time)
        .agg(
            pl.when(pl.col(c).null_count() > 0).then(None).otherwise(pl.col(c).sum()).alias(c)
            for c in outcomes
        )
        .sort(to, time)
    )
