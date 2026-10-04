"""Domain exceptions for geoexp.

Mirrors augsynth-py's convention: ``ValueError`` for bad input, subclasses of
:class:`GeoexpError` for domain errors a caller may want to catch
specifically.
"""

from __future__ import annotations


class GeoexpError(Exception):
    """Base class for geoexp domain errors."""


class HierarchyError(GeoexpError):
    """A hierarchy's data cannot support the requested operation.

    Raised for nulls or a repeated finest unit when a :class:`geoexp.Hierarchy`
    is built, and when an aggregation direction is not a function (a source
    value that maps to more than one target value).
    """
