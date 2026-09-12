"""Domain exceptions for geoexp.

Mirrors augsynth-py's convention: ``ValueError`` for bad input, subclasses of
:class:`GeoexpError` for domain errors a caller may want to catch
specifically.
"""

from __future__ import annotations


class GeoexpError(Exception):
    """Base class for geoexp domain errors."""
