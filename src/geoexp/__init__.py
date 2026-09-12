"""geoexp: geo-experiment design on top of augsynth-py.

Market selection, power-based design ranking, and experiment reporting for
geo-experiments. This package answers *which design should I run?*; its
dependency `augsynth-py` answers *what can this design detect?* — see the
package-boundary section of CLAUDE.md.

The public API is fixed by docs/design-2026-09-05-v0.5-api.md.
"""

from geoexp._version import __version__
from geoexp.selection import DesignRanking, enumerate_candidates, rank_designs

__all__ = [
    "DesignRanking",
    "__version__",
    "enumerate_candidates",
    "rank_designs",
]
