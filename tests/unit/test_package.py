"""Smoke tests for the package skeleton."""

from __future__ import annotations


def test_version_is_exposed() -> None:
    import geoexp

    assert isinstance(geoexp.__version__, str)
    assert geoexp.__version__ != ""
    assert "__version__" in geoexp.__all__


def test_domain_exception_base() -> None:
    from geoexp.exceptions import GeoexpError

    assert issubclass(GeoexpError, Exception)


def test_augsynth_public_contract_is_importable() -> None:
    # geoexp consumes augsynth-py through its public API only; these are the
    # names the v0.5 implementation is written against (contract freeze:
    # augsynth-py 0.5.0, docs/power-api-contract-review.md there).
    from augsynth_py import (
        DEFAULT_EFFECT_SIZES,
        AugSynth,
        PowerEstimator,
        PowerParams,
        PowerResults,
        Synth,
        simulate_power,
    )

    assert callable(simulate_power)
    assert len(DEFAULT_EFFECT_SIZES) == 6
    for obj in (AugSynth, Synth, PowerEstimator, PowerParams, PowerResults):
        assert obj is not None
