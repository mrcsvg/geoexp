"""Shared pytest fixtures for the geoexp test suite.

Unit tests (tests/unit/) must run without R. The GeoLift parity fixtures
live in tests/validation_against_r/conftest.py behind
pytest.importorskip("rpy2").
"""
