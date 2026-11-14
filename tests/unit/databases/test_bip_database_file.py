"""
Data-integrity tests for the shipped thermodynamics/databases/BIP_database.json.

Read-only: these tests only inspect the file via ``shipped_bip_database_path``.
"""
from __future__ import annotations

import pytest


@pytest.mark.todo
def test_is_valid_json(shipped_bip_database_path) -> None:
    """File parses as JSON and the top level is a dict."""
    raise NotImplementedError


@pytest.mark.todo
def test_top_level_sections_are_supported_activity_models(shipped_bip_database_path) -> None:
    """Every top-level key is an ActivityModel value (WilsonActivityModel / NRTLActivityModel)."""
    raise NotImplementedError


@pytest.mark.todo
def test_entries_have_metadata_and_bip_data(shipped_bip_database_path) -> None:
    """Every entry has exactly 'metadata' and 'BIP_data' keys."""
    raise NotImplementedError


@pytest.mark.todo
def test_metadata_schema(shipped_bip_database_path) -> None:
    """Metadata has all required keys with the expected types."""
    raise NotImplementedError


@pytest.mark.todo
def test_name_and_cas_keys_are_mirrored(shipped_bip_database_path) -> None:
    """Each name-keyed entry has an identical CAS-keyed twin built from its metadata."""
    raise NotImplementedError


@pytest.mark.todo
def test_keys_are_sorted_component_pairs(shipped_bip_database_path) -> None:
    """Name keys are alphabetically sorted 'a-b' pairs consistent with _get_component_pair_key."""
    raise NotImplementedError


@pytest.mark.todo
def test_bip_names_match_activity_model(shipped_bip_database_path) -> None:
    """Wilson entries hold Lambda_12/Lambda_21; NRTL entries hold tau_ij (and alpha_ij)."""
    raise NotImplementedError


@pytest.mark.todo
def test_polynomial_coefficient_count(shipped_bip_database_path) -> None:
    """Temperature-dependent BIPs store 4 finite coefficients."""
    raise NotImplementedError


@pytest.mark.todo
def test_polynomial_matches_activity_model(shipped_bip_database_path) -> None:
    """Wilson -> PolynomialExponentialDIPPR, NRTL -> PolynomialNRTL (same rule as export checks)."""
    raise NotImplementedError


@pytest.mark.todo
def test_temperature_and_pressure_bounds_are_consistent(shipped_bip_database_path) -> None:
    """lb <= ub and values are physically sensible (T > 0 K, P > 0 Pa)."""
    raise NotImplementedError


@pytest.mark.todo
def test_goodness_of_fit_in_unit_interval(shipped_bip_database_path) -> None:
    """0 <= goodness_of_fit <= 1."""
    raise NotImplementedError
