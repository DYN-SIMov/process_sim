"""
End-to-end tests: direct regression from VLE data with the (memetic) genetic algorithm.

These are slow (GA with pop_size=1000, 100 generations); deselect with ``-m "not slow"``.
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.slow


@pytest.mark.todo
@pytest.mark.parametrize("config", ["wilson-dippr", "nrtl-nrtl"])
def test_memetic_regression_improves_on_elementwise(methanol_water_vle_data, bip_estimator_factory, config) -> None:
    """DIRECT_VLE goodness of fit >= ELEMENTWISE goodness of fit after the memetic run (n_jobs=1)."""
    raise NotImplementedError


@pytest.mark.todo
def test_plain_ga_without_memetic_step(methanol_water_vle_data, bip_estimator_factory) -> None:
    """is_memetic=False completes and records DIRECT_VLE results."""
    raise NotImplementedError


@pytest.mark.todo
def test_without_biased_initialisation(methanol_water_vle_data, bip_estimator_factory) -> None:
    """use_biased_initialization=False uses FloatRandomSampling and still converges to a sensible fit."""
    raise NotImplementedError


@pytest.mark.todo
def test_nrtl_variable_alpha_regresses_alpha_within_bounds(methanol_water_vle_data, bip_estimator_factory) -> None:
    """Regressed alpha_12/alpha_21 lie within (1e-3, 0.5)."""
    raise NotImplementedError


@pytest.mark.todo
def test_reproducible_with_fixed_seed(methanol_water_vle_data, bip_estimator_factory) -> None:
    """Two runs with n_jobs=1 give identical results (pymoo seed=1, sampling seed=42)."""
    raise NotImplementedError


@pytest.mark.todo
def test_parallel_and_serial_runs_agree(methanol_water_vle_data, bip_estimator_factory) -> None:
    """n_jobs=1 and n_jobs=2 produce the same optimum (joblib evaluation is order-preserving)."""
    raise NotImplementedError
