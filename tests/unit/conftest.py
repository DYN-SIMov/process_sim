"""
Fixtures for isolated unit tests.

Unit tests should not depend on thermo / CoolProp / chemicals lookups or on the shipped CSV
files; everything here is synthetic or mocked.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from core.properties import PureComponentDataBackend
from thermodynamics.activity_models_regression.activity_models_aux import (
    BinaryInteractionParameter,
    NRTLActivityModelRegression,
    WilsonActivityModelRegression,
)
from thermodynamics.activity_models_regression.data_handling import DataPoint, TxyPoint
from thermodynamics.activity_models_regression.optimization import (
    AbsoluteForm,
    NormalizedForm,
    PolynomialExponentialDIPPR,
    PolynomialNRTL,
    PolynomialRegular,
)


COMPONENTS: tuple[str, str] = ("methanol", "water")


# --------------------------------------------------------------------------------------
# Activity models
# --------------------------------------------------------------------------------------
@pytest.fixture
def fake_pure_component_backend() -> MagicMock:
    """Stand-in for ``PureComponentDataBackend`` (avoids chemicals database lookups)."""
    return MagicMock(spec=PureComponentDataBackend, components=COMPONENTS)


@pytest.fixture
def wilson_model(fake_pure_component_backend: MagicMock) -> WilsonActivityModelRegression:
    return WilsonActivityModelRegression(
        components=COMPONENTS, pure_component_data_backend=fake_pure_component_backend
    )


@pytest.fixture
def nrtl_model_fixed_alpha(fake_pure_component_backend: MagicMock) -> NRTLActivityModelRegression:
    return NRTLActivityModelRegression(
        components=COMPONENTS,
        pure_component_data_backend=fake_pure_component_backend,
        alpha_is_fixed=True,
        alpha=0.3,
    )


@pytest.fixture
def nrtl_model_variable_alpha(fake_pure_component_backend: MagicMock) -> NRTLActivityModelRegression:
    return NRTLActivityModelRegression(
        components=COMPONENTS,
        pure_component_data_backend=fake_pure_component_backend,
        alpha_is_fixed=False,
        alpha=0.2,
    )


@pytest.fixture
def bip_factory() -> Callable[..., BinaryInteractionParameter]:
    """Factory for ``BinaryInteractionParameter`` with sensible defaults."""

    def _make(
        name: str = "Lambda_12",
        value: float | np.ndarray | None = None,
        initial_guess: float = 1.0,
        bounds: tuple[float | None, float | None] | None = (1e-3, None),
        is_temperature_dependant: bool = True,
        is_regressed: bool = True,
    ) -> BinaryInteractionParameter:
        return BinaryInteractionParameter(
            name=name,
            value=value,
            initial_guess=initial_guess,
            bounds=bounds,
            is_temperature_dependant=is_temperature_dependant,
            is_regressed=is_regressed,
        )

    return _make


# --------------------------------------------------------------------------------------
# Polynomials
# --------------------------------------------------------------------------------------
POLYNOMIAL_CLASSES = (PolynomialExponentialDIPPR, PolynomialNRTL, PolynomialRegular)
POLYNOMIAL_FORMS = (AbsoluteForm, NormalizedForm)


@pytest.fixture(params=POLYNOMIAL_CLASSES, ids=lambda cls: cls.__name__)
def polynomial(request: pytest.FixtureRequest) -> Any:
    """Every concrete polynomial (degree 4, as currently hardcoded in the implementations)."""
    return request.param(degree=4)


@pytest.fixture(params=POLYNOMIAL_FORMS, ids=lambda cls: cls.__name__)
def polynomial_form_cls(request: pytest.FixtureRequest) -> Any:
    return request.param


# --------------------------------------------------------------------------------------
# Synthetic VLE data
# --------------------------------------------------------------------------------------
@pytest.fixture
def data_point_factory() -> Callable[..., DataPoint]:
    def _make(
        temperature_K: float = 350.0,
        pressure_Pa: float = 1.0e5,
        x1: float = 0.5,
        y1: float = 0.7,
        components: Sequence[str] = COMPONENTS,
    ) -> DataPoint:
        return DataPoint(
            pressure_Pa=pressure_Pa,
            temperature_K=temperature_K,
            x1_mol_frac=x1,
            y1_mol_frac=y1,
            x2_mol_frac=1.0 - x1,
            y2_mol_frac=1.0 - y1,
            comp1=components[0],
            comp2=components[1],
        )

    return _make


@pytest.fixture
def ideal_raoult_vle_data(data_point_factory: Callable[..., DataPoint]) -> SimpleNamespace:
    """
    Duck-typed ``VLEData`` generated from Raoult's law (gamma = 1, ideal gas).

    Exposes ``components``, ``source`` and ``T_x_y_points`` (with saturation pressures set),
    which is everything ``BinaryInteractionParametersRegression`` and ``BIPDatabaseManager``
    read from ``VLEData``.
    """
    txy_points: list[TxyPoint] = []
    for temperature_K, psat_1, psat_2 in [(330.0, 1.5e5, 0.6e5), (340.0, 2.0e5, 0.8e5), (350.0, 2.6e5, 1.1e5)]:
        points = []
        for x1 in (0.2, 0.4, 0.6, 0.8):
            pressure_Pa = x1 * psat_1 + (1.0 - x1) * psat_2
            y1 = x1 * psat_1 / pressure_Pa
            points.append(data_point_factory(temperature_K=temperature_K, pressure_Pa=pressure_Pa, x1=x1, y1=y1))
        txy = TxyPoint(data_points=points)
        txy.comp_1_saturation_pressure_Pa = psat_1
        txy.comp_2_saturation_pressure_Pa = psat_2
        txy_points.append(txy)

    return SimpleNamespace(
        components=list(COMPONENTS),
        source="pytest synthetic Raoult data",
        T_x_y_points=txy_points,
    )
