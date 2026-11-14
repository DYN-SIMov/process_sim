"""Fixtures for BIPDatabaseManager unit tests."""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from thermodynamics.activity_models_regression.activity_models_aux import (
    NRTLActivityModelRegression,
    WilsonActivityModelRegression,
)
from thermodynamics.activity_models_regression.optimization import (
    AbsoluteForm,
    PolynomialExponentialDIPPR,
    PolynomialNRTL,
)
from thermodynamics.activity_models_regression.regression_aux import RegressionMethod
from thermodynamics.databases.bip_db_manager import BIPDatabaseManager


class _FakeEoSBackend:
    """Only the class name is read by BIPDatabaseManager."""


@pytest.fixture
def fake_bip_estimator_factory(
    ideal_raoult_vle_data: SimpleNamespace,
    fake_pure_component_backend: Any,
) -> Callable[..., SimpleNamespace]:
    """
    Duck-typed ``BinaryInteractionParametersRegression`` exposing exactly the attributes read by
    ``BIPDatabaseManager``: VLE_data, activity_model_backend, polynomial, eos_backend,
    regression_results_cache and result_to_save.
    """

    def _make(
        activity_model: str = "wilson",
        polynomial_cls: type | None = None,
        components: tuple[str, str] = ("methanol", "water"),
    ) -> SimpleNamespace:
        if activity_model == "wilson":
            model = WilsonActivityModelRegression(components, fake_pure_component_backend)
            polynomial_cls = polynomial_cls or PolynomialExponentialDIPPR
        elif activity_model == "nrtl":
            model = NRTLActivityModelRegression(components, fake_pure_component_backend)
            polynomial_cls = polynomial_cls or PolynomialNRTL
        else:
            raise ValueError(f"Unknown activity model {activity_model!r}")

        regressed_bips = [
            bip.__class__(**{**bip.__dict__, "value": np.arange(4.0) + k if bip.is_temperature_dependant else bip.initial_guess})
            for k, bip in enumerate(model.BIPs)
        ]
        cache = {
            RegressionMethod.ELEMENTWISE: {"regressed_BIPs": regressed_bips, "goodness_of_fit": 0.98},
            RegressionMethod.DIRECT_VLE: {"regressed_BIPs": regressed_bips, "goodness_of_fit": 0.99},
        }

        vle_data = SimpleNamespace(**{**ideal_raoult_vle_data.__dict__, "components": list(components)})
        return SimpleNamespace(
            VLE_data=vle_data,
            activity_model_backend=model,
            polynomial=AbsoluteForm(polynomial_cls(degree=4)),
            eos_backend=_FakeEoSBackend(),
            regression_results_cache=cache,
            result_to_save={
                "regression_method": RegressionMethod.ELEMENTWISE.value,
                "regressed_BIPs": regressed_bips,
                "goodness_of_fit": 0.98,
            },
        )

    return _make


@pytest.fixture
def db_manager(empty_bip_database: Path) -> BIPDatabaseManager:
    """Manager bound to an empty temporary database (Wilson/NRTL sections present)."""
    return BIPDatabaseManager(database_filepath=str(empty_bip_database))
