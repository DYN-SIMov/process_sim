"""
Root conftest: infrastructure shared by unit, integration and e2e tests.

Responsibilities:
    * force a non-interactive matplotlib backend,
    * auto-mark tests by folder (unit / integration / e2e),
    * skip scaffolded tests marked with ``@pytest.mark.todo`` before fixture setup,
    * forbid unscripted ``input()`` calls (several code paths prompt the user),
    * protect the shipped ``BIP_database.json`` from being modified by tests,
    * provide path / temporary-file fixtures,
    * provide real-data fixtures shared by integration and e2e tests.
"""
from __future__ import annotations

import hashlib
import os
import shutil
from collections.abc import Callable, Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")

import pytest

import thermodynamics.activity_models_regression as regression_pkg
import thermodynamics.databases as databases_pkg


TESTS_ROOT = Path(__file__).parent
THERMO_DATA_DIR = Path(regression_pkg.__file__).parent / "thermo_data"
SHIPPED_BIP_DATABASE = Path(databases_pkg.__file__).parent / "BIP_database.json"

_FOLDER_MARKERS = ("unit", "integration", "e2e")


# --------------------------------------------------------------------------------------
# Hooks
# --------------------------------------------------------------------------------------
def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    skip_todo = pytest.mark.skip(reason="scaffold: test not implemented yet")
    for item in items:
        relative_parts = Path(str(item.path)).relative_to(TESTS_ROOT).parts
        if relative_parts and relative_parts[0] in _FOLDER_MARKERS:
            item.add_marker(getattr(pytest.mark, relative_parts[0]))
        if item.get_closest_marker("todo") is not None:
            item.add_marker(skip_todo)


# --------------------------------------------------------------------------------------
# Safety guards
# --------------------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _forbid_interactive_input(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail loudly if code under test calls ``input()`` without ``scripted_input``."""

    def _raise(prompt: str = "") -> str:
        raise RuntimeError(
            f"Unexpected call to input() during test (prompt: {prompt!r}). "
            f"Use the 'scripted_input' fixture to provide answers."
        )

    monkeypatch.setattr("builtins.input", _raise)


@pytest.fixture(autouse=True, scope="session")
def _protect_shipped_bip_database() -> Iterator[None]:
    """Ensure no test writes to the BIP database shipped with the package."""
    digest_before = hashlib.sha256(SHIPPED_BIP_DATABASE.read_bytes()).hexdigest()
    yield
    digest_after = hashlib.sha256(SHIPPED_BIP_DATABASE.read_bytes()).hexdigest()
    assert digest_before == digest_after, (
        f"{SHIPPED_BIP_DATABASE} was modified during the test session. "
        f"Use the 'tmp_bip_database' fixture instead of the default database path."
    )


@pytest.fixture(autouse=True)
def _close_matplotlib_figures() -> Iterator[None]:
    yield
    import matplotlib.pyplot as plt

    plt.close("all")


# --------------------------------------------------------------------------------------
# Interactive input
# --------------------------------------------------------------------------------------
@pytest.fixture
def scripted_input(monkeypatch: pytest.MonkeyPatch) -> Callable[[Iterable[str]], list[str]]:
    """
    Factory replacing ``input()`` with a queue of predefined answers.

    Usage::

        prompts = scripted_input(["1", "y"])
        ...  # code under test
        assert len(prompts) == 2   # prompts that were shown

    Raises if the code asks for more answers than were provided.
    """

    def _install(answers: Iterable[str]) -> list[str]:
        queue = list(answers)
        prompts: list[str] = []

        def _fake_input(prompt: str = "") -> str:
            prompts.append(prompt)
            if not queue:
                raise RuntimeError(f"No scripted answer left for prompt: {prompt!r}")
            return queue.pop(0)

        monkeypatch.setattr("builtins.input", _fake_input)
        return prompts

    return _install


# --------------------------------------------------------------------------------------
# Paths and files
# --------------------------------------------------------------------------------------
@pytest.fixture(scope="session")
def thermo_data_dir() -> Path:
    return THERMO_DATA_DIR


@pytest.fixture(scope="session")
def shipped_bip_database_path() -> Path:
    """Read-only path to the shipped BIP database (never write to it)."""
    return SHIPPED_BIP_DATABASE


@pytest.fixture
def tmp_bip_database(tmp_path: Path) -> Path:
    """Writable copy of the shipped BIP database."""
    destination = tmp_path / "BIP_database.json"
    shutil.copyfile(SHIPPED_BIP_DATABASE, destination)
    return destination


@pytest.fixture
def empty_bip_database(tmp_path: Path) -> Path:
    """Database with empty sections for every supported activity model."""
    import json

    from thermodynamics.databases.bip_db_manager import ActivityModel

    destination = tmp_path / "BIP_database.json"
    destination.write_text(json.dumps({model.value: {} for model in ActivityModel}))
    return destination


VLECsvFactory = Callable[..., Path]


@pytest.fixture
def vle_csv_factory(tmp_path: Path) -> VLECsvFactory:
    """
    Factory writing a VLE CSV file in the format expected by ``VLEData``.

    Example::

        path = vle_csv_factory(
            components=("methanol", "water"),
            rows=[(64.5, 1.0, 0.95, 0.98), ...],   # T (degC), P (atm), x_comp1, y_comp1
        )
    """

    def _write(
        components: Sequence[str] = ("methanol", "water"),
        rows: Sequence[Sequence[float]] = (),
        source: str | None = "pytest synthetic data",
        composition_component: str | None = None,
        extra_header_lines: Sequence[str] = (),
        filename: str = "vle_data.csv",
    ) -> Path:
        key = composition_component or components[0]
        lines = ["# Synthetic VLE data generated by the test-suite"]
        lines += [f"# Component {k + 1}: {name}" for k, name in enumerate(components)]
        if source is not None:
            lines.append(f"# Source: {source}")
        lines += list(extra_header_lines)
        lines.append(f"T (degC),P (atm),x_{key},y_{key}")
        lines += [",".join(str(value) for value in row) for row in rows]

        path = tmp_path / filename
        path.write_text("\n".join(lines) + "\n")
        return path

    return _write


# --------------------------------------------------------------------------------------
# Real-data fixtures (integration / e2e)
# Shipped CSV files and real thermo / CoolProp / chemicals backends.
# --------------------------------------------------------------------------------------
SHIPPED_VLE_FILES = ("VLE_MeOH_H2O.csv", "VLE_MeOH_PrOH.csv", "VLE_H2O_NH3.csv")


@pytest.fixture(params=SHIPPED_VLE_FILES)
def shipped_vle_csv(request: pytest.FixtureRequest) -> Path:
    return THERMO_DATA_DIR / request.param


@pytest.fixture
def methanol_water_vle_data() -> Any:
    """Isobaric methanol-water data (Gmehling & Onken), within thermo's Psat validity range."""
    from thermodynamics.activity_models_regression.data_handling import VLEData

    return VLEData(filepath=str(THERMO_DATA_DIR / "VLE_MeOH_H2O.csv"))


@pytest.fixture
def methanol_propanol_vle_data() -> Any:
    from thermodynamics.activity_models_regression.data_handling import VLEData

    return VLEData(filepath=str(THERMO_DATA_DIR / "VLE_MeOH_PrOH.csv"))


@pytest.fixture
def bip_estimator_factory() -> Callable[..., Any]:
    """
    Factory building a real ``BinaryInteractionParametersRegression``.

    ``config`` selects a compatible activity model / polynomial pair:
    ``"wilson-dippr"`` or ``"nrtl-nrtl"``.
    """
    from core.properties import SoaveRedlichKwongEoSBackend
    from thermodynamics.activity_models_regression.activity_models_aux import (
        NRTLActivityModelRegression,
        WilsonActivityModelRegression,
    )
    from thermodynamics.activity_models_regression.optimization import (
        AbsoluteForm,
        PolynomialExponentialDIPPR,
        PolynomialNRTL,
    )
    from thermodynamics.activity_models_regression.regression_aux import (
        BinaryInteractionParametersRegression,
    )

    configs: dict[str, tuple[Any, Any]] = {
        "wilson-dippr": (WilsonActivityModelRegression, PolynomialExponentialDIPPR),
        "nrtl-nrtl": (NRTLActivityModelRegression, PolynomialNRTL),
    }

    def _make(
        VLE_data: Any,
        config: str = "wilson-dippr",
        polynomial_form: Any = AbsoluteForm,
        equation_of_state: Any = SoaveRedlichKwongEoSBackend,
        activity_model_regression: Any = None,
    ) -> Any:
        model_cls, polynomial_cls = configs[config]
        return BinaryInteractionParametersRegression(
            activity_model_regression=activity_model_regression or model_cls,
            equation_of_state=equation_of_state,
            VLE_data=VLE_data,
            polynomial=polynomial_cls(degree=4),
            polynomial_form=polynomial_form,
        )

    return _make


@pytest.fixture
def tmp_db_manager(tmp_bip_database: Path) -> Any:
    """BIPDatabaseManager bound to a temporary copy of the shipped database."""
    from thermodynamics.databases.bip_db_manager import BIPDatabaseManager

    return BIPDatabaseManager(database_filepath=str(tmp_bip_database))
