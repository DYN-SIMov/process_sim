"""Unit tests for thermodynamics.activity_models_regression.activity_models_aux."""
from __future__ import annotations

import pytest


# --------------------------------------------------------------------------------------
# RegressionAuxiliariesMixin (BIP bookkeeping)
# --------------------------------------------------------------------------------------
class TestRegressionAuxiliariesMixin:

    @pytest.mark.todo
    def test_get_BIP_names_preserves_order(self, nrtl_model_fixed_alpha) -> None:
        """Names are returned in the order of self.BIPs."""
        raise NotImplementedError

    @pytest.mark.todo
    def test_initial_guess_elementwise_returns_provided_guess(self, wilson_model) -> None:
        """A non-None initial_guess is returned unchanged (warm-start from previous isotherm)."""
        raise NotImplementedError

    @pytest.mark.todo
    def test_initial_guess_elementwise_default(self, nrtl_model_variable_alpha) -> None:
        """Default guess contains only regressed, temperature-dependent BIPs."""
        raise NotImplementedError

    @pytest.mark.todo
    def test_get_bounds_elementwise(self, nrtl_model_variable_alpha) -> None:
        """Bounds contain only regressed, temperature-dependent BIPs, aligned with the initial guess."""
        raise NotImplementedError

    @pytest.mark.todo
    @pytest.mark.parametrize("model_fixture", ["wilson_model", "nrtl_model_fixed_alpha", "nrtl_model_variable_alpha"])
    def test_temp_dependant_and_independant_partition(self, request, model_fixture) -> None:
        """Dependent + independent lists cover exactly the regressed BIPs, without overlap."""
        raise NotImplementedError

    @pytest.mark.todo
    @pytest.mark.parametrize(
        ("model_fixture", "expected_map"),
        [
            ("wilson_model", {"Lambda_12": 0, "Lambda_21": 1}),
            ("nrtl_model_fixed_alpha", {"tau_12": 0, "tau_21": 1}),
            ("nrtl_model_variable_alpha", {"tau_12": 0, "tau_21": 1, "alpha_12": 2, "alpha_21": 3}),
        ],
    )
    def test_get_bip_index_map(self, request, model_fixture, expected_map) -> None:
        """Index map skips non-regressed BIPs and enumerates regressed ones contiguously."""
        raise NotImplementedError


class TestRegressionMessages:

    @pytest.mark.todo
    def test_elementwise_message_multiple_points(self, wilson_model, capsys) -> None:
        """Successful result with several points prints P/x/y ranges and fitted BIPs."""
        raise NotImplementedError

    @pytest.mark.todo
    def test_elementwise_message_single_point(self, wilson_model, capsys) -> None:
        """Successful result with one point prints single P/x/y values."""
        raise NotImplementedError

    @pytest.mark.todo
    def test_elementwise_message_failed_optimisation(self, wilson_model, capsys) -> None:
        """Unsuccessful result prints the 'did not converge' message with optimizer output."""
        raise NotImplementedError

    @pytest.mark.todo
    def test_elementwise_message_raises_on_empty_data(self, wilson_model) -> None:
        """Empty pressure data raises ValueError."""
        raise NotImplementedError

    @pytest.mark.todo
    def test_polynomial_coeffs_estimation_message(self, wilson_model, capsys) -> None:
        """Prints polynomial name/equation and lettered coefficients for each temperature-dependent BIP."""
        raise NotImplementedError

    @pytest.mark.todo
    def test_estimation_from_VLE_message(self, nrtl_model_variable_alpha, capsys) -> None:
        """Prints coefficients for temperature-dependent BIPs, scalar values for the others, and residual."""
        raise NotImplementedError


# --------------------------------------------------------------------------------------
# Wilson
# --------------------------------------------------------------------------------------
class TestWilsonActivityModelRegression:

    @pytest.mark.todo
    def test_bip_configuration(self, wilson_model) -> None:
        """Two regressed, temperature-dependent BIPs (Lambda_12, Lambda_21) with positive lower bound."""
        raise NotImplementedError

    @pytest.mark.todo
    @pytest.mark.parametrize("x1", [0.1, 0.5, 0.9])
    def test_ideal_solution_when_lambdas_equal_one(self, wilson_model, x1) -> None:
        """Lambda_12 = Lambda_21 = 1 gives gamma_1 = gamma_2 = 1."""
        raise NotImplementedError

    @pytest.mark.todo
    def test_pure_component_limits(self, wilson_model) -> None:
        """gamma_1 -> 1 as x1 -> 1 and gamma_2 -> 1 as x1 -> 0."""
        raise NotImplementedError

    @pytest.mark.todo
    def test_infinite_dilution_limits(self, wilson_model) -> None:
        """ln(gamma_1^inf) = -ln(Lambda_12) + 1 - Lambda_21 (and symmetric expression for gamma_2)."""
        raise NotImplementedError

    @pytest.mark.todo
    def test_gibbs_duhem_consistency(self, wilson_model) -> None:
        """x1 d(ln gamma_1)/dx1 + x2 d(ln gamma_2)/dx1 = 0 (finite differences)."""
        raise NotImplementedError

    @pytest.mark.todo
    def test_matches_reference_implementation(self, wilson_model) -> None:
        """Gammas agree with an independent reference (e.g. thermo.Wilson or tabulated values)."""
        raise NotImplementedError

    @pytest.mark.todo
    def test_accepts_list_and_ndarray_coefficients(self, wilson_model) -> None:
        """list[float] and np.ndarray BIP_coeffs give identical results."""
        raise NotImplementedError


# --------------------------------------------------------------------------------------
# NRTL
# --------------------------------------------------------------------------------------
class TestNRTLActivityModelRegression:

    @pytest.mark.todo
    def test_bip_configuration_fixed_alpha(self, nrtl_model_fixed_alpha) -> None:
        """alpha_12/alpha_21 are not regressed, have value=alpha and no bounds."""
        raise NotImplementedError

    @pytest.mark.todo
    def test_bip_configuration_variable_alpha(self, nrtl_model_variable_alpha) -> None:
        """alpha_12/alpha_21 are regressed, temperature-independent, bounded in (1e-3, 0.5)."""
        raise NotImplementedError

    @pytest.mark.todo
    @pytest.mark.parametrize("x1", [0.1, 0.5, 0.9])
    def test_ideal_solution_when_taus_zero(self, nrtl_model_fixed_alpha, x1) -> None:
        """tau_12 = tau_21 = 0 gives gamma_1 = gamma_2 = 1."""
        raise NotImplementedError

    @pytest.mark.todo
    def test_fixed_alpha_ignores_extra_coefficients(self, nrtl_model_fixed_alpha) -> None:
        """With fixed alpha, self.alpha is used regardless of any extra entries in BIP_coeffs."""
        raise NotImplementedError

    @pytest.mark.todo
    def test_variable_alpha_read_from_coefficients(self, nrtl_model_variable_alpha) -> None:
        """With variable alpha, alpha_12/alpha_21 are read from BIP_coeffs via the index map."""
        raise NotImplementedError

    @pytest.mark.todo
    def test_infinite_dilution_limits(self, nrtl_model_fixed_alpha) -> None:
        """ln(gamma_1^inf) = tau_21 + tau_12 * exp(-alpha * tau_12)."""
        raise NotImplementedError

    @pytest.mark.todo
    def test_gibbs_duhem_consistency(self, nrtl_model_fixed_alpha) -> None:
        """x1 d(ln gamma_1)/dx1 + x2 d(ln gamma_2)/dx1 = 0 (finite differences)."""
        raise NotImplementedError

    @pytest.mark.todo
    def test_matches_reference_implementation(self, nrtl_model_fixed_alpha) -> None:
        """Gammas agree with an independent reference (e.g. thermo.NRTL or tabulated values)."""
        raise NotImplementedError


# --------------------------------------------------------------------------------------
# Interface conformance
# --------------------------------------------------------------------------------------
@pytest.mark.todo
@pytest.mark.parametrize("model_fixture", ["wilson_model", "nrtl_model_fixed_alpha", "nrtl_model_variable_alpha"])
def test_models_conform_to_regression_interface(request, model_fixture) -> None:
    """Every regression model exposes the attributes/methods of ActivityModelRegressionInterface."""
    raise NotImplementedError


@pytest.mark.todo
@pytest.mark.parametrize("model_fixture", ["wilson_model", "nrtl_model_fixed_alpha"])
def test_model_instances_do_not_share_BIP_objects(request, model_fixture, fake_pure_component_backend) -> None:
    """Two instances of the same model own independent BinaryInteractionParameter objects."""
    raise NotImplementedError
