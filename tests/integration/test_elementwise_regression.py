"""Integration tests: elementwise regression on real VLE data with real backends."""
from __future__ import annotations

import pytest


class TestElementwiseRegression:

    @pytest.mark.todo
    @pytest.mark.parametrize("config", ["wilson-dippr", "nrtl-nrtl"])
    def test_methanol_water_goodness_of_fit(self, methanol_water_vle_data, bip_estimator_factory, config) -> None:
        """Elementwise regression reaches R^2 above an agreed threshold (e.g. 0.99)."""
        raise NotImplementedError

    @pytest.mark.todo
    @pytest.mark.parametrize("polynomial_form", ["AbsoluteForm", "NormalizedForm"])
    def test_polynomial_forms_give_equivalent_fits(self, methanol_water_vle_data, bip_estimator_factory, polynomial_form) -> None:
        """Absolute and normalised forms reach comparable R^2 / predicted y."""
        raise NotImplementedError

    @pytest.mark.todo
    def test_ideal_gas_vs_srk_eos(self, methanol_water_vle_data, bip_estimator_factory) -> None:
        """At ~1 atm, fits with and without SRK fugacity corrections are close."""
        raise NotImplementedError

    @pytest.mark.todo
    def test_nrtl_with_variable_alpha(self, methanol_water_vle_data, bip_estimator_factory) -> None:
        """NRTL with alpha_is_fixed=False (functools.partial) runs and stores alpha at its initial guess."""
        raise NotImplementedError

    @pytest.mark.todo
    def test_elementwise_bips_are_physical(self, methanol_water_vle_data, bip_estimator_factory) -> None:
        """Wilson Lambdas from every isotherm are positive and finite."""
        raise NotImplementedError

    @pytest.mark.todo
    def test_regressed_model_reproduces_azeotrope_free_behaviour(self, methanol_water_vle_data, bip_estimator_factory) -> None:
        """Predicted y1 > x1 over the whole range for methanol-water (no azeotrope)."""
        raise NotImplementedError

    @pytest.mark.todo
    def test_results_cache_and_latest_results(self, methanol_water_vle_data, bip_estimator_factory) -> None:
        """After the run, cache holds ELEMENTWISE results and latest_regression_results is set."""
        raise NotImplementedError

    @pytest.mark.todo
    def test_regression_is_deterministic(self, methanol_water_vle_data, bip_estimator_factory) -> None:
        """Two runs on the same data give identical coefficients."""
        raise NotImplementedError


class TestGammaPhiConsistency:

    @pytest.mark.todo
    def test_regression_models_agree_with_core_property_models(self, methanol_water_vle_data, bip_estimator_factory) -> None:
        """Gammas from *ActivityModelRegression match core.properties models fed the same BIPs."""
        raise NotImplementedError
