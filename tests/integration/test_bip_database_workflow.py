"""Integration tests: regression results -> BIPDatabaseManager -> JSON file (temporary copy)."""
from __future__ import annotations

import pytest


class TestAddEntryFromRealEstimator:

    @pytest.mark.todo
    def test_new_pair_written_to_database(self, methanol_propanol_vle_data, bip_estimator_factory, empty_bip_database, scripted_input) -> None:
        """Elementwise result for methanol-n-propanol is saved under name and CAS keys."""
        raise NotImplementedError

    @pytest.mark.todo
    def test_metadata_reflects_estimator(self, methanol_water_vle_data, bip_estimator_factory, empty_bip_database, scripted_input) -> None:
        """Stored metadata (model, EoS, polynomial, T/P bounds, source) matches the estimator and VLE data."""
        raise NotImplementedError

    @pytest.mark.todo
    def test_existing_pair_overwrite_confirmed(self, methanol_water_vle_data, bip_estimator_factory, tmp_db_manager, scripted_input) -> None:
        """Existing methanol-water Wilson entry is replaced after answering 'y'."""
        raise NotImplementedError

    @pytest.mark.todo
    def test_existing_pair_overwrite_declined(self, methanol_water_vle_data, bip_estimator_factory, tmp_db_manager, scripted_input) -> None:
        """Answering 'n' raises OverwriteAttemptError and leaves the file byte-identical."""
        raise NotImplementedError

    @pytest.mark.todo
    def test_incompatible_polynomial_rejected(self, methanol_water_vle_data, bip_estimator_factory, empty_bip_database, scripted_input) -> None:
        """Wilson + PolynomialRegular raises PolynomialException; nothing is written."""
        raise NotImplementedError

    @pytest.mark.todo
    def test_reversed_component_order_in_csv(self, vle_csv_factory, bip_estimator_factory, empty_bip_database, scripted_input) -> None:
        """Data declared as (water, methanol) is stored under 'methanol-water' with correctly swapped BIPs."""
        raise NotImplementedError


class TestDatabaseRoundTrip:

    @pytest.mark.todo
    def test_saved_coefficients_reproduce_activity_coefficients(self, methanol_water_vle_data, bip_estimator_factory, empty_bip_database, scripted_input) -> None:
        """Gammas from reloaded JSON coefficients equal gammas from the in-memory regression result."""
        raise NotImplementedError

    @pytest.mark.todo
    def test_shipped_methanol_water_entry_reproduces_experimental_vle(self, methanol_water_vle_data, shipped_bip_database_path) -> None:
        """Shipped Wilson coefficients predict y1 of VLE_MeOH_H2O.csv within an agreed tolerance."""
        raise NotImplementedError

    @pytest.mark.todo
    def test_shipped_entry_goodness_of_fit_is_reproducible(self, methanol_water_vle_data, bip_estimator_factory, shipped_bip_database_path) -> None:
        """Recomputing R^2 from the shipped coefficients matches the stored goodness_of_fit."""
        raise NotImplementedError
