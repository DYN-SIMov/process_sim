"""Integration tests: loading VLE CSV files with real saturation-pressure backends."""
from __future__ import annotations

import pytest


class TestShippedDatasets:

    @pytest.mark.todo
    def test_loads_without_error(self, shipped_vle_csv, scripted_input) -> None:
        """Every shipped CSV loads into VLEData (answer any out-of-range Psat prompts)."""
        raise NotImplementedError

    @pytest.mark.todo
    def test_components_and_source_parsed(self, shipped_vle_csv, scripted_input) -> None:
        """Two components recognised by chemicals.CAS_from_any; source differs from the default."""
        raise NotImplementedError

    @pytest.mark.todo
    def test_txy_points_have_positive_saturation_pressures(self, shipped_vle_csv, scripted_input) -> None:
        """Every remaining TxyPoint has finite, positive Psat for both components."""
        raise NotImplementedError

    @pytest.mark.todo
    def test_extreme_concentration_rows_removed(self, methanol_water_vle_data) -> None:
        """The x=0 / y=0 pure-water row of VLE_MeOH_H2O.csv is filtered out."""
        raise NotImplementedError

    @pytest.mark.todo
    def test_isobaric_dataset_produces_one_txy_point_per_row(self, methanol_water_vle_data) -> None:
        """Isobaric methanol-water data gives one TxyPoint per (filtered) row."""
        raise NotImplementedError

    @pytest.mark.todo
    def test_isothermal_dataset_groups_rows(self, thermo_data_dir, scripted_input) -> None:
        """Water-ammonia data (several rows per temperature) is grouped into isotherms."""
        raise NotImplementedError

    @pytest.mark.todo
    def test_trailing_empty_columns_are_tolerated(self, methanol_water_vle_data) -> None:
        """Trailing ',,,,' columns in the CSV do not break parsing."""
        raise NotImplementedError


class TestSaturationPressureBackends:

    @pytest.mark.todo
    @pytest.mark.parametrize(("component", "temperature_K"), [("methanol", 337.85), ("water", 373.15)])
    def test_thermo_psat_at_normal_boiling_point(self, component, temperature_K) -> None:
        """Psat at the normal boiling point is ~101325 Pa."""
        raise NotImplementedError

    @pytest.mark.todo
    @pytest.mark.parametrize("component", ["methanol", "water"])
    def test_thermo_and_coolprop_agree(self, component) -> None:
        """Thermo and CoolProp Psat agree within a small relative tolerance."""
        raise NotImplementedError

    @pytest.mark.todo
    def test_supercritical_temperature_is_dropped(self, vle_csv_factory, scripted_input) -> None:
        """TxyPoints above a component's critical temperature are removed (or user-skipped)."""
        raise NotImplementedError


class TestSyntheticFiles:

    @pytest.mark.todo
    def test_roundtrip_from_generated_csv(self, vle_csv_factory) -> None:
        """A CSV written by vle_csv_factory loads with expected components, points and units."""
        raise NotImplementedError

    @pytest.mark.todo
    def test_missing_composition_columns_raise(self, vle_csv_factory) -> None:
        """VLEDataError when composition column names don't match the declared components."""
        raise NotImplementedError

    @pytest.mark.todo
    def test_unknown_component_name_raises(self, vle_csv_factory) -> None:
        """Unrecognised component names fail with a clear error during Psat evaluation."""
        raise NotImplementedError
