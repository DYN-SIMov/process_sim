"""Unit tests for thermodynamics.activity_models_regression.data_handling."""
from __future__ import annotations

import pytest


# --------------------------------------------------------------------------------------
# DataPoint / TxyPoint
# --------------------------------------------------------------------------------------
class TestDataPoint:

    @pytest.mark.todo
    def test_is_immutable(self, data_point_factory) -> None:
        """DataPoint is a frozen dataclass: assigning a field raises FrozenInstanceError."""
        raise NotImplementedError

    @pytest.mark.todo
    def test_equality_and_hashability(self, data_point_factory) -> None:
        """Two DataPoints with identical fields compare equal and have the same hash."""
        raise NotImplementedError


class TestTxyPoint:

    @pytest.mark.todo
    def test_temperature_taken_from_first_data_point(self, data_point_factory) -> None:
        """TxyPoint.temperature_K equals the temperature of data_points[0]."""
        raise NotImplementedError

    @pytest.mark.todo
    def test_saturation_pressures_default_to_none(self, data_point_factory) -> None:
        """comp_1/comp_2 saturation pressures are None until VLEData populates them."""
        raise NotImplementedError

    @pytest.mark.todo
    def test_empty_data_points_raises(self) -> None:
        """Constructing a TxyPoint from an empty list raises (IndexError today; document behaviour)."""
        raise NotImplementedError


# --------------------------------------------------------------------------------------
# RawExperimentalData
# --------------------------------------------------------------------------------------
class TestRawExperimentalDataFiltering:

    @pytest.mark.todo
    @pytest.mark.parametrize("x1", [0.0, 1.0, 5e-4, 1 - 5e-4])
    def test_removes_points_with_extreme_liquid_composition(self, data_point_factory, x1) -> None:
        """Points with x1 outside [tol, 1 - tol] are dropped and a warning is printed."""
        raise NotImplementedError

    @pytest.mark.todo
    @pytest.mark.parametrize("y1", [0.0, 1.0, 5e-4, 1 - 5e-4])
    def test_removes_points_with_extreme_vapour_composition(self, data_point_factory, y1) -> None:
        """Points with y1 outside [tol, 1 - tol] are dropped and a warning is printed."""
        raise NotImplementedError

    @pytest.mark.todo
    def test_tolerance_boundaries_are_inclusive(self, data_point_factory) -> None:
        """Points with x1/y1 exactly equal to tol or 1 - tol are kept."""
        raise NotImplementedError

    @pytest.mark.todo
    def test_custom_concentration_tolerance(self, data_point_factory) -> None:
        """A larger concentration_tol removes more points than the default."""
        raise NotImplementedError

    @pytest.mark.todo
    def test_no_filtering_when_disabled(self, data_point_factory) -> None:
        """remove_extreme_concentrations=False keeps every data point unchanged."""
        raise NotImplementedError

    @pytest.mark.todo
    def test_default_data_points_not_shared_between_instances(self) -> None:
        """Mutable default argument (data_points=[]) must not leak state between instances."""
        raise NotImplementedError


class TestFindConstantTemperaturePoints:

    @pytest.mark.todo
    def test_groups_consecutive_points_at_same_temperature(self, data_point_factory) -> None:
        """Isothermal data at N temperatures yields N TxyPoints with the right members."""
        raise NotImplementedError

    @pytest.mark.todo
    def test_isobaric_data_yields_one_txy_point_per_temperature(self, data_point_factory) -> None:
        """Isobaric data (every point at a distinct T) yields one TxyPoint per data point."""
        raise NotImplementedError

    @pytest.mark.todo
    @pytest.mark.parametrize("temperature_K_tol", [1e-3, 0.1, 1.0])
    def test_temperature_tolerance_is_respected(self, data_point_factory, temperature_K_tol) -> None:
        """Temperatures within temperature_K_tol are grouped; beyond it they are split."""
        raise NotImplementedError

    @pytest.mark.todo
    def test_non_consecutive_repeated_temperature(self, data_point_factory) -> None:
        """Sequence T0, T1, T0: the second T0 point must not be merged into the T1 group."""
        raise NotImplementedError

    @pytest.mark.todo
    def test_preserves_total_number_of_points(self, data_point_factory) -> None:
        """Sum of len(txy.data) over all TxyPoints equals the number of input points."""
        raise NotImplementedError

    @pytest.mark.todo
    def test_raises_when_no_data_points(self) -> None:
        """Empty RawExperimentalData raises VLEDataError."""
        raise NotImplementedError

    @pytest.mark.todo
    def test_warns_when_only_one_isotherm(self, data_point_factory, capsys) -> None:
        """A single isotherm prints a 'lack of temperature variability' warning."""
        raise NotImplementedError

    @pytest.mark.todo
    def test_single_isotherm_with_single_point_does_not_crash(self, data_point_factory) -> None:
        """Warning message indexes constant_temperature_points[1]; one point must not raise IndexError."""
        raise NotImplementedError


# --------------------------------------------------------------------------------------
# VLEData - CSV header parsing (static methods, no thermo calls)
# --------------------------------------------------------------------------------------
class TestParseComponentsFromComment:

    @pytest.mark.todo
    @pytest.mark.parametrize(
        "header_line",
        ["# Component 1: methanol", "#component: methanol", "# CHEMICAL methanol", "#   Chemical 1 : methanol"],
    )
    def test_accepts_supported_header_variants(self, vle_csv_factory, header_line) -> None:
        """'component'/'chemical' keywords are matched case-insensitively with optional spacing."""
        raise NotImplementedError

    @pytest.mark.todo
    def test_returns_components_in_file_order(self, vle_csv_factory) -> None:
        """Components are returned in the order they appear in the header."""
        raise NotImplementedError

    @pytest.mark.todo
    @pytest.mark.parametrize("n_components", [0, 1, 3])
    def test_raises_unless_exactly_two_components(self, vle_csv_factory, n_components) -> None:
        """VLEDataError is raised for anything other than a binary system."""
        raise NotImplementedError

    @pytest.mark.todo
    def test_multi_word_component_names(self, vle_csv_factory) -> None:
        """Document behaviour for names with spaces (e.g. 'acetic acid'); only the last token is kept today."""
        raise NotImplementedError

    @pytest.mark.todo
    def test_ignores_non_comment_lines(self, vle_csv_factory) -> None:
        """Lines not starting with '#' (e.g. a data column named 'component') are ignored."""
        raise NotImplementedError


class TestParseSourceFromComment:

    @pytest.mark.todo
    @pytest.mark.parametrize("keyword", ["Source", "source", "Reference", "REFERENCE"])
    def test_extracts_source_text(self, vle_csv_factory, keyword) -> None:
        """Text after 'Source:'/'Reference:' is returned stripped."""
        raise NotImplementedError

    @pytest.mark.todo
    def test_default_when_source_missing(self, vle_csv_factory) -> None:
        """Returns 'no data source provided' when no source line exists."""
        raise NotImplementedError

    @pytest.mark.todo
    def test_returns_first_match_only(self, vle_csv_factory) -> None:
        """When several source lines exist, the first one is returned."""
        raise NotImplementedError

    @pytest.mark.todo
    def test_source_containing_commas_and_colons(self, vle_csv_factory) -> None:
        """Full source text (with commas, colons, URLs) is preserved."""
        raise NotImplementedError


# --------------------------------------------------------------------------------------
# VLEData - dataframe extraction and saturation pressures (thermo backends mocked)
# --------------------------------------------------------------------------------------
class TestExtractDataFromDataFrame:

    @pytest.mark.todo
    def test_unit_conversion(self) -> None:
        """'P (atm)' is multiplied by 1e5 and 'T (degC)' shifted by 273.15 (note: atm vs bar factor)."""
        raise NotImplementedError

    @pytest.mark.todo
    def test_complementary_mole_fractions(self) -> None:
        """x2 = 1 - x1 and y2 = 1 - y1 for every DataPoint."""
        raise NotImplementedError

    @pytest.mark.todo
    def test_composition_columns_of_second_component(self) -> None:
        """If only x_/y_ columns of component 2 exist they are used; document whether x1 is then inverted."""
        raise NotImplementedError

    @pytest.mark.todo
    def test_raises_when_no_matching_composition_columns(self) -> None:
        """VLEDataError is raised when no x_<comp>/y_<comp> pair is present."""
        raise NotImplementedError

    @pytest.mark.todo
    def test_respects_remove_extreme_concentrations_flag(self) -> None:
        """The VLEData flag is forwarded to RawExperimentalData."""
        raise NotImplementedError


class TestSaturationPressureData:

    @pytest.mark.todo
    def test_assigns_saturation_pressures_to_each_txy_point(self, monkeypatch) -> None:
        """comp_1/comp_2 saturation pressures are set from _get_saturation_pressure_value."""
        raise NotImplementedError

    @pytest.mark.todo
    def test_removes_txy_points_with_nan_saturation_pressure(self, monkeypatch) -> None:
        """TxyPoints for which any component returns NaN are removed from T_x_y_points."""
        raise NotImplementedError

    @pytest.mark.todo
    def test_uses_thermo_when_successful(self, monkeypatch) -> None:
        """CoolProp fallback is not called when the thermo backend succeeds."""
        raise NotImplementedError

    @pytest.mark.todo
    def test_falls_back_to_coolprop_when_thermo_fails(self, monkeypatch) -> None:
        """CoolProp backend is called when the thermo backend reports failure."""
        raise NotImplementedError


class TestSaturationPressureThermoBackend:

    @pytest.mark.todo
    def test_returns_nan_and_failure_on_value_error(self, monkeypatch) -> None:
        """thermo.Chemical raising ValueError yields (nan, False)."""
        raise NotImplementedError

    @pytest.mark.todo
    def test_returns_psat_within_valid_range(self, monkeypatch) -> None:
        """Within [Tmin, Tmax] the Chemical.Psat value is returned without prompting."""
        raise NotImplementedError

    @pytest.mark.todo
    @pytest.mark.parametrize(
        ("answer", "expect_nan"), [("y", True), ("Y", True), ("n", False), ("maybe", True)]
    )
    def test_out_of_range_temperature_prompts_user(self, monkeypatch, scripted_input, answer, expect_nan) -> None:
        """Out-of-range T prompts the user: 'y'/invalid -> skip (nan), 'n' -> extrapolated Psat."""
        raise NotImplementedError


class TestSaturationPressureCoolPropBackend:

    @pytest.mark.todo
    def test_returns_value_on_success(self, monkeypatch) -> None:
        """PropsSI result is returned together with success=True."""
        raise NotImplementedError

    @pytest.mark.todo
    def test_returns_nan_and_failure_on_value_error(self, monkeypatch) -> None:
        """PropsSI raising ValueError yields (nan, False) and prints a warning."""
        raise NotImplementedError
