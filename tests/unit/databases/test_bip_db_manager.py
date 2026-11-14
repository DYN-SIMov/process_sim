"""Unit tests for thermodynamics.databases.bip_db_manager."""
from __future__ import annotations

import pytest


class TestEnums:

    @pytest.mark.todo
    def test_activity_model_enum_matches_core_class_names(self) -> None:
        """ActivityModel values equal WilsonActivityModel/NRTLActivityModel class names (DB section keys)."""
        raise NotImplementedError

    @pytest.mark.todo
    def test_polynomial_enum_matches_polynomial_class_names(self) -> None:
        """PolynimalType values equal the polynomial class names stored in metadata."""
        raise NotImplementedError


class TestLoadDatabase:

    @pytest.mark.todo
    def test_creates_empty_file_when_missing(self, tmp_path) -> None:
        """A non-existent path is created containing '{}' and data == {}."""
        raise NotImplementedError

    @pytest.mark.todo
    def test_reads_existing_database(self, tmp_bip_database) -> None:
        """Existing JSON content is loaded into manager.data."""
        raise NotImplementedError

    @pytest.mark.todo
    def test_invalid_json_raises(self, tmp_path) -> None:
        """Corrupted file raises json.JSONDecodeError (or a domain-specific error, if introduced)."""
        raise NotImplementedError

    @pytest.mark.todo
    def test_initial_selection_state_is_unset(self, db_manager) -> None:
        """activity_model and polynomial are None after construction."""
        raise NotImplementedError


class TestSelectionGetters:

    @pytest.mark.todo
    def test_get_selected_activity_model_raises_when_unset(self, db_manager) -> None:
        """RuntimeError before add_entry() has set the activity model."""
        raise NotImplementedError

    @pytest.mark.todo
    def test_get_selected_polynomial_raises_when_unset(self, db_manager) -> None:
        """RuntimeError before add_entry() has set the polynomial."""
        raise NotImplementedError

    @pytest.mark.todo
    @pytest.mark.parametrize(("activity_model", "expected"), [("wilson", "WILSON"), ("nrtl", "NRTL")])
    def test_get_activity_model_from_estimator(self, db_manager, fake_bip_estimator_factory, activity_model, expected) -> None:
        """Resolved from the first base class of the regression model."""
        raise NotImplementedError

    @pytest.mark.todo
    @pytest.mark.parametrize("polynomial_name", ["PolynomialExponentialDIPPR", "PolynomialNRTL", "PolynomialRegular"])
    def test_get_polynomial_from_estimator(self, db_manager, fake_bip_estimator_factory, polynomial_name) -> None:
        """Resolved from the wrapped polynomial class name (independent of Absolute/Normalized form)."""
        raise NotImplementedError


class TestComponentPairKey:

    @pytest.mark.todo
    def test_names_sorted_alphabetically(self, db_manager) -> None:
        """('water', 'methanol') -> 'methanol-water' with CAS in the same order."""
        raise NotImplementedError

    @pytest.mark.todo
    def test_key_is_order_independent(self, db_manager) -> None:
        """(A, B) and (B, A) produce identical keys."""
        raise NotImplementedError

    @pytest.mark.todo
    def test_unknown_component_raises(self, db_manager) -> None:
        """CAS_from_any raises ValueError itself; the 'is None' branch is dead code - document/align."""
        raise NotImplementedError

    @pytest.mark.todo
    def test_synonyms_map_to_same_cas_key(self, db_manager) -> None:
        """e.g. 'methanol'/'methyl alcohol' give different name keys but the same CAS key."""
        raise NotImplementedError


class TestExtractMetadata:

    @pytest.mark.todo
    def test_contains_all_expected_fields(self, db_manager, fake_bip_estimator_factory) -> None:
        """Metadata keys match the schema used in the shipped BIP_database.json."""
        raise NotImplementedError

    @pytest.mark.todo
    def test_temperature_and_pressure_ranges(self, db_manager, fake_bip_estimator_factory) -> None:
        """Lower/upper T and P bounds equal min/max over the TxyPoints actually used in the regression."""
        raise NotImplementedError

    @pytest.mark.todo
    def test_regression_info_taken_from_result_to_save(self, db_manager, fake_bip_estimator_factory) -> None:
        """goodness_of_fit and regression_method come from result_to_save, not from the cache."""
        raise NotImplementedError

    @pytest.mark.todo
    def test_date_of_entry_creation_is_iso_format(self, db_manager, fake_bip_estimator_factory) -> None:
        """date_of_entry_creation parses with datetime.fromisoformat."""
        raise NotImplementedError


class TestExtractBIPData:

    @pytest.mark.todo
    def test_values_keyed_by_bip_name(self, db_manager, fake_bip_estimator_factory) -> None:
        """Every regressed BIP name appears with its value when components are already sorted."""
        raise NotImplementedError

    @pytest.mark.todo
    def test_numpy_values_are_json_serialisable(self, db_manager, fake_bip_estimator_factory) -> None:
        """ndarray values are converted to lists; json.dumps succeeds."""
        raise NotImplementedError

    @pytest.mark.todo
    @pytest.mark.parametrize("activity_model", ["wilson", "nrtl"])
    def test_parameters_relabelled_when_components_swapped(self, db_manager, fake_bip_estimator_factory, activity_model) -> None:
        """
        For ('water', 'methanol') the stored 'Lambda_12' must be the original 'Lambda_21' value.
        The current pairwise swap reorders BIP objects but keeps their names, so the mapping is unchanged.
        """
        raise NotImplementedError


class TestExportChecks:

    @pytest.mark.todo
    @pytest.mark.parametrize(
        ("activity_model", "polynomial_name", "should_raise"),
        [
            ("wilson", "PolynomialExponentialDIPPR", False),
            ("wilson", "PolynomialNRTL", True),
            ("wilson", "PolynomialRegular", True),
            ("nrtl", "PolynomialNRTL", False),
            ("nrtl", "PolynomialExponentialDIPPR", True),
            ("nrtl", "PolynomialRegular", True),
        ],
    )
    def test_polynomial_compatibility(self, db_manager, fake_bip_estimator_factory, activity_model, polynomial_name, should_raise) -> None:
        """PolynomialException for incompatible activity model / polynomial combinations."""
        raise NotImplementedError

    @pytest.mark.todo
    def test_model_compatibility_raises_for_missing_section(self, tmp_path) -> None:
        """ValueError when the database has no top-level section for the selected activity model."""
        raise NotImplementedError


class TestOverwriteHandling:

    @pytest.mark.todo
    def test_no_prompt_when_entry_absent(self, db_manager) -> None:
        """No input() call and no error when neither name nor CAS key exists."""
        raise NotImplementedError

    @pytest.mark.todo
    @pytest.mark.parametrize("existing_key", ["name", "CAS"])
    def test_force_overwrite_skips_prompt(self, db_manager, existing_key) -> None:
        """force_overwrite=True never prompts, regardless of which key exists."""
        raise NotImplementedError

    @pytest.mark.todo
    @pytest.mark.parametrize("existing_key", ["name", "CAS"])
    def test_user_declines_overwrite_raises(self, db_manager, scripted_input, existing_key) -> None:
        """Answer 'n' raises OverwriteAttemptError."""
        raise NotImplementedError

    @pytest.mark.todo
    @pytest.mark.parametrize("existing_key", ["name", "CAS"])
    def test_user_accepts_overwrite(self, db_manager, scripted_input, existing_key) -> None:
        """Answer 'y' passes without error."""
        raise NotImplementedError

    @pytest.mark.todo
    def test_prompts_only_once_when_both_keys_exist(self, db_manager, scripted_input) -> None:
        """When name and CAS keys both exist the user is asked a single time."""
        raise NotImplementedError

    @pytest.mark.todo
    @pytest.mark.parametrize(("answer", "expected"), [("y", True), ("Y", True), ("n", False), ("", False), ("yes", False)])
    def test_offer_overwrite_answer_parsing(self, db_manager, scripted_input, answer, expected) -> None:
        """Only 'y'/'Y' confirms the overwrite."""
        raise NotImplementedError


class TestRegressionMethodSelection:

    @pytest.mark.todo
    @pytest.mark.parametrize("answer", ["1", "2"])
    def test_valid_selection_sets_result_to_save(self, db_manager, fake_bip_estimator_factory, scripted_input, answer) -> None:
        """result_to_save holds the chosen method's value, BIPs and goodness of fit."""
        raise NotImplementedError

    @pytest.mark.todo
    @pytest.mark.parametrize("answer", ["0", "3", "-1"])
    def test_out_of_range_selection_defaults_to_first(self, db_manager, fake_bip_estimator_factory, scripted_input, answer) -> None:
        """Out-of-range index falls back to the first cached method."""
        raise NotImplementedError

    @pytest.mark.todo
    @pytest.mark.parametrize("answer", ["", "abc", "1.5"])
    def test_non_numeric_selection_defaults_to_first(self, db_manager, fake_bip_estimator_factory, scripted_input, answer) -> None:
        """int() ValueError is not caught today (only RegressionMethodSelectionError is) - should default."""
        raise NotImplementedError

    @pytest.mark.todo
    def test_empty_results_cache(self, db_manager, fake_bip_estimator_factory, scripted_input) -> None:
        """No regression results available -> clear error instead of IndexError."""
        raise NotImplementedError

    @pytest.mark.todo
    def test_lists_available_results(self, db_manager, fake_bip_estimator_factory, scripted_input, capsys) -> None:
        """Every cached method is printed with BIP values and R^2."""
        raise NotImplementedError


class TestInsertAndSave:

    @pytest.mark.todo
    def test_insert_entry_under_name_and_cas_keys(self, db_manager) -> None:
        """Entry is stored twice (name key and CAS key) with identical content."""
        raise NotImplementedError

    @pytest.mark.todo
    def test_save_database_writes_json(self, db_manager) -> None:
        """File content equals manager.data after _save_database()."""
        raise NotImplementedError

    @pytest.mark.todo
    def test_saved_file_reloads_identically(self, db_manager) -> None:
        """A new manager on the same path reads back the saved data."""
        raise NotImplementedError


class TestAddEntry:

    @pytest.mark.todo
    def test_add_entry_happy_path(self, db_manager, fake_bip_estimator_factory, scripted_input) -> None:
        """New pair is written to disk under both keys with metadata and BIP_data."""
        raise NotImplementedError

    @pytest.mark.todo
    def test_add_entry_incompatible_polynomial_leaves_file_untouched(self, db_manager, fake_bip_estimator_factory, scripted_input) -> None:
        """PolynomialException is raised and the database file is unchanged."""
        raise NotImplementedError

    @pytest.mark.todo
    def test_add_entry_missing_model_section_raises_value_error(self, tmp_path, fake_bip_estimator_factory, scripted_input) -> None:
        """_check_overwrite runs before _check_model_compatibility -> KeyError today instead of ValueError."""
        raise NotImplementedError

    @pytest.mark.todo
    def test_add_entry_declined_overwrite_leaves_file_untouched(self, db_manager, fake_bip_estimator_factory, scripted_input) -> None:
        """OverwriteAttemptError is raised and the existing entry is preserved."""
        raise NotImplementedError


class TestQueryAPI:

    @pytest.mark.todo
    def test_get_entry_by_name(self, tmp_bip_database) -> None:
        """Placeholder: get_entry_by_name is not implemented yet (returns None)."""
        raise NotImplementedError

    @pytest.mark.todo
    def test_get_entry_by_CAS(self, tmp_bip_database) -> None:
        """Placeholder: get_entry_by_CAS is not implemented yet (returns None)."""
        raise NotImplementedError
