"""
End-to-end tests for the interactive CLI in regression_main.main().

All prompts are answered via ``scripted_input`` and the database manager is redirected to a
temporary file, so the shipped BIP_database.json is never modified.
"""
from __future__ import annotations

import pytest


@pytest.mark.todo
def test_main_elementwise_only_without_saving(monkeypatch, scripted_input) -> None:
    """Answers: close plots, 'n' (no memetic), 'n' (no save) -> completes, database untouched."""
    raise NotImplementedError


@pytest.mark.todo
@pytest.mark.slow
def test_main_full_pipeline_saves_to_temporary_database(monkeypatch, scripted_input, tmp_bip_database) -> None:
    """Answers: close plots, 'y' (memetic), close plots, 'y' (save), method '2', overwrite 'y'."""
    raise NotImplementedError


@pytest.mark.todo
def test_main_handles_invalid_answers(monkeypatch, scripted_input) -> None:
    """Unexpected answers (e.g. 'maybe') are treated as 'no' without crashing."""
    raise NotImplementedError


@pytest.mark.todo
def test_main_script_entry_point(monkeypatch, scripted_input) -> None:
    """Running the module via runpy as __main__ executes main()."""
    raise NotImplementedError
