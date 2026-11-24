"""Guards the guard: every other test is meaningless if it imports a different juliet."""
from pathlib import Path

import pytest

import juliet


@pytest.mark.unit
def test_tests_import_this_checkout():
    assert Path(juliet.__file__).resolve().parent == Path(__file__).resolve().parents[2] / 'juliet'
