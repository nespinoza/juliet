"""
The suite's own protection against the stale-results behaviour (first rows of section 2.6 of the plan).

A fit without out_folder reads and writes results in the working directory. conftest.py moves every test into its own
temporary directory, and fails any test that leaves result files there.
"""
import os
import re
from pathlib import Path

import pytest

import juliet
from contract.helpers import write_results
from support.build import make_priors, make_samples, transit_parameters
from support.datasets import lc_data

pytestmark = pytest.mark.contract

TESTS = Path(__file__).parents[1]
REPOSITORY = TESTS.parent


def test_tests_run_in_their_own_empty_directory(tmp_path):
    assert Path(os.getcwd()).resolve() == tmp_path.resolve()
    assert REPOSITORY.resolve() not in Path(os.getcwd()).resolve().parents
    assert list(tmp_path.iterdir()) == []


def test_load_alone_writes_nothing_without_out_folder(tmp_path):
    juliet.load(priors=make_priors(transit_parameters()), **lc_data())
    assert list(tmp_path.iterdir()) == []


@pytest.mark.writes_to_cwd
def test_a_fit_without_out_folder_uses_the_working_directory(tmp_path):
    """Frozen: this is how the tutorials without out_folder behave. The guard in conftest.py relies on it."""
    params = transit_parameters()
    write_results(str(tmp_path) + '/', make_samples(params, nsamples=20))
    results = juliet.load(priors=make_priors(params), **lc_data()).fit(sampler='dynesty')
    assert Path(results.out_folder).resolve() == tmp_path.resolve()
    assert (tmp_path / 'posteriors.dat').exists()


def test_every_fit_in_the_suite_has_an_out_folder():
    """A .fit( call whose dataset was loaded without out_folder or input_folder, in the same function."""
    offenders = []
    for path in sorted(TESTS.rglob('*.py')):
        if path.name == Path(__file__).name:
            continue
        for function in re.split(r'\n(?=def |class |@)', path.read_text()):
            if '.fit(' in function and 'juliet.load(' in function and 'writes_to_cwd' not in function:
                for call in re.findall(r'juliet\.load\((?:[^()]|\([^()]*\))*\)', function):
                    if 'out_folder' not in call and 'input_folder' not in call:
                        offenders.append('{}: {}'.format(path.relative_to(TESTS), call[:80]))
    assert offenders == []
