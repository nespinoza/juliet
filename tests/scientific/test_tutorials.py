"""
Full fits of the paper's systems, compared with its tables and with a stored baseline, in units of the uncertainties.

Hours long, so they need --run-scientific. A nested-sampling posterior isn't deterministic, so these look for drift
that matters, not for exact numbers.
"""
import json
import os
from pathlib import Path

import pytest

from scientific.fit_tutorial import BASELINES, run_fit
from scientific.targets import PAPER, PROBLEMS

pytestmark = pytest.mark.scientific

# A single parameter can differ by more than this from the paper for reasons other than juliet: newer samplers and
# dependencies, and the assumed supersampling. A shift beyond this should be looked at by a person:
MAX_SIGMA_FROM_PAPER = 2.
MAX_SIGMA_FROM_BASELINE = 1.
NTHREADS = int(os.environ.get('JULIET_SCIENTIFIC_NTHREADS', 4))


def sigmas_apart(a, a_up, a_down, b, b_up, b_down):
    return abs(a - b) / max(0.5 * (a_up + a_down), 0.5 * (b_up + b_down))


@pytest.fixture(scope='module', params=sorted(PROBLEMS))
def fit(request, tmp_path_factory):
    folder = os.environ.get('JULIET_SCIENTIFIC_FOLDER')
    out = Path(folder) / request.param if folder else tmp_path_factory.mktemp(request.param)
    return request.param, run_fit(request.param, out, nthreads=NTHREADS)


def test_against_paper(fit):
    problem, summary = fit
    apart = {name: sigmas_apart(*summary['parameters'][name], *row) for name, row in PAPER[problem].items()}
    assert {name: value for name, value in apart.items() if value > MAX_SIGMA_FROM_PAPER} == {}


def test_against_baseline(fit):
    problem, summary = fit
    path = BASELINES / (problem + '.json')
    if not path.exists():
        pytest.skip('No stored baseline yet. Create one with tests/scientific/fit_tutorial.py --store.')
    with open(path) as f:
        baseline = json.load(f)
    apart = {name: sigmas_apart(*summary['parameters'][name], *row) for name, row in baseline['parameters'].items()}
    assert {name: value for name, value in apart.items() if value > MAX_SIGMA_FROM_BASELINE} == {}
    assert abs(summary['lnZ'] - baseline['lnZ']) < 3. * max(summary['lnZerr'], baseline['lnZerr'], 0.3)
