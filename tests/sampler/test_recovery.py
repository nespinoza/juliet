"""Real fits with each installed sampler. Needs --run-sampler. Seeds can't be set for dynesty yet (issue #138)."""
import importlib

import pytest

from sampler.recovery import MAX_SIGMA, sigmas_from_truth

pytestmark = pytest.mark.sampler


@pytest.mark.parametrize('name,module', [
    ('dynesty', 'dynesty'),
    ('dynamic_dynesty', 'dynesty'),
    ('ultranest', 'ultranest'),
    ('emcee', 'emcee'),
    pytest.param('multinest', 'pymultinest', marks=pytest.mark.requires_multinest),
])
def test_recovers_injected_values(name, module, out_folder):
    pytest.importorskip(module)
    # MultiNest has a limit on the length of its paths, and going over it stops the whole process with a Fortran error.
    # Tests run inside their own temporary directory, so a relative path is short:
    sigmas = sigmas_from_truth(name, 'fit/' if name == 'multinest' else out_folder)
    assert all(value < MAX_SIGMA for value in sigmas.values()), sigmas
