"""
The frozen-numerics layer: fixed parameters in, the same numbers out.

A failure here means a change altered results. Unless that's a reviewed science change, the change is wrong and the
baseline is right. See tests/README.md before using --update-golden.
"""
import re
from pathlib import Path

import numpy as np
import pytest

from golden.compute import compute
from golden.configs import CONFIGS
from golden.tolerances import ARRAY_ABS, ARRAY_REL, LOGLIKE_ABS, LOGLIKE_REL

DATA = Path(__file__).parent / 'data'
# With log_like_calc=True juliet doesn't fill these per-planet arrays, on purpose: they are placeholders.
NOT_FILLED_DURING_FITS = re.compile(r'/model/(\w+/)?p\d+$')

pytestmark = pytest.mark.golden


PARAMETERS = [pytest.param(config, id=config.id,
                           marks=[getattr(pytest.mark, config.requires)] if config.requires else [])
              for config in CONFIGS]


def assert_close(name, computed, stored):
    if name.endswith('loglike'):
        np.testing.assert_allclose(computed, stored, rtol=LOGLIKE_REL, atol=LOGLIKE_ABS, err_msg=name)
    else:
        np.testing.assert_allclose(computed, stored, rtol=ARRAY_REL, atol=ARRAY_ABS, err_msg=name)


def load_baseline(config):
    path = DATA / (config.id + '.npz')
    if not path.exists():
        message = 'No baseline for {}: generate one with --update-golden.'.format(config.id)
        if config.requires:
            pytest.skip(message + ' It depends on an optional package that was missing when baselines were last made.')
        pytest.fail(message)
    with np.load(path) as stored:
        return dict(stored)


@pytest.mark.parametrize('config', PARAMETERS)
def test_against_baseline(config, request):
    computed = compute(config, log_like_calc=False)
    if request.config.getoption('--update-golden'):
        DATA.mkdir(exist_ok=True)
        np.savez_compressed(DATA / (config.id + '.npz'), **computed)
        request.config.golden_files_written.append(config.id)
        return
    stored = load_baseline(config)
    assert sorted(computed) == sorted(stored), 'Quantities computed for this configuration changed.'
    for name in stored:
        assert_close(name, computed[name], stored[name])


@pytest.mark.parametrize('config', PARAMETERS)
def test_log_like_calc_agrees(config):
    """During a fit juliet runs separate copies of the model code (log_like_calc=True). They must give the same answer."""
    fast, full = compute(config, log_like_calc=True), compute(config, log_like_calc=False)
    assert any(name.endswith('loglike') for name in fast) or not float(fast[config.modeltypes[0] + '/modelOK'])
    for name in sorted(set(fast) & set(full)):
        if NOT_FILLED_DURING_FITS.search(name):
            continue
        assert_close(name, fast[name], full[name])
