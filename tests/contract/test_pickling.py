"""
Samplers with pools (dynesty and emcee with nthreads, zeus) send fit.loglike to other processes with pickle, so it must
survive the trip with every kind of noise model.
"""
import pickle

import numpy as np
import pytest

from contract.helpers import fit_without_sampling, free_vector, restore_parameters
from support.build import rv_parameters, transit_parameters
from support.datasets import lc_data, rv_data, stack
from support.kernels import gp_parameters

pytestmark = pytest.mark.contract

INSTRUMENTS = ('X', 'Y')


def photometry(kernel=None, mode='instrument'):
    data, params = lc_data(INSTRUMENTS, n=50), transit_parameters(INSTRUMENTS)
    if kernel and mode == 'global':
        data['GP_regressors_lc'] = {'lc': stack(data['t_lc'], INSTRUMENTS)}
        params.update(gp_parameters(kernel, 'lc', 'lc'))
    elif kernel:
        data['GP_regressors_lc'] = {name: data['t_lc'][name] for name in INSTRUMENTS}
        for name in INSTRUMENTS:
            params.update(gp_parameters(kernel, 'lc', name))
    return params, data


def velocities(kernel=None):
    data, params = rv_data(INSTRUMENTS), rv_parameters(INSTRUMENTS)
    if kernel:
        data['GP_regressors_rv'] = {'rv': stack(data['t_rv'], INSTRUMENTS)}
        params.update(gp_parameters(kernel, 'rv', 'rv'))
    return params, data


def joint():
    lc_params, lc_inputs = photometry('CeleriteMaternKernel')
    rv_params, rv_inputs = velocities('SEKernel')
    return dict(rv_params, **lc_params), dict(lc_inputs, **rv_inputs)


CASES = {
    'lc': photometry,
    'lc_celerite_gp': lambda: photometry('CeleriteMaternKernel'),
    'lc_george_gp': lambda: photometry('SEKernel'),
    'lc_celerite_global_gp': lambda: photometry('CeleriteMaternKernel', 'global'),
    'lc_george_global_gp': lambda: photometry('SEKernel', 'global'),
    'rv': velocities,
    'rv_celerite_global_gp': lambda: velocities('CeleriteMaternKernel'),
    'rv_george_global_gp': lambda: velocities('SEKernel'),
    'joint_with_gps': joint,
}


def other_point(params):
    """A second, valid parameter vector: one that the pickled copy hasn't been evaluated at before pickling."""
    changes = {name: value * 1.01 for name, value in params.items()
               if name.startswith(('sigma_w', 'p_p', 'K_p', 'GP_sigma', 'mflux'))}
    return dict(params, **changes)


@pytest.mark.parametrize('case', CASES)
@pytest.mark.parametrize('sampler', ['dynesty', 'emcee'])
def test_loglike_survives_pickling(case, sampler, out_folder):
    params, data = CASES[case]()
    results = restore_parameters(fit_without_sampling(params, out_folder, sampler=sampler, **data))
    theta, other = free_vector(results, params), free_vector(results, other_point(params))
    expected, expected_other = results.loglike(theta), results.loglike(other)
    assert np.isfinite(expected) and expected != expected_other
    copy = pickle.loads(pickle.dumps(results.loglike))
    assert copy(other) == expected_other
    assert copy(theta) == expected
    # The original still works, and doesn't share state with the copy:
    assert results.loglike(theta) == expected


@pytest.mark.parametrize('case', ['lc_celerite_gp', 'rv_george_global_gp'])
def test_logprob_and_prior_transform_survive_pickling(case, out_folder):
    params, data = CASES[case]()
    mcmc = restore_parameters(fit_without_sampling(params, out_folder + 'mcmc/', sampler='emcee', **data))
    theta = free_vector(mcmc, params)
    assert pickle.loads(pickle.dumps(mcmc.logprob))(theta) == mcmc.logprob(theta)
    nested = restore_parameters(fit_without_sampling(params, out_folder + 'nested/', sampler='dynesty', **data))
    cube = np.linspace(0.2, 0.8, len(theta))
    np.testing.assert_array_equal(pickle.loads(pickle.dumps(nested.prior_transform_r))(cube),
                                  nested.prior_transform_r(cube))


# (Global GPs are left out: evaluate() without t fails for them today. See GLOBAL_STATE in test_evaluate_matrix.py.)
@pytest.mark.parametrize('case', ['lc', 'lc_celerite_gp', 'lc_george_gp', 'rv'])
def test_results_object_survives_pickling(case, out_folder):
    """Users save fit results with pickle to plot them later."""
    params, data = CASES[case]()
    results = fit_without_sampling(params, out_folder, **data)
    copy = pickle.loads(pickle.dumps(results))
    for kind in ('lc', 'rv'):
        if hasattr(results, kind):
            original, copied = getattr(results, kind), getattr(copy, kind)
            for instrument in original.inames:
                np.testing.assert_array_equal(copied.evaluate(instrument, all_samples=True),
                                              original.evaluate(instrument, all_samples=True))
