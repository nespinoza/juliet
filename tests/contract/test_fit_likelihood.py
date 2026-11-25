"""
The functions juliet.fit hands to samplers: prior_transform, logprior, loglike and logprob.

The golden layer freezes juliet.model. This checks the layer above it, which maps a sampler's vector to named
parameters, skips fixed ones and adds the datasets up. The fit objects are made from a results file, so no sampler runs.
"""
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

import juliet
from contract.helpers import fit_without_sampling, free_vector, restore_parameters
from golden.configs import CONFIGS
from golden.tolerances import CATWOMAN_LOGLIKE_REL, LOGLIKE_ABS, LOGLIKE_REL
from support.build import rv_parameters, transit_parameters
from support.datasets import lc_data, rv_data

pytestmark = pytest.mark.contract

BASELINES = Path(__file__).parents[1] / 'golden' / 'data'
REFUSED = -1e101

WITH_LIKELIHOOD = [pytest.param(config, id=config.id,
                                marks=[getattr(pytest.mark, config.requires)] if config.requires else [])
                   for config in CONFIGS if config.id != 'lc_model_refused_above_ecclim']


def ready_fit(config, out_folder, **kwargs):
    results = fit_without_sampling(config.params, out_folder, fixed=config.fixed, fit_kwargs=dict(config.model_kwargs),
                                   **dict(config.data(), **kwargs))
    return restore_parameters(results)


@pytest.mark.parametrize('config', WITH_LIKELIHOOD)
def test_loglike_equals_the_frozen_model_likelihood(config, out_folder):
    path = BASELINES / (config.id + '.npz')
    if not path.exists():
        pytest.skip('No baseline for this configuration.')
    with np.load(path) as stored:
        expected = sum(float(stored[kind + '/loglike']) for kind in config.modeltypes)
    results = ready_fit(config, out_folder)
    rtol = CATWOMAN_LOGLIKE_REL if config.requires == 'requires_catwoman' else LOGLIKE_REL
    # Twice the tolerance of one dataset, because a joint fit adds two of them:
    np.testing.assert_allclose(results.loglike(free_vector(results, config.params)), expected, rtol=2 * rtol,
                               atol=2 * LOGLIKE_ABS)


def test_loglike_skips_fixed_parameters(out_folder):
    """The sampler's vector only has the free parameters, in the order of the priors."""
    params = transit_parameters()
    fixed = ('P_p1', 'ecc_p1', 'omega_p1', 'mdilution_A')
    results = restore_parameters(fit_without_sampling(params, out_folder, fixed=fixed, **lc_data()))
    assert results.paramnames == [name for name in params if name not in fixed]
    everything_free = restore_parameters(fit_without_sampling(params, out_folder + 'free/', **lc_data()))
    assert results.loglike(free_vector(results, params)) == everything_free.loglike(free_vector(everything_free, params))


def test_loglike_refuses_eccentricities_above_the_limit(out_folder):
    params = dict(transit_parameters(), ecc_p1=0.5, omega_p1=40.)
    results = restore_parameters(fit_without_sampling(params, out_folder, fit_kwargs={'ecclim': 0.3}, **lc_data()))
    assert results.loglike(free_vector(results, params)) == REFUSED
    allowed = dict(params, ecc_p1=0.2)
    assert np.isfinite(results.loglike(free_vector(results, allowed)))
    assert results.loglike(free_vector(results, allowed)) > REFUSED


@pytest.mark.parametrize('kind,parameters,data', [('lc', transit_parameters, lc_data), ('rv', rv_parameters, rv_data)])
def test_loglike_refuses_periods_out_of_order(kind, parameters, data, out_folder):
    params = parameters(planets=(1, 2))
    results = restore_parameters(fit_without_sampling(params, out_folder, **data()))
    assert results.loglike(free_vector(results, params)) > REFUSED
    swapped = dict(params, P_p1=params['P_p2'], P_p2=params['P_p1'])
    assert results.loglike(free_vector(results, swapped)) == REFUSED


def test_loglike_is_minus_1e101_for_a_nan(out_folder):
    params = transit_parameters()
    results = restore_parameters(fit_without_sampling(params, out_folder, **lc_data()))
    assert results.loglike(free_vector(results, dict(params, mflux_A=np.nan))) == REFUSED


# One parameter per prior distribution. The third element is scipy's version of it:
PRIORS = {
    'P_p1': ('normal', [1.7, 0.1], stats.norm(1.7, 0.1)),
    't0_p1': ('uniform', [2458460., 2458461.], stats.uniform(2458460., 1.)),
    'a_p1': ('loguniform', [1., 100.], stats.loguniform(1., 100.)),
    'b_p1': ('beta', [2., 0.5], stats.beta(2., 0.5)),
    'p_p1': ('truncatednormal', [0.1, 0.05, 0., 1.], stats.truncnorm(-2., 18., loc=0.1, scale=0.05)),
    'ecc_p1': ('fixed', 0., None),
    'omega_p1': ('fixed', 90., None),
    'q1_A': ('uniform', [0., 1.], stats.uniform(0., 1.)),
    'q2_A': ('fixed', 0.35, None),
    'mdilution_A': ('fixed', 1., None),
    'mflux_A': ('normal', [0., 0.1], stats.norm(0., 0.1)),
    'sigma_w_A': ('jeffreys', [1., 1000.], stats.loguniform(1., 1000.)),
}
VALUES = {'P_p1': 1.7, 't0_p1': 2458460.5, 'a_p1': 6., 'b_p1': 0.2, 'p_p1': 0.1, 'q1_A': 0.3, 'mflux_A': 1e-5,
          'sigma_w_A': 50.}
FREE = [name for name, prior in PRIORS.items() if prior[0] != 'fixed']
CUBE = np.linspace(0.07, 0.93, len(FREE))


def prior_fit(out_folder, sampler):
    priors = {name: {'distribution': prior[0], 'hyperparameters': prior[1]} for name, prior in PRIORS.items()}
    values = dict(VALUES, ecc_p1=0., omega_p1=90., q2_A=0.35, mdilution_A=1.)
    return restore_parameters(fit_without_sampling(values, out_folder, priors=priors, sampler=sampler, **lc_data()))


def test_prior_transform_maps_the_unit_cube_through_each_prior(out_folder):
    results = prior_fit(out_folder, 'dynesty')
    assert results.paramnames == FREE
    expected = [PRIORS[name][2].ppf(u) for name, u in zip(FREE, CUBE)]
    np.testing.assert_allclose(results.prior_transform_r(np.copy(CUBE)), expected, rtol=1e-12)
    # The MultiNest version writes into the vector it's given:
    cube = np.copy(CUBE)
    assert results.prior_transform(cube) is None
    np.testing.assert_allclose(cube, expected, rtol=1e-12)


def test_logprior_and_logprob_for_mcmc(out_folder):
    results = prior_fit(out_folder, 'emcee')
    theta = free_vector(results, VALUES)
    expected = sum(PRIORS[name][2].logpdf(VALUES[name]) for name in FREE)
    assert results.logprior(theta) == pytest.approx(expected, rel=1e-12)
    assert results.logprob(theta) == pytest.approx(expected + results.loglike(theta), rel=1e-12)
    outside = free_vector(results, dict(VALUES, t0_p1=2458462.))
    assert results.logprior(outside) == -np.inf
    assert results.logprob(outside) == -np.inf


def test_nested_and_mcmc_fits_share_the_likelihood(out_folder):
    nested, mcmc = prior_fit(out_folder + 'nested/', 'dynesty'), prior_fit(out_folder + 'mcmc/', 'emcee')
    theta = free_vector(nested, VALUES)
    assert nested.loglike(theta) == mcmc.loglike(theta)


def test_joint_loglike_is_the_sum_of_the_datasets(out_folder):
    lc_params, rv_params = transit_parameters(('A',)), rv_parameters(('R',))
    joint_params = dict(rv_params, **lc_params)
    photometry = restore_parameters(fit_without_sampling(lc_params, out_folder + 'lc/', **lc_data(('A',))))
    velocities = restore_parameters(fit_without_sampling(dict(rv_params, P_p1=1.7, t0_p1=2458460.5),
                                                         out_folder + 'rv/', **rv_data(('R',))))
    joint = restore_parameters(fit_without_sampling(joint_params, out_folder + 'joint/',
                                                    **dict(lc_data(('A',)), **rv_data(('R',)))))
    total = (photometry.loglike(free_vector(photometry, joint_params)) +
             velocities.loglike(free_vector(velocities, joint_params)))
    assert joint.loglike(free_vector(joint, joint_params)) == pytest.approx(total, rel=1e-13)
