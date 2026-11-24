"""
Spellings of distribution names in a prior dictionary.

Prior files have always accepted any capitalisation (readpriors lowercases the name), and the docs write 'Normal',
'Uniform' and 'Fixed'. Dictionaries are kept as given, and the code compares them in three different ways, so some
spellings work and some don't. The ones that don't are strict xfails.
"""
import numpy as np
import pytest
from scipy import stats

import juliet
from contract.helpers import fit_without_sampling, free_vector, restore_parameters
from support.build import transit_parameters
from support.datasets import lc_data

pytestmark = pytest.mark.contract

PARAMS = transit_parameters()
# name: (hyperparameters, scipy's version)
DISTRIBUTIONS = {
    'uniform': ([0., 1.], stats.uniform(0., 1.)),
    'normal': ([0.1, 0.01], stats.norm(0.1, 0.01)),
    'truncatednormal': ([0.1, 0.01, 0., 1.], stats.truncnorm(-10., 90., loc=0.1, scale=0.01)),
    'loguniform': ([0.01, 1.], stats.loguniform(0.01, 1.)),
    'jeffreys': ([0.01, 1.], stats.loguniform(0.01, 1.)),
    'beta': ([2., 0.5], stats.beta(2., 0.5)),
}
CAMEL = {'truncatednormal': 'TruncatedNormal', 'loguniform': 'LogUniform', 'modjeffreys': 'ModJeffreys'}
NOT_LOWERCASED = 'set_prior_transform and set_logpriors compare "jeffreys" and "loguniform" without lowercasing ' \
                 '(fit.py:1599, 1622), so the parameter gets no prior function: KeyError at the first sample.'


def spellings(name):
    return sorted({name, name.capitalize(), name.upper(), CAMEL.get(name, name)})


def cases():
    out = []
    for name in DISTRIBUTIONS:
        for spelling in spellings(name):
            broken = name in ('loguniform', 'jeffreys') and spelling != name
            marks = [pytest.mark.xfail(strict=True, raises=KeyError, reason=NOT_LOWERCASED)] if broken else []
            out.append(pytest.param(name, spelling, marks=marks, id=spelling))
    return out


def fit_with(distribution, hyperparameters, out_folder, sampler):
    priors = {name: {'distribution': 'fixed', 'hyperparameters': value} for name, value in PARAMS.items()}
    priors['mflux_A'] = {'distribution': 'normal', 'hyperparameters': [0., 0.1]}
    priors['p_p1'] = {'distribution': distribution, 'hyperparameters': hyperparameters}
    return restore_parameters(fit_without_sampling(PARAMS, out_folder, priors=priors, sampler=sampler, **lc_data()))


@pytest.mark.parametrize('name,spelling', cases())
def test_nested_samplers_accept_the_spelling(name, spelling, out_folder):
    hyperparameters, distribution = DISTRIBUTIONS[name]
    results = fit_with(spelling, hyperparameters, out_folder, 'dynesty')
    assert results.paramnames == ['p_p1', 'mflux_A']
    assert results.prior_transform_r(np.array([0.3, 0.5]))[0] == pytest.approx(distribution.ppf(0.3), rel=1e-12)
    assert np.isfinite(results.loglike(free_vector(results, PARAMS)))


@pytest.mark.parametrize('name,spelling', cases())
def test_mcmc_samplers_accept_the_spelling(name, spelling, out_folder):
    hyperparameters, distribution = DISTRIBUTIONS[name]
    results = fit_with(spelling, hyperparameters, out_folder, 'emcee')
    expected = distribution.logpdf(PARAMS['p_p1']) + stats.norm(0., 0.1).logpdf(PARAMS['mflux_A'])
    assert results.logprior(free_vector(results, PARAMS)) == pytest.approx(expected, rel=1e-12)


def test_lowercase_fixed_is_left_out_of_the_fit(out_folder):
    results = fit_with('fixed', 0.1, out_folder, 'dynesty')
    assert results.paramnames == ['mflux_A']
    assert results.data.nparams == 1


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason='fit compares with "fixed" exactly when it lists the free parameters (fit.py:1867) but '
                          'lowercases when it counts and transforms them, so "Fixed" is free in one place and fixed '
                          'in the other. A real fit then fails with IndexError.')
@pytest.mark.parametrize('spelling', ['Fixed', 'FIXED'])
def test_capitalised_fixed_is_left_out_of_the_fit(spelling, out_folder):
    results = fit_with(spelling, 0.1, out_folder, 'dynesty')
    assert results.paramnames == ['mflux_A']


@pytest.mark.parametrize('spelling', ['fixed', 'Fixed', 'FIXED', 'Normal', 'LogUniform', 'JEFFREYS', 'TruncatedNormal'])
def test_prior_files_accept_any_spelling(spelling, tmp_path):
    values = {'fixed': '0.1', 'normal': '0.1,0.01', 'loguniform': '0.01,1.', 'jeffreys': '0.01,1.',
              'truncatednormal': '0.1,0.01,0.,1.'}
    path = tmp_path / 'priors.dat'
    path.write_text('p_p1 {} {}\n'.format(spelling, values[spelling.lower()]))
    priors = juliet.utils.readpriors(str(path))[0]
    assert priors['p_p1']['distribution'] == spelling.lower()
