"""Small models shared by the contract tests."""
import copy
import os
import pickle

import numpy as np

import juliet

from support.build import build, make_priors, make_samples, instrument_parameters, rv_parameters, transit_parameters
from support.datasets import lc_data, rv_data, stack
from support.kernels import gp_parameters

INSTRUMENTS = ('X', 'Y')
KERNEL = 'CeleriteMaternKernel'
# get_quantiles needs more than a handful of samples for return_err:
NSAMPLES = 20


def gp_inputs(kind, mode, transit=True):
    """(params, load keyword arguments) for two instruments and a Matern GP."""
    if kind == 'lc':
        data = lc_data(INSTRUMENTS, n=50)
        params = transit_parameters(INSTRUMENTS)
    else:
        data = rv_data(INSTRUMENTS)
        params = rv_parameters(INSTRUMENTS)
    times = data['t_' + kind]
    if mode == 'global':
        data['GP_regressors_' + kind] = {kind: stack(times, INSTRUMENTS)}
        params.update(gp_parameters(KERNEL, kind, kind))
    else:
        data['GP_regressors_' + kind] = {name: times[name] for name in INSTRUMENTS}
        for name in INSTRUMENTS:
            params.update(gp_parameters(KERNEL, kind, name))
    return params, data


def gp_model(kind, mode, posterior=True):
    params, data = gp_inputs(kind, mode)
    _, model = build(params, kind, posterior=posterior, nsamples=NSAMPLES, **data)
    return model, params, data


def new_times(data, kind, name='X', n=37):
    t = data['t_' + kind][name]
    return np.linspace(t[0], t[-1], n)


RESULT_FILES = {'multinest': 'posteriors.pkl', 'dynesty': '_dynesty_NS_posteriors.pkl',
                'dynamic_dynesty': '_dynesty_DNS_posteriors.pkl'}


def write_results(out_folder, samples, sampler='dynesty', pl=0., pu=1., ta=2458460.):
    """Write a results file as juliet does, so that fit() reads it instead of sampling. samples: {name: array}."""
    os.makedirs(out_folder, exist_ok=True)
    posterior = {'unnamed': np.column_stack(list(samples.values())), 'loglike': np.zeros(len(next(iter(samples.values()))))}
    posterior.update(samples)
    with open(out_folder + RESULT_FILES.get(sampler, sampler + '_posteriors.pkl'), 'wb') as f:
        pickle.dump({'posterior_samples': posterior, 'lnZ': 0., 'lnZerr': 0., 'pl': pl, 'pu': pu, 'ta': ta}, f)


def fit_without_sampling(params, out_folder, fixed=(), priors=None, sampler='dynesty', nsamples=20, fit_kwargs=None,
                         **load_kwargs):
    """A juliet.fit object for params, made by reading a results file. No sampler runs."""
    priors = priors or make_priors(params, fixed)
    # (Compared exactly, as fit() does when it reads results.)
    free = [name for name in priors if priors[name]['distribution'] != 'fixed']
    samples = make_samples({name: params[name] for name in free}, nsamples=nsamples)
    fit_kwargs = fit_kwargs or {}
    write_results(out_folder, samples, sampler, **{k: fit_kwargs[k] for k in ('pl', 'pu', 'ta') if k in fit_kwargs})
    dataset = juliet.load(priors=priors, out_folder=out_folder, starting_point={name: params[name] for name in free},
                          **copy.deepcopy(load_kwargs))
    return dataset.fit(sampler=sampler, **fit_kwargs)


def free_vector(results, params):
    return np.array([params[name] for name in results.paramnames], dtype=float)


def restore_parameters(results):
    """
    fit() ends by replacing its dictionary of current parameter values with the results, which drops the fixed
    parameters, so loglike raises KeyError on the object fit() returns (see test_known_bugs.py). This puts them back.
    """
    priors = results.data.priors
    results.posteriors = {name: priors[name]['hyperparameters'] if priors[name]['distribution'] == 'fixed' else 0.
                          for name in results.model_parameters}
    return results
