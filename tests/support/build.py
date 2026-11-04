"""Build juliet.load and juliet.model objects directly, without running a sampler."""
import copy

import numpy as np

import juliet


def make_priors(params, fixed=()):
    """Give every parameter a prior. juliet reads only the name and the 'fixed' flag when evaluating models."""
    priors = {}
    for name, value in params.items():
        if name in fixed:
            priors[name] = {'distribution': 'fixed', 'hyperparameters': value}
        else:
            priors[name] = {'distribution': 'uniform', 'hyperparameters': [0., 1e10]}
    return priors


def make_samples(params, fixed=(), nsamples=4):
    """A few samples close to params. Fixed parameters are left out, as they are in real posteriors."""
    return {name: np.array([value * (1. + 1e-5 * i) for i in range(nsamples)], dtype=float)
            for name, value in params.items() if name not in fixed}


def load(params, fixed=(), **load_kwargs):
    """juliet.load on deep copies, because load keeps user inputs by reference and evaluate writes to them."""
    return juliet.load(priors=make_priors(params, fixed), **copy.deepcopy(load_kwargs))


def build(params, modeltype, fixed=(), posterior=False, nsamples=4, model_kwargs=None, **load_kwargs):
    """Returns (dataset, model). With posterior=True the model holds samples and is ready for evaluate."""
    dataset = load(params, fixed=fixed, **load_kwargs)
    model = juliet.model(dataset, modeltype=modeltype, **(model_kwargs or {}))
    if posterior:
        model.set_posterior_samples(make_samples(params, fixed, nsamples))
    return dataset, model


def constant_samples(params, nsamples=3):
    """Identical samples, so that an evaluate call's result is exactly the model at params."""
    return {name: np.full(nsamples, value, dtype=float) for name, value in params.items()}


def transit_parameters(instruments=('A',), planets=(1,)):
    """A valid set of photometric parameters: (a, b, p), (q1, q2), circular orbits."""
    periods = {1: 1.7, 2: 3.9}
    values = {}
    for planet in planets:
        values.update({'P_p{}'.format(planet): periods[planet],
                       't0_p{}'.format(planet): 2458460.5 + 0.11 * (planet - 1),
                       'a_p{}'.format(planet): 6. + 5. * (planet - 1),
                       'b_p{}'.format(planet): 0.2,
                       'p_p{}'.format(planet): 0.1 - 0.03 * (planet - 1),
                       'ecc_p{}'.format(planet): 0.,
                       'omega_p{}'.format(planet): 90.})
    values.update(instrument_parameters(instruments))
    for name in instruments:
        values.update({'q1_' + name: 0.3, 'q2_' + name: 0.35})
    return values


def instrument_parameters(instruments=('A',)):
    values = {}
    for i, name in enumerate(instruments):
        values.update({'mdilution_' + name: 1. - 0.02 * i, 'mflux_' + name: 1e-5 * (i + 1),
                       'sigma_w_' + name: 50. + 10. * i})
    return values


def rv_parameters(instruments=('A',), planets=(1,)):
    periods = {1: 10., 2: 23.}
    values = {}
    for planet in planets:
        values.update({'P_p{}'.format(planet): periods[planet],
                       't0_p{}'.format(planet): 2458461. + 0.3 * (planet - 1),
                       'K_p{}'.format(planet): 5. - 2. * (planet - 1),
                       'ecc_p{}'.format(planet): 0.,
                       'omega_p{}'.format(planet): 90.})
    for i, name in enumerate(instruments):
        values.update({'mu_' + name: 10. - 15. * i, 'sigma_w_' + name: 0.5 + 0.2 * i})
    return values
