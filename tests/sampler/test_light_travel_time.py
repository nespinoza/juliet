"""
A joint transit and eclipse fit with the light-travel-time correction. Needs --run-sampler.

The data hold a transit and an eclipse of the same planet in one instrument and a second transit in another one, as
in the script this test replaces. The eclipse is injected late by the light travel time, using the circular-orbit
formula and not juliet's own code.
"""
import astropy.constants as const
import batman
import numpy as np
import pytest

import juliet
from sampler.recovery import MAX_SIGMA

pytestmark = pytest.mark.sampler

PERIOD, T0, A, RS, FP, NOISE = 1., 0., 3.6, 1., 0.01, 100e-6
TRUTH = {'fp_p1': FP, 'p_p1_inst': 0.1, 'p_p1_inst2': 0.2, 'a_p1': A}


def planet(rp):
    params = batman.TransitParams()
    params.t0, params.per, params.a, params.rp, params.inc, params.ecc, params.w = T0, PERIOD, A, rp, 90., 0., 90.
    params.limb_dark, params.u = 'quadratic', [0.2, 0.3]
    params.fp, params.t_secondary = FP, T0 + PERIOD / 2.
    return params


def synthetic_data(n=150):
    rng = np.random.RandomState(42)
    window = np.linspace(-0.1, 0.1, n)
    t_transit, t_eclipse, t_second = T0 + window, T0 + PERIOD / 2. + window, T0 + PERIOD + window

    transit = batman.TransitModel(planet(0.1), t_transit).light_curve(planet(0.1))
    late = A * RS * const.R_sun.value / const.c.value / 86400. * (1. - np.cos(2. * np.pi * (t_eclipse - T0) / PERIOD))
    eclipse = batman.TransitModel(planet(0.1), t_eclipse - late, transittype='secondary').light_curve(planet(0.1)) - FP
    second = batman.TransitModel(planet(0.2), t_second).light_curve(planet(0.2))

    times = {'inst': np.append(t_transit, t_eclipse), 'inst2': t_second}
    fluxes = {'inst': np.append(transit, eclipse), 'inst2': second}
    fluxes = {name: flux + rng.normal(0., NOISE, len(flux)) for name, flux in fluxes.items()}
    # Error bars are underestimated on purpose, so that the jitter terms have something to fit:
    errors = {name: np.full(len(flux), 0.1 * NOISE) for name, flux in fluxes.items()}
    return times, fluxes, errors


def fit_priors():
    names = ['P_p1', 't0_p1', 'b_p1', 'ecc_p1', 'omega_p1', 'p_p1_inst', 'p_p1_inst2', 'q1_inst_inst2', 'q2_inst_inst2',
             'a_p1', 'mdilution_inst_inst2', 'mflux_inst', 'sigma_w_inst', 'fp_p1', 'mflux_inst2', 'sigma_w_inst2']
    distributions = ['fixed', 'fixed', 'fixed', 'fixed', 'fixed', 'uniform', 'uniform', 'uniform', 'uniform',
                     'normal', 'fixed', 'normal', 'loguniform', 'uniform', 'normal', 'loguniform']
    hyperparameters = [PERIOD, T0, 0., 0., 90., [0., 0.3], [0., 0.3], [0., 1.], [0., 1.],
                       [A, 0.1], 1., [0., 0.1], [0.1, 1000.], [0., 0.1], [0., 0.1], [0.1, 1000.]]
    return juliet.utils.generate_priors(names, distributions, hyperparameters)


def test_transit_and_eclipse_fit_with_light_travel_delay(out_folder):
    pytest.importorskip('dynesty')
    times, fluxes, errors = synthetic_data()
    dataset = juliet.load(priors=fit_priors(), t_lc=times, y_lc=fluxes, yerr_lc=errors, out_folder=out_folder)
    results = dataset.fit(sampler='dynesty', n_live_points=100, light_travel_delay=True, stellar_radius=RS)

    # fp_p1 names no instrument, so both get the transit and eclipse model, although inst2 only saw a transit:
    assert results.lc.dictionary['inst']['TranEclFit'] and results.lc.dictionary['inst2']['TranEclFit']
    samples = results.posteriors['posterior_samples']
    sigmas = {name: abs(np.median(samples[name]) - value) / np.std(samples[name]) for name, value in TRUTH.items()}
    assert all(value < MAX_SIGMA for value in sigmas.values()), sigmas

    # The fitted model follows the data in both instruments, to within the noise:
    for name in ('inst', 'inst2'):
        model = results.lc.evaluate(name, all_samples=True)
        assert model.shape == times[name].shape
        assert np.std(fluxes[name] - model) < 1.5 * NOISE
