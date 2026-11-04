"""A small transit fit with a known answer, shared by the sampler tests and by the flake-rate measurement."""
import batman
import numpy as np

import juliet

TRUTH = {'t0_p1': 2458460.5, 'p_p1': 0.1}
# In posterior standard deviations. One noise realization's offset is typically below 2, leaving room for sampling noise.
# See tests/README.md for the measured failure rate.
MAX_SIGMA = 5.

SETTINGS = {
    'dynesty': dict(n_live_points=100),
    'dynamic_dynesty': dict(n_live_points=100),
    'ultranest': dict(n_live_points=100),
    'multinest': dict(n_live_points=100),
    'emcee': dict(nwalkers=16, nsteps=500, nburnin=300),
}


def synthetic_transit():
    t = np.linspace(TRUTH['t0_p1'] - 0.15, TRUTH['t0_p1'] + 0.15, 120)
    params = batman.TransitParams()
    params.t0, params.per, params.rp, params.a = TRUTH['t0_p1'], 1.7, TRUTH['p_p1'], 6.
    params.inc, params.ecc, params.w = 90., 0., 90.
    params.limb_dark, params.u = 'quadratic', [0.3, 0.2]
    y = batman.TransitModel(params, t).light_curve(params) + np.random.RandomState(5).normal(0., 2e-4, len(t))
    return {'A': t}, {'A': y}, {'A': np.full(len(t), 2e-4)}


def fit_priors():
    names = ['P_p1', 't0_p1', 'a_p1', 'b_p1', 'p_p1', 'ecc_p1', 'omega_p1', 'u1_A', 'u2_A', 'mdilution_A', 'mflux_A', 'sigma_w_A']
    distributions = ['fixed', 'uniform', 'fixed', 'fixed', 'uniform', 'fixed', 'fixed', 'fixed', 'fixed', 'fixed', 'fixed', 'fixed']
    hyperparameters = [1.7, [TRUTH['t0_p1'] - 0.02, TRUTH['t0_p1'] + 0.02], 6., 0., [0.05, 0.15], 0., 90., 0.3, 0.2, 1., 0., 0.]
    return juliet.utils.generate_priors(names, distributions, hyperparameters)


def sigmas_from_truth(sampler, out_folder):
    """Fits, and returns how many posterior standard deviations each parameter is from its true value."""
    t, y, yerr = synthetic_transit()
    # (starting_point is only used by the MCMC samplers, which require it)
    start = {name: value + 1e-4 for name, value in TRUTH.items()}
    dataset = juliet.load(priors=fit_priors(), starting_point=start, t_lc=t, y_lc=y, yerr_lc=yerr, out_folder=out_folder)
    samples = dataset.fit(sampler=sampler, **SETTINGS[sampler]).posteriors['posterior_samples']
    return {name: abs(np.median(samples[name]) - value) / np.std(samples[name]) for name, value in TRUTH.items()}
