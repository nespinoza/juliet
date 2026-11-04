"""
Every kernel's log-likelihood and prediction against an independent dense-matrix calculation.

The covariance functions are in support/kernels.py and state what juliet does today, including where that differs from
the documentation (see tests/golden/CONVENTIONS.md). The means are analytic: no transit in photometry, and a circular
orbit in RVs, so that a bug in the model code can't hide a bug in the GP.
"""
import numpy as np
import pytest

from support.build import build, constant_samples, instrument_parameters, rv_parameters
from support.datasets import lc_data, rv_data, stack
from support.kernels import GEORGE_DEFAULT_WHITE_NOISE, KERNELS, gp_parameters

pytestmark = pytest.mark.unit

INSTRUMENTS = ('X', 'Y')
MATERN_EPS = 0.01
GEORGE = ('SEKernel', 'M32Kernel', 'ExpSineSquaredSEKernel')
# Measured agreement is up to 6e-6 in photometric log-likelihoods, because of celerite's precision at variances of 1e-8.
TOLERANCE = {'lc': {'loglike': 1e-5, 'prediction': 1e-9}, 'rv': {'loglike': 1e-7, 'prediction': 1e-7}}


def setup(kernel, kind, mode, george_hodlr=False):
    if kind == 'lc':
        data, params = lc_data(INSTRUMENTS, n=50), instrument_parameters(INSTRUMENTS)
    else:
        data, params = rv_data(INSTRUMENTS), rv_parameters(INSTRUMENTS)
    times = data['t_' + kind]
    if mode == 'global':
        data['GP_regressors_' + kind] = {kind: stack(times, INSTRUMENTS)}
        params.update(gp_parameters(kernel, kind, kind))
    else:
        data['GP_regressors_' + kind] = {name: times[name] for name in INSTRUMENTS}
        for name in INSTRUMENTS:
            params.update(gp_parameters(kernel, kind, name))
    data['george_hodlr'] = george_hodlr
    return data, params


def mean_function(kind, name, t, params):
    if kind == 'lc':
        return np.full(len(t), 1. / (1. + params['mdilution_' + name] * params['mflux_' + name]))
    return -params['K_p1'] * np.sin(2. * np.pi * (t - params['t0_p1']) / params['P_p1']) + params['mu_' + name]


def covariance(kernel, kind, t1, t2):
    tau = np.abs(t1[:, None] - t2[None, :])
    return KERNELS[kernel]['closed_form'](tau, KERNELS[kernel][kind], kind, MATERN_EPS)


def gaussian_loglike(residuals, K, variances):
    C = K + np.diag(variances)
    return -0.5 * (len(residuals) * np.log(2. * np.pi) + np.linalg.slogdet(C)[1] + residuals @ np.linalg.solve(C, residuals))


def arrays(kind, data, params, names, kernel=None):
    t = np.concatenate([data['t_' + kind][name] for name in names])
    y = np.concatenate([data['y_' + kind][name] - mean_function(kind, name, data['t_' + kind][name], params)
                        for name in names])
    scale = 1e-6 if kind == 'lc' else 1.
    variances = np.concatenate([data['yerr_' + kind][name]**2 + (params['sigma_w_' + name] * scale)**2 for name in names])
    if kernel in GEORGE and len(names) > 1:
        variances = variances + GEORGE_DEFAULT_WHITE_NOISE
    return t, y, variances


CASES = [(kernel, kind, mode) for kernel in KERNELS for kind in ('lc', 'rv') for mode in ('instrument', 'global')]


@pytest.mark.parametrize('kernel,kind,mode', CASES, ids=['-'.join(case) for case in CASES])
def test_log_likelihood(kernel, kind, mode):
    data, params = setup(kernel, kind, mode)
    _, model = build(params, kind, **data)
    model.generate(params)
    if mode == 'global':
        t, r, variances = arrays(kind, data, params, INSTRUMENTS, kernel)
        expected = gaussian_loglike(r, covariance(kernel, kind, t, t), variances)
    else:
        expected = 0.
        for name in INSTRUMENTS:
            t, r, variances = arrays(kind, data, params, (name,))
            expected += gaussian_loglike(r, covariance(kernel, kind, t, t), variances)
    assert model.get_log_likelihood(params) == pytest.approx(expected, abs=TOLERANCE[kind]['loglike'])


# Global photometric GPs can't be evaluated today: see tests/contract/test_evaluate_matrix.py.
PREDICTION_CASES = [case for case in CASES if not (case[1] == 'lc' and case[2] == 'global')]


@pytest.mark.parametrize('kernel,kind,mode', PREDICTION_CASES, ids=['-'.join(case) for case in PREDICTION_CASES])
def test_prediction(kernel, kind, mode):
    data, params = setup(kernel, kind, mode)
    _, model = build(params, kind, **data)
    model.set_posterior_samples(constant_samples(params))
    target = data['t_' + kind]['X']
    t_new = np.linspace(target[0], target[-1], 33)
    computed = model.evaluate('X', t=t_new, GPregressors=t_new, all_samples=True)

    t, r, variances = arrays(kind, data, params, INSTRUMENTS if mode == 'global' else ('X',), kernel)
    C = covariance(kernel, kind, t, t) + np.diag(variances)
    expected = mean_function(kind, 'X', t_new, params) + covariance(kernel, kind, t_new, t) @ np.linalg.solve(C, r)
    np.testing.assert_allclose(computed, expected, rtol=0., atol=TOLERANCE[kind]['prediction'])


@pytest.mark.parametrize('kernel', GEORGE)
@pytest.mark.parametrize('kind', ['lc', 'rv'])
def test_hodlr_solver_matches_exact_solver(kernel, kind):
    likelihoods = []
    for george_hodlr in (True, False):
        data, params = setup(kernel, kind, 'global', george_hodlr)
        _, model = build(params, kind, **data)
        model.generate(params)
        likelihoods.append(model.get_log_likelihood(params))
    assert likelihoods[0] == pytest.approx(likelihoods[1], abs=1e-5)


def test_every_kernel_in_juliet_is_covered():
    data, params = setup('CeleriteMaternKernel', 'rv', 'instrument')
    dataset, _ = build(params, 'rv', **data)
    assert set(dataset.rv_options['X']['noise_model'].all_kernel_variables) == set(KERNELS)
