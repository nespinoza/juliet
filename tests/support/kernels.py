"""
Every GP kernel juliet offers: example parameters and an independent closed form of its covariance.

The closed forms state what the code does today, including where that differs from the documentation. See
tests/golden/CONVENTIONS.md before changing one.
"""
import numpy as np


# george adds this variance by default. Only global GPs keep the default, as per-instrument ones set it to the jitter.
GEORGE_DEFAULT_WHITE_NOISE = 1.25e-12


def _sho(tau, S0, Q, w0):
    eta = np.sqrt(1. - 1. / (4. * Q**2))
    x = w0 * tau
    return S0 * w0 * Q * np.exp(-x / (2. * Q)) * (np.cos(eta * x) + np.sin(eta * x) / (2. * eta * Q))


def _celerite_matern(tau, sigma, rho, eps):
    w = np.sqrt(3.) / rho
    return sigma**2 * np.exp(-w * tau) * (np.cos(eps * tau) + (w / eps) * np.sin(eps * tau))


def se(tau, p, kind, eps):
    amplitude = p['sigma'] * (1e-6 if kind == 'lc' else 1.)
    return amplitude**2 * np.exp(-0.5 * p['alpha0'] * tau**2)


def m32(tau, p, kind, eps):
    amplitude = p['sigma'] * (1e-6 if kind == 'lc' else 1.)
    r = np.sqrt(3. * p['malpha0']) * tau
    return amplitude**2 * (1. + r) * np.exp(-r)


def exp_sine_squared_se(tau, p, kind, eps):
    amplitude = p['sigma'] * (1e-6 if kind == 'lc' else 1.)
    # juliet passes log(Gamma), but george's gamma isn't a logarithm, so the coefficient is ln(Gamma). See CONVENTIONS.md.
    return amplitude**2 * np.exp(-0.5 * p['alpha'] * tau**2) * np.exp(-np.log(p['Gamma']) * np.sin(np.pi * tau / p['Prot'])**2)


def celerite_qp(tau, p, kind, eps):
    return (p['B'] / (2. + p['C'])) * np.exp(-tau / p['L']) * (np.cos(2. * np.pi * tau / p['Prot']) + (1. + p['C']))


def celerite_exp(tau, p, kind, eps):
    return p['sigma'] * np.exp(-p['timescale'] * tau)


def celerite_matern(tau, p, kind, eps):
    return _celerite_matern(tau, p['sigma'], p['rho'], eps)


def celerite_matern_exp(tau, p, kind, eps):
    return p['sigma'] * np.exp(-p['timescale'] * tau) * _celerite_matern(tau, 1., p['rho'], eps)


def celerite_sho(tau, p, kind, eps):
    return _sho(tau, p['S0'], p['Q'], p['omega0'])


def celerite_double_sho(tau, p, kind, eps):
    # omega2 is computed from Q1 here, as in juliet today. celerite2's RotationTerm, which juliet's comment says it
    # follows, uses Q2. See tests/golden/CONVENTIONS.md.
    Q1, Q2 = 0.5 + p['Q0'] + p['dQ'], 0.5 + p['Q0']
    w1 = 4. * np.pi * Q1 / (p['period'] * np.sqrt(4. * Q1**2 - 1.))
    w2 = 8. * np.pi * Q1 / (p['period'] * np.sqrt(4. * Q1**2 - 1.))
    S1 = p['sigma']**2 / ((1. + p['f']) * w1 * Q1)
    S2 = p['f'] * p['sigma']**2 / ((1. + p['f']) * w2 * Q2)
    return _sho(tau, S1, Q1, w1) + _sho(tau, S2, Q2, w2)


def celerite_matern_sho(tau, p, kind, eps):
    return _celerite_matern(tau, p['sigma'], p['rho'], eps) + _sho(tau, p['S0'], p['Q'], p['omega0'])


# Parameter values for each data type. Photometric ones are chosen for residuals of about 1e-3, in the units each
# kernel really uses: george kernels convert sigma from ppm, celerite kernels don't.
KERNELS = {
    'SEKernel': {'closed_form': se,
                 'lc': {'sigma': 300., 'alpha0': 2.}, 'rv': {'sigma': 3., 'alpha0': 0.05}},
    'M32Kernel': {'closed_form': m32,
                  'lc': {'sigma': 300., 'malpha0': 2.}, 'rv': {'sigma': 3., 'malpha0': 0.05}},
    'ExpSineSquaredSEKernel': {'closed_form': exp_sine_squared_se,
                               'lc': {'sigma': 300., 'alpha': 0.5, 'Gamma': 2., 'Prot': 1.3},
                               'rv': {'sigma': 3., 'alpha': 0.01, 'Gamma': 2., 'Prot': 25.}},
    'CeleriteQPKernel': {'closed_form': celerite_qp,
                         'lc': {'B': 1e-7, 'C': 0.5, 'L': 2., 'Prot': 1.3},
                         'rv': {'B': 9., 'C': 0.5, 'L': 30., 'Prot': 25.}},
    'CeleriteExpKernel': {'closed_form': celerite_exp,
                          'lc': {'sigma': 1e-7, 'timescale': 2.}, 'rv': {'sigma': 9., 'timescale': 0.1}},
    'CeleriteMaternKernel': {'closed_form': celerite_matern,
                             'lc': {'sigma': 3e-4, 'rho': 0.5}, 'rv': {'sigma': 3., 'rho': 5.}},
    'CeleriteMaternExpKernel': {'closed_form': celerite_matern_exp,
                                'lc': {'sigma': 1e-7, 'timescale': 0.5, 'rho': 0.5},
                                'rv': {'sigma': 9., 'timescale': 0.05, 'rho': 5.}},
    'CeleriteSHOKernel': {'closed_form': celerite_sho,
                          'lc': {'S0': 1e-7, 'Q': 2., 'omega0': 3.}, 'rv': {'S0': 3., 'Q': 2., 'omega0': 0.3}},
    'CeleriteDoubleSHOKernel': {'closed_form': celerite_double_sho,
                                'lc': {'sigma': 3e-4, 'Q0': 1., 'period': 1.3, 'f': 0.5, 'dQ': 0.5},
                                'rv': {'sigma': 3., 'Q0': 1., 'period': 25., 'f': 0.5, 'dQ': 0.5}},
    'CeleriteMaternSHOKernel': {'closed_form': celerite_matern_sho,
                                'lc': {'sigma': 3e-4, 'rho': 0.5, 'S0': 1e-7, 'Q': 2., 'omega0': 3.},
                                'rv': {'sigma': 3., 'rho': 5., 'S0': 3., 'Q': 2., 'omega0': 0.3}},
}


def gp_parameters(kernel, kind, suffix):
    return {'GP_{}_{}'.format(name, suffix): value for name, value in KERNELS[kernel][kind].items()}
