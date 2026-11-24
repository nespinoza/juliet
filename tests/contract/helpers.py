"""Small models shared by the contract tests."""
import numpy as np

from support.build import build, instrument_parameters, rv_parameters, transit_parameters
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
