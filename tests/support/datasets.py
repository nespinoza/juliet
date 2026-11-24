"""
Small deterministic synthetic datasets.

Noise comes from np.random.RandomState because, unlike Generator, numpy guarantees that its stream never changes
between versions (NEP 19). The fluxes and velocities are plain analytic functions, so a change in batman or radvel
can't alter the data itself: only juliet's model of it.
"""
import numpy as np

LC_T0 = 2458460.5
RV_T0 = 2458461.0
INSTRUMENTS = ('A', 'B', 'C')


def _noise(seed, n, scale):
    return np.random.RandomState(seed).normal(0., scale, n)


def lc_times(index, n=80):
    """Sorted per instrument. Different instruments interleave, so their stack is unsorted."""
    return np.linspace(LC_T0 - 0.2 + 0.013 * index, LC_T0 + 3.6 + 0.017 * index, n)


def lc_data(instruments=('A',), n=80, seed=11):
    t, y, yerr = {}, {}, {}
    for i, name in enumerate(instruments):
        t[name] = lc_times(i, n)
        phase = ((t[name] - LC_T0 + 0.85) % 1.7) - 0.85
        y[name] = 1. - 0.01 * np.exp(-0.5 * (phase / 0.03)**2) + _noise(seed + i, n, 1e-4 * (1 + i))
        yerr[name] = np.full(n, 1e-4 * (1 + i))
    return {'t_lc': t, 'y_lc': y, 'yerr_lc': yerr}


def rv_times(index, n=30):
    return np.linspace(RV_T0 - 1. + 0.37 * index, RV_T0 + 59. + 0.53 * index, n - 5 * index)


def rv_data(instruments=('A',), seed=23):
    t, y, yerr = {}, {}, {}
    for i, name in enumerate(instruments):
        t[name] = rv_times(i)
        n = len(t[name])
        y[name] = (-5. * np.sin(2. * np.pi * (t[name] - RV_T0) / 10.) + 3. * np.sin(2. * np.pi * t[name] / 25.)
                   + 10. - 15. * i + _noise(seed + i, n, 1. + 0.5 * i))
        yerr[name] = np.full(n, 1. + 0.5 * i)
    return {'t_rv': t, 'y_rv': y, 'yerr_rv': yerr}


def stack(times, instruments):
    """The stacked regressor juliet expects for a global GP: one column, in instrument order."""
    return np.concatenate([times[name] for name in instruments])[:, None]
