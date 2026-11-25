"""Fits with a pool of processes: the likelihood has to be pickled and sent to the workers. Needs --run-sampler."""
import numpy as np
import pytest

import juliet
from sampler.recovery import MAX_SIGMA, SETTINGS, TRUTH, fit_priors, synthetic_transit

pytestmark = pytest.mark.sampler


@pytest.mark.parametrize('sampler', ['dynesty', 'emcee'])
def test_fit_with_two_processes(sampler, out_folder):
    pytest.importorskip(sampler)
    t, y, yerr = synthetic_transit()
    start = {name: value + 1e-4 for name, value in TRUTH.items()}
    dataset = juliet.load(priors=fit_priors(), starting_point=start, t_lc=t, y_lc=y, yerr_lc=yerr,
                          out_folder=out_folder)
    samples = dataset.fit(sampler=sampler, nthreads=2, **SETTINGS[sampler]).posteriors['posterior_samples']
    for name, value in TRUTH.items():
        assert abs(np.median(samples[name]) - value) / np.std(samples[name]) < MAX_SIGMA
