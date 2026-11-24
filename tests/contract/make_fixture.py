"""
Recreates tests/contract/fixtures/transit_fit: a real, small dynesty fit, as written by the code at baseline-2.2.11.

This is here to document how the fixture was made. It should normally never be run again, because the fixture's whole
point is to check that juliet can still read outputs written by older versions.

Usage: python tests/contract/make_fixture.py
"""
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))

import juliet
from support.datasets import lc_data

FIXTURE = Path(__file__).parent / 'fixtures' / 'transit_fit'
KEEP_SAMPLES = 200


def main():
    shutil.rmtree(FIXTURE, ignore_errors=True)
    data = lc_data(n=40)
    params = ['P_p1', 't0_p1', 'a_p1', 'b_p1', 'p_p1', 'ecc_p1', 'omega_p1', 'q1_A', 'q2_A', 'mdilution_A', 'mflux_A',
              'sigma_w_A']
    distributions = ['fixed', 'normal', 'fixed', 'fixed', 'uniform', 'fixed', 'fixed', 'fixed', 'fixed', 'fixed',
                     'normal', 'fixed']
    hyperparameters = [1.7, [2458460.5, 0.01], 6., 0.2, [0., 0.2], 0., 90., 0.3, 0.35, 1., [0., 1e-3], 50.]
    priors = juliet.utils.generate_priors(params, distributions, hyperparameters)
    dataset = juliet.load(priors=priors, out_folder=str(FIXTURE) + '/', **data)
    dataset.fit(sampler='dynesty', n_live_points=50)

    # Thin the samples, to keep a small file in the repository. Everything else stays as juliet wrote it:
    path = FIXTURE / '_dynesty_NS_posteriors.pkl'
    with open(path, 'rb') as f:
        out = pickle.load(f)
    index = np.linspace(0, len(out['posterior_samples']['unnamed']) - 1, KEEP_SAMPLES).astype(int)
    out['posterior_samples'] = {name: values[index] for name, values in out['posterior_samples'].items()}
    out.pop('dynesty_output', None)
    with open(path, 'wb') as f:
        pickle.dump(out, f)
    (FIXTURE / 'posteriors.dat').unlink(missing_ok=True)
    juliet.load(input_folder=str(FIXTURE) + '/').fit(sampler='dynesty')


if __name__ == '__main__':
    main()
