"""
Runs one of the paper's full fits and optionally stores the result as the baseline for later comparisons.

These take hours, so they are only run on request:

    python tests/scientific/fit_tutorial.py k2-140 --out /path/to/folder --nthreads 8 --store

--store writes tests/scientific/baselines/<problem>.json, which test_tutorials.py compares later runs with.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))

import juliet
from scientific.targets import PROBLEMS
from support import versions

BASELINES = Path(__file__).parent / 'baselines'


def run_fit(problem, out_folder, sampler='dynesty', n_live_points=1000, nthreads=None):
    dataset = juliet.load(out_folder=str(out_folder) + '/', **PROBLEMS[problem])
    results = dataset.fit(sampler=sampler, n_live_points=n_live_points, nthreads=nthreads)
    return summarize(results.posteriors)


def summarize(posteriors):
    summary = {}
    for name, samples in posteriors['posterior_samples'].items():
        if name != 'unnamed':
            low, median, high = np.percentile(samples, [16., 50., 84.])
            summary[name] = [float(median), float(high - median), float(median - low)]
    return {'parameters': summary, 'lnZ': float(posteriors['lnZ']), 'lnZerr': float(posteriors['lnZerr'])}


def store(problem, summary):
    commit = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True,
                            cwd=Path(__file__).parent).stdout.strip()
    BASELINES.mkdir(exist_ok=True)
    with open(BASELINES / (problem + '.json'), 'w') as f:
        json.dump(dict(summary, commit=commit, versions=versions()), f, indent=1, sort_keys=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('problem', choices=sorted(PROBLEMS))
    parser.add_argument('--out', required=True, help='Output folder. An existing fit in it is reused, not repeated.')
    parser.add_argument('--sampler', default='dynesty')
    parser.add_argument('--n-live-points', type=int, default=1000)
    parser.add_argument('--nthreads', type=int, default=None)
    parser.add_argument('--store', action='store_true')
    arguments = parser.parse_args()
    summary = run_fit(arguments.problem, arguments.out, arguments.sampler, arguments.n_live_points, arguments.nthreads)
    if arguments.store:
        store(arguments.problem, summary)
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
