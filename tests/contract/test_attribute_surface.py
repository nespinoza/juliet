"""
What users reach into after a fit: the attributes of the load, fit and model objects, and the keys, types and shapes of
their dictionaries. Compared with tests/contract/attribute_surface.json.

Scripts in the wild read these directly (results.posteriors['posterior_samples'], dataset.times_lc, results.lc.model...),
so removing or renaming one, or changing a type or a shape, breaks them. Adding one is fine. If this fails on purpose:

    python tests/contract/test_attribute_surface.py

and commit the new snapshot with your change.
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
sys.path.insert(0, str(Path(__file__).parents[2]))

from contract.helpers import fit_without_sampling
from support.build import rv_parameters, transit_parameters
from support.datasets import lc_data, rv_data
from support.kernels import gp_parameters

SNAPSHOT = Path(__file__).parent / 'attribute_surface.json'


def joint_fit(out_folder):
    """Two photometric instruments (one with a GP, one with linear regressors) and two RV instruments with a trend."""
    data = dict(lc_data(('X', 'Y'), n=40), **rv_data(('R', 'S')))
    data['GP_regressors_lc'] = {'X': data['t_lc']['X']}
    data['linear_regressors_lc'] = {'Y': np.column_stack([np.sin(data['t_lc']['Y']), np.cos(data['t_lc']['Y'])])}
    params = dict(rv_parameters(('R', 'S')), **transit_parameters(('X', 'Y')))
    params.update(gp_parameters('CeleriteMaternKernel', 'lc', 'X'))
    params.update({'theta0_Y': 1e-4, 'theta1_Y': -1e-4, 'rv_slope': 0.01, 'rv_intercept': 0.1})
    fixed = ('ecc_p1', 'omega_p1', 'mdilution_X', 'mdilution_Y')
    return fit_without_sampling(params, out_folder, fixed=fixed, **data)


def describe(value):
    """Type, and shape for arrays. Numbers are all 'number', because numpy and Python scalars are interchangeable."""
    if isinstance(value, np.ndarray):
        kind = {'f': 'float', 'i': 'int', 'U': 'str', 'b': 'bool', 'O': 'object'}.get(value.dtype.kind, value.dtype.kind)
        return 'array of {} {}'.format(kind, list(value.shape))
    if isinstance(value, (bool, np.bool_)):
        return 'bool'
    if isinstance(value, (int, float, np.integer, np.floating)):
        return 'number'
    if isinstance(value, (str, np.str_)):
        return 'str'
    return type(value).__name__


def describe_dictionary(dictionary, depth=3):
    out = {}
    for key, value in dictionary.items():
        if isinstance(value, dict) and depth > 1:
            out[str(key)] = describe_dictionary(value, depth - 1)
        else:
            out[str(key)] = describe(value)
    return out


def attributes(instance):
    return {name: describe(value) for name, value in vars(instance).items() if not name.startswith('_')}


def current_surface():
    with tempfile.TemporaryDirectory() as folder:
        results = joint_fit(folder + '/')
        dataset = results.data
        surface = {
            'load attributes': attributes(dataset),
            'fit attributes': attributes(results),
            'model attributes (lc)': attributes(results.lc),
            'model attributes (rv)': attributes(results.rv),
            'results.posteriors': describe_dictionary(results.posteriors),
            'dataset.lc_options': describe_dictionary(dataset.lc_options),
            'dataset.rv_options': describe_dictionary(dataset.rv_options),
            'dataset.priors[name]': {'free': describe_dictionary(dataset.priors['P_p1']),
                                     'fixed': describe_dictionary(dataset.priors['ecc_p1'])},
            'results.lc.model': describe_dictionary(results.lc.model),
            'results.rv.model': describe_dictionary(results.rv.model),
            'results.lc.dictionary': describe_dictionary(results.lc.dictionary),
            'results.rv.dictionary': describe_dictionary(results.rv.dictionary),
        }
        for instrument in ('X', 'Y'):
            results.lc.evaluate(instrument, all_samples=True)
        for instrument in ('R', 'S'):
            results.rv.evaluate(instrument, all_samples=True)
        surface['results.lc.model after evaluate'] = describe_dictionary(results.lc.model)
        surface['results.rv.model after evaluate'] = describe_dictionary(results.rv.model)
    return surface


def differences(current, stored, path=''):
    out = []
    for key in sorted(set(current) | set(stored)):
        where = '{}/{}'.format(path, key)
        if key not in current:
            out.append('{}: gone (was {})'.format(where, stored[key] if not isinstance(stored[key], dict) else 'a dict'))
        elif key not in stored:
            out.append('{}: new'.format(where))
        elif isinstance(current[key], dict) and isinstance(stored[key], dict):
            out.extend(differences(current[key], stored[key], where))
        elif current[key] != stored[key]:
            out.append('{}: {} (was {})'.format(where, current[key], stored[key]))
    return out


@pytest.mark.contract
def test_attribute_surface_is_unchanged():
    with open(SNAPSHOT) as f:
        stored = json.load(f)
    assert differences(current_surface(), stored) == []


if __name__ == '__main__':
    with open(SNAPSHOT, 'w') as f:
        json.dump(current_surface(), f, indent=1, sort_keys=True)
        f.write('\n')
