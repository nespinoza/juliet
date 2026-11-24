"""
Public signatures, compared with tests/contract/signatures.json.

If this fails on purpose, for example because you added a keyword, run:

    python tests/contract/test_signatures.py

and commit the new snapshot with your change. Adding a keyword at the end is fine. Removing or renaming one, or changing
a default or an order, breaks users' scripts.
"""
import inspect
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[2]))

import juliet
from juliet import utils

SNAPSHOT = Path(__file__).parent / 'signatures.json'

CLASSES = {'load': juliet.load, 'fit': juliet.fit, 'model': juliet.model, 'gaussian_process': juliet.gaussian_process}
METHODS = {'load': ['fit', 'save'], 'fit': [],
           'model': ['evaluate_model', 'generate_lc_model', 'generate_rv_model', 'get_log_likelihood',
                     'set_posterior_samples'],
           'gaussian_process': ['compute_GP', 'set_parameter_vector']}


def current_signatures():
    out = {}
    for name, cls in CLASSES.items():
        out['juliet.{}'.format(name)] = str(inspect.signature(cls.__init__))
        for method in METHODS[name]:
            out['juliet.{}.{}'.format(name, method)] = str(inspect.signature(getattr(cls, method)))
    for name, function in inspect.getmembers(utils, inspect.isfunction):
        if function.__module__ == 'juliet.utils' and not name.startswith('_'):
            out['juliet.utils.{}'.format(name)] = str(inspect.signature(function))
    return out


@pytest.mark.contract
def test_public_signatures_are_unchanged():
    with open(SNAPSHOT) as f:
        assert current_signatures() == json.load(f)


if __name__ == '__main__':
    with open(SNAPSHOT, 'w') as f:
        json.dump(current_signatures(), f, indent=1, sort_keys=True)
