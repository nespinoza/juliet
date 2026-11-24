"""juliet must leave user inputs and its own model alone, whether a call succeeds or fails."""
import copy

import numpy as np
import pytest

from contract.helpers import gp_inputs, gp_model, new_times
from support.build import make_priors, make_samples

import juliet

pytestmark = pytest.mark.contract

ALIASED = 'load keeps user dictionaries by reference, and a failed global evaluate leaves them changed'


def assert_same(a, b, path=''):
    assert type(a) is type(b), path
    if isinstance(a, dict):
        assert a.keys() == b.keys(), path
        for key in a:
            assert_same(a[key], b[key], '{}/{}'.format(path, key))
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b, err_msg=path)
    else:
        assert a == b, path


def model_state(model):
    key = 'global_model' if model.global_model else None
    noise = {}
    if key:
        noise['global'] = np.copy(model.dictionary[key]['noise_model'].X)
    else:
        for name in model.inames:
            noise[name] = np.copy(model.dictionary[name]['noise_model'].X)
    return copy.deepcopy({'inames': list(model.inames), 'times': model.times, 't': model.t,
                          'instrument_indexes': model.instrument_indexes, 'X': noise})


def call_with_times(model, data, kind):
    t = new_times(data, kind)
    model.evaluate('X', t=t, GPregressors=t, all_samples=True)


def call_failing(model, data, kind):
    # Times without the regressors they need: this raises for every kind of model here.
    with pytest.raises(Exception):
        model.evaluate('X', t=new_times(data, kind), all_samples=True)


MODES = [('rv', 'instrument'), ('rv', 'global'), ('lc', 'instrument'), ('lc', 'global')]


SORTED_IN_PLACE = 'For a global celerite GP, sort_GP sorts the arrays in the user dictionaries in place'


@pytest.mark.parametrize('kind,mode', [
    ('rv', 'instrument'),
    pytest.param('rv', 'global', marks=pytest.mark.xfail(strict=True, raises=AssertionError, reason=SORTED_IN_PLACE)),
    ('lc', 'instrument'),
    pytest.param('lc', 'global', marks=pytest.mark.xfail(strict=True, raises=AssertionError, reason=SORTED_IN_PLACE)),
])
def test_load_leaves_inputs_unmodified(kind, mode):
    params, data = gp_inputs(kind, mode)
    original = copy.deepcopy(data)
    juliet.load(priors=make_priors(params), **data)
    assert_same(data, original)


def loaded_model(kind, mode):
    """The dictionaries given to load (and possibly changed by it), a copy of them, and a model ready to evaluate."""
    params, data = gp_inputs(kind, mode)
    dataset = juliet.load(priors=make_priors(params), **data)
    model = juliet.model(dataset, modeltype=kind)
    model.set_posterior_samples(make_samples(params))
    return data, copy.deepcopy(data), model


@pytest.mark.parametrize('kind,mode', [
    ('rv', 'instrument'),
    ('lc', 'instrument'),
    ('rv', 'global'),
])
def test_evaluate_at_times_leaves_inputs_unmodified(kind, mode):
    data, original, model = loaded_model(kind, mode)
    call_with_times(model, data, kind)
    assert_same(data, original)


@pytest.mark.parametrize('kind,mode', [
    ('rv', 'instrument'),
    ('lc', 'instrument'),
    pytest.param('rv', 'global', marks=pytest.mark.xfail(strict=True, raises=AssertionError, reason=ALIASED)),
    ('lc', 'global'),
])
def test_failed_evaluate_leaves_inputs_unmodified(kind, mode):
    data, original, model = loaded_model(kind, mode)
    if mode == 'global':
        # The no-instrument call fails part of the way through, after it has written to the times dictionary:
        t = new_times(data, kind)
        with pytest.raises(Exception):
            model.evaluate(t=t, GPregressors=t, all_samples=True)
    else:
        call_failing(model, data, kind)
    assert_same(data, original)


NOT_RESTORED = 'evaluate with t= changes the model and restores only part of it'
FAILED_STATE = 'A failed evaluate leaves the model changed, e.g. inames narrowed to one instrument'


@pytest.mark.parametrize('kind,mode', [
    pytest.param('rv', 'instrument', marks=pytest.mark.xfail(strict=True, raises=AssertionError, reason=NOT_RESTORED)),
    ('rv', 'global'),
    pytest.param('lc', 'instrument', marks=pytest.mark.xfail(strict=True, raises=AssertionError, reason=NOT_RESTORED)),
])
def test_evaluate_at_times_restores_the_model(kind, mode):
    model, _, data = gp_model(kind, mode)
    before = model_state(model)
    call_with_times(model, data, kind)
    assert_same(model_state(model), before)


@pytest.mark.parametrize('kind,mode', [
    pytest.param('rv', 'instrument', marks=pytest.mark.xfail(strict=True, raises=AssertionError, reason=FAILED_STATE)),
    pytest.param('rv', 'global', marks=pytest.mark.xfail(strict=True, raises=AssertionError, reason=FAILED_STATE)),
    pytest.param('lc', 'instrument', marks=pytest.mark.xfail(strict=True, raises=AssertionError, reason=FAILED_STATE)),
    pytest.param('lc', 'global', marks=pytest.mark.xfail(strict=True, raises=AssertionError, reason=FAILED_STATE)),
])
def test_failed_evaluate_restores_the_model(kind, mode):
    model, _, data = gp_model(kind, mode)
    before = model_state(model)
    call_failing(model, data, kind)
    assert_same(model_state(model), before)


@pytest.mark.parametrize('kind,mode', [('rv', 'instrument'), ('lc', 'instrument')])
def test_plain_evaluate_restores_the_model(kind, mode):
    model, _, data = gp_model(kind, mode)
    before = model_state(model)
    model.evaluate('X', all_samples=True)
    assert_same(model_state(model), before)


@pytest.mark.parametrize('kind,mode', MODES)
def test_likelihood_does_not_depend_on_evaluation_history(kind, mode):
    model, params, data = gp_model(kind, mode)
    model.generate(params)
    before = model.get_log_likelihood(params)
    try:
        call_with_times(model, data, kind)
    except Exception:
        pass
    fresh, _, _ = gp_model(kind, mode)
    fresh.generate(params)
    assert fresh.get_log_likelihood(params) == before
