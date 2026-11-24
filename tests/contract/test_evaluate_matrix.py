"""
The evaluation matrix of section 2.2 in claude_refactoring.md: each way of calling evaluate, on each kind of GP model.

Every test states the correct behaviour. Where juliet doesn't do that today the test is a strict xfail, so that fixing
the bug makes the suite fail until the xfail is removed: nothing gets fixed or broken without being noticed.

For a global GP, evaluate(instrument) returns only that instrument's points, and evaluate() with no instrument returns
the full model at all timestamps. This was decided by the maintainer: see section 6 of the report.
"""
import numpy as np
import pytest

from contract.helpers import NSAMPLES, gp_model, new_times

pytestmark = pytest.mark.contract

MODES = [('rv', 'instrument'), ('rv', 'global'), ('lc', 'instrument'), ('lc', 'global')]
ID = {mode: '{}-{}'.format(*mode) for mode in MODES}


def n_points(data, kind, name='X'):
    return len(data['t_' + kind][name])


def n_total(data, kind):
    return sum(len(t) for t in data['t_' + kind].values())


def plain(model, params, data, kind):
    assert model.evaluate('X', all_samples=True).shape == (n_points(data, kind),)


def no_instrument(model, params, data, kind):
    assert model.evaluate(all_samples=True).shape == (n_total(data, kind),)


def at_times(model, params, data, kind):
    t = new_times(data, kind)
    assert model.evaluate('X', t=t, GPregressors=t, all_samples=True).shape == t.shape


def at_times_no_instrument(model, params, data, kind):
    t = new_times(data, kind)
    assert model.evaluate(t=t, GPregressors=t, all_samples=True).shape == t.shape


def at_times_regressor_dictionary(model, params, data, kind):
    # As the docstring of evaluate_model describes GPregressors for global models:
    t = new_times(data, kind)
    assert model.evaluate('X', t=t, GPregressors={kind: t}, all_samples=True).shape == t.shape


def scalars(model, params, data, kind):
    assert model.evaluate('X', parameter_values=params).shape == (n_points(data, kind),)


def scalars_at_times(model, params, data, kind):
    t = new_times(data, kind)
    assert model.evaluate('X', parameter_values=params, t=t, GPregressors=t).shape == t.shape


def plain_after_times(model, params, data, kind):
    t = new_times(data, kind)
    model.evaluate('X', t=t, GPregressors=t, all_samples=True)
    assert model.evaluate('X', all_samples=True).shape == (n_points(data, kind),)


def other_instrument_after_times(model, params, data, kind):
    t = new_times(data, kind)
    model.evaluate('X', t=t, GPregressors=t, all_samples=True)
    assert model.evaluate('Y', all_samples=True).shape == (n_points(data, kind, 'Y'),)


def return_err(model, params, data, kind):
    median, upper, lower = model.evaluate('X', all_samples=True, return_err=True)
    assert np.all(upper >= median) and np.all(lower <= median)


def return_samples(model, params, data, kind):
    samples, median = model.evaluate('X', all_samples=True, return_samples=True)
    assert samples.shape == (NSAMPLES, n_points(data, kind))


def return_components(model, params, data, kind):
    median, components = model.evaluate('X', all_samples=True, return_components=True)
    assert median.shape == (n_points(data, kind),)
    assert components


def at_times_return_components(model, params, data, kind):
    t = new_times(data, kind)
    median, components = model.evaluate('X', t=t, GPregressors=t, all_samples=True, return_components=True)
    assert median.shape == t.shape
    assert components


def repeated_calls_agree(model, params, data, kind):
    first = model.evaluate('X', all_samples=True)
    model.evaluate('Y', all_samples=True)
    np.testing.assert_array_equal(model.evaluate('X', all_samples=True), first)


def at_times_is_repeatable(model, params, data, kind):
    t = new_times(data, kind)
    first = model.evaluate('X', t=t, GPregressors=t, all_samples=True)
    np.testing.assert_array_equal(model.evaluate('X', t=t, GPregressors=t, all_samples=True), first)


def at_data_times_matches_plain(model, params, data, kind):
    t = data['t_' + kind]['X']
    at_data = model.evaluate('X', t=t, GPregressors=t, all_samples=True)
    fresh = gp_model(kind, 'instrument' if not model.global_model else 'global')[0]
    np.testing.assert_allclose(at_data, fresh.evaluate('X', all_samples=True), rtol=1e-8)


# The calls that make no sense for a per-instrument model:
GLOBAL_ONLY = {no_instrument, at_times_no_instrument, at_times_regressor_dictionary}

GLOBAL_STATE = "original_GPregressors is only assigned when t is given, but always used for global GPs: fit.py:2648"
NOT_RESTORED = 'noise_model.X is overwritten for a t= call and never restored for per-instrument GPs: fit.py:3038'
LC_GLOBAL_T = 'The t= path of global photometric GPs uses variables that only the RV path defines: fit.py:3083'
RV_COMPONENTS = 'RV components are sized for one instrument, but model["p1"] holds all of them: fit.py:3204'

KNOWN = {
    (plain, ('rv', 'global')): (AttributeError, GLOBAL_STATE),
    (plain, ('lc', 'global')): (AttributeError, GLOBAL_STATE),
    (no_instrument, ('rv', 'global')): (AttributeError, GLOBAL_STATE),
    (no_instrument, ('lc', 'global')): (KeyError, 'evaluate() with no instrument uses dictionary[None]: fit.py:2754'),
    (at_times, ('lc', 'global')): (KeyError, LC_GLOBAL_T),
    (at_times_no_instrument, ('rv', 'global')): (KeyError, 'instrument=None reaches parameter names: fit.py:2613'),
    (at_times_no_instrument, ('lc', 'global')): (KeyError, 'evaluate() with no instrument uses dictionary[None]: fit.py:2754'),
    (at_times_regressor_dictionary, ('rv', 'global')): (TypeError, 'The docstring asks for a dictionary, the code needs an array: fit.py:2650'),
    (at_times_regressor_dictionary, ('lc', 'global')): (KeyError, LC_GLOBAL_T),
    (scalars, ('rv', 'global')): (AttributeError, 'Issue #125. ' + GLOBAL_STATE),
    (scalars, ('lc', 'global')): (AttributeError, GLOBAL_STATE),
    (scalars_at_times, ('rv', 'instrument')): (AssertionError, 'Scalar parameter_values silently ignore t'),
    (scalars_at_times, ('rv', 'global')): (AttributeError, GLOBAL_STATE),
    (scalars_at_times, ('lc', 'instrument')): (AssertionError, 'Scalar parameter_values silently ignore t'),
    (scalars_at_times, ('lc', 'global')): (AttributeError, GLOBAL_STATE),
    (plain_after_times, ('rv', 'instrument')): (ValueError, NOT_RESTORED),
    (plain_after_times, ('rv', 'global')): (AssertionError, 'Returns the stack of all instruments, not this one (decided in section 6 of the report)'),
    (plain_after_times, ('lc', 'instrument')): (ValueError, NOT_RESTORED),
    (plain_after_times, ('lc', 'global')): (KeyError, LC_GLOBAL_T),
    (other_instrument_after_times, ('rv', 'global')): (AssertionError, 'Returns the stack of all instruments, not this one (decided in section 6 of the report)'),
    (other_instrument_after_times, ('lc', 'global')): (KeyError, LC_GLOBAL_T),
    (return_err, ('rv', 'global')): (AttributeError, GLOBAL_STATE),
    (return_err, ('lc', 'global')): (AttributeError, GLOBAL_STATE),
    (return_samples, ('rv', 'global')): (AttributeError, GLOBAL_STATE),
    (return_samples, ('lc', 'global')): (AttributeError, GLOBAL_STATE),
    (return_components, ('rv', 'instrument')): (ValueError, RV_COMPONENTS),
    (return_components, ('rv', 'global')): (AttributeError, GLOBAL_STATE),
    (return_components, ('lc', 'global')): (AttributeError, GLOBAL_STATE),
    (at_times_return_components, ('lc', 'global')): (KeyError, LC_GLOBAL_T),
    (repeated_calls_agree, ('rv', 'global')): (AttributeError, GLOBAL_STATE),
    (repeated_calls_agree, ('lc', 'global')): (AttributeError, GLOBAL_STATE),
    (at_times_is_repeatable, ('lc', 'global')): (KeyError, LC_GLOBAL_T),
    (at_data_times_matches_plain, ('rv', 'global')): (AttributeError, GLOBAL_STATE),
    (at_data_times_matches_plain, ('lc', 'global')): (KeyError, LC_GLOBAL_T),
}

CALLS = [plain, no_instrument, at_times, at_times_no_instrument, at_times_regressor_dictionary, scalars, scalars_at_times,
         plain_after_times, other_instrument_after_times, return_err, return_samples, return_components,
         at_times_return_components, repeated_calls_agree, at_times_is_repeatable, at_data_times_matches_plain]


def cases():
    out = []
    for call in CALLS:
        for mode in MODES:
            if call in GLOBAL_ONLY and mode[1] != 'global':
                continue
            marks = []
            if (call, mode) in KNOWN:
                exception, reason = KNOWN[(call, mode)]
                marks = [pytest.mark.xfail(strict=True, raises=exception, reason=reason)]
            out.append(pytest.param(call, mode, id='{}-{}'.format(call.__name__, ID[mode]), marks=marks))
    return out


@pytest.mark.parametrize('call,mode', cases())
def test_evaluate(call, mode):
    kind, gp_mode = mode
    model, params, data = gp_model(kind, gp_mode)
    call(model, params, data, kind)
