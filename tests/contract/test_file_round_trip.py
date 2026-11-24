"""
load(..., out_folder=...) writes its inputs to files, and load(input_folder=...) reads them back.

The files keep ten decimals, so the reloaded dataset isn't identical: the log-likelihood moves by up to 1e-4 for
photometry (see tests/README.md). Arguments that aren't data (ld_laws, supersampling, matern_eps...) aren't written at
all, so they have to be given again when reloading.
"""
import copy

import numpy as np
import pytest

import juliet
from golden.configs import CONFIGS
from support.build import make_priors, transit_parameters
from support.datasets import lc_data

pytestmark = pytest.mark.contract

DATA_ARGUMENTS = {'t_lc', 'y_lc', 'yerr_lc', 't_rv', 'y_rv', 'yerr_rv', 'GP_regressors_lc', 'GP_regressors_rv',
                  'linear_regressors_lc', 'linear_regressors_rv'}
# Files hold ten decimals. Times are around 2.4e6, where double precision itself is 5e-10:
DATA_ROUNDING, TIME_ROUNDING = 5.1e-11, 1e-9
# Largest change measured over these configurations: 9e-5, and 3e-4 on a log-likelihood of -8e6.
LOGLIKE_ABS, LOGLIKE_REL = 5e-4, 1e-10


def round_trips(config):
    arguments = config.data()
    # Configurations that already read files don't write them, and non-linear functions are covered at the bottom:
    return not (config.requires or config.loglike_only or 'non_linear_functions' in arguments
                or 'lcfilename' in arguments or 'rvfilename' in arguments)


ROUND_TRIP = [pytest.param(config, id=config.id) for config in CONFIGS if round_trips(config)]


def save_and_reload(config, out_folder, again=True):
    """Returns (original, reloaded). With again=True, the arguments that aren't written to files are passed again."""
    arguments = config.data()
    original = juliet.load(priors=make_priors(config.params, config.fixed), out_folder=out_folder,
                           **copy.deepcopy(arguments))
    extra = {name: value for name, value in arguments.items() if name not in DATA_ARGUMENTS} if again else {}
    return original, juliet.load(input_folder=out_folder, **copy.deepcopy(extra))


def loglike(dataset, config):
    total = 0.
    for kind in config.modeltypes:
        model = juliet.model(dataset, modeltype=kind, log_like_calc=True, **config.model_kwargs)
        model.generate(dict(config.params))
        if not model.modelOK:
            return None
        total += model.get_log_likelihood(dict(config.params))
    return total


def as_columns(regressors):
    return {name: np.reshape(values, (len(values), -1)) for name, values in regressors.items()}


@pytest.mark.parametrize('config', ROUND_TRIP)
def test_data_come_back_to_ten_decimals(config, out_folder):
    original, reloaded = save_and_reload(config, out_folder)
    for kind in config.modeltypes:
        assert getattr(reloaded, 'inames_' + kind) == getattr(original, 'inames_' + kind)
        for attribute, atol in (('times_', TIME_ROUNDING), ('data_', DATA_ROUNDING), ('errors_', DATA_ROUNDING)):
            before, after = getattr(original, attribute + kind), getattr(reloaded, attribute + kind)
            assert list(after) == list(before)
            for instrument in before:
                np.testing.assert_allclose(after[instrument], before[instrument], rtol=0, atol=atol)
        for attribute in ('GP_{}_arguments', 'lm_{}_arguments'):
            before, after = getattr(original, attribute.format(kind)), getattr(reloaded, attribute.format(kind))
            # (With no regressors, the original has None and the reloaded dataset an empty dictionary.)
            before, after = as_columns(before or {}), as_columns(after or {})
            assert list(after) == list(before)
            for name in before:
                np.testing.assert_allclose(after[name], before[name], rtol=0, atol=TIME_ROUNDING)


def comparable(options):
    """Options without the GP objects, which are rebuilt on every load."""
    if isinstance(options, dict):
        return {key: comparable(value) for key, value in options.items() if key != 'noise_model'}
    return options


@pytest.mark.parametrize('config', ROUND_TRIP)
def test_options_come_back(config, out_folder):
    original, reloaded = save_and_reload(config, out_folder)
    for kind in config.modeltypes:
        before, after = getattr(original, kind + '_options'), getattr(reloaded, kind + '_options')
        np.testing.assert_equal(comparable(after), comparable(before))
    assert list(reloaded.priors) == list(original.priors)
    for name, prior in original.priors.items():
        assert reloaded.priors[name]['distribution'] == prior['distribution']
        np.testing.assert_allclose(reloaded.priors[name]['hyperparameters'], prior['hyperparameters'], rtol=1e-12)


@pytest.mark.parametrize('config', ROUND_TRIP)
def test_likelihood_changes_only_by_the_rounding(config, out_folder):
    original, reloaded = save_and_reload(config, out_folder)
    before, after = loglike(original, config), loglike(reloaded, config)
    if before is None:
        assert after is None
        return
    assert after == pytest.approx(before, rel=LOGLIKE_REL, abs=LOGLIKE_ABS)


def test_rv_likelihood_survives_the_round_trip_almost_exactly(out_folder):
    """Ten decimals are far below RV errors, unlike photometry at the ppm level."""
    config = next(config for config in CONFIGS if config.id == 'rv_one_planet')
    original, reloaded = save_and_reload(config, out_folder)
    assert loglike(reloaded, config) == pytest.approx(loglike(original, config), abs=1e-7)


def config_with(argument):
    return [pytest.param(config, id=config.id) for config in CONFIGS if round_trips(config)
            and argument in config.data()]


NOT_WRITTEN = 'load(out_folder=...) writes data, regressors and priors, but not this argument, so a reload falls back ' \
              'to its default with no warning.'


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=NOT_WRITTEN)
@pytest.mark.parametrize('config', config_with('ld_laws')[:1] + config_with('lc_n_supersamp')[:1]
                         + config_with('matern_eps')[:1])
def test_reloading_alone_gives_the_same_model(config, out_folder):
    original, reloaded = save_and_reload(config, out_folder, again=False)
    assert loglike(reloaded, config) == pytest.approx(loglike(original, config), rel=LOGLIKE_REL, abs=LOGLIKE_ABS)


def non_linear_inputs():
    data = lc_data()
    return dict(data, non_linear_functions={'A': {'function': lambda x, p: p['slope_A'] * x,
                                                  'regressor': data['t_lc']['A'] - data['t_lc']['A'][0]}})


def test_non_linear_functions_act_on_data_given_as_arrays():
    params = dict(transit_parameters(), slope_A=1e-3)
    with_function = juliet.load(priors=make_priors(params), **non_linear_inputs())
    without = juliet.load(priors=make_priors(params), **lc_data())
    config = type('Config', (), {'modeltypes': ('lc',), 'model_kwargs': {}, 'params': params})
    assert loglike(with_function, config) != loglike(without, config)


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason='When data come from files (input_folder or lcfilename), nlm_lc_boolean is set to False for '
                          'every instrument, so non_linear_functions is ignored with no warning: fit.py:1320.')
def test_non_linear_functions_act_on_data_read_from_files(out_folder):
    params = dict(transit_parameters(), slope_A=1e-3)
    arguments = non_linear_inputs()
    original = juliet.load(priors=make_priors(params), out_folder=out_folder, **arguments)
    reloaded = juliet.load(input_folder=out_folder, non_linear_functions=arguments['non_linear_functions'])
    config = type('Config', (), {'modeltypes': ('lc',), 'model_kwargs': {}, 'params': params})
    assert loglike(reloaded, config) == pytest.approx(loglike(original, config), rel=LOGLIKE_REL, abs=LOGLIKE_ABS)
