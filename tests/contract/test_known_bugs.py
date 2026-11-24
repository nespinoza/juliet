"""
Verified defects outside the evaluation matrix, each stated as the correct behaviour and marked as a strict xfail.

To fix one: fix it, watch its test turn into a failure (XPASS), and remove the marker. See tests/README.md.
Sections refer to claude_refactoring.md.
"""
import numpy as np
import pytest

import dynesty

from golden.configs import additive_nonlinear
from support.build import build, instrument_parameters, rv_parameters, transit_parameters
from support.datasets import lc_data, lc_times, rv_data, stack
from support.kernels import KERNELS, gp_parameters

pytestmark = pytest.mark.contract


def xfail(exception, reason):
    return pytest.mark.xfail(strict=True, raises=exception, reason=reason)


# ---------------------------------------------------------------- Silently wrong results (section 2.3)
def ttv_model(**dts):
    params = transit_parameters(planets=(1, 2))
    params.update(dts)
    _, model = build(params, 'lc', **lc_data())
    model.generate(params)
    return model


@xfail(AssertionError, 'Issue #135. All planets use the last planet\'s TTV shifts, because dummy_time is overwritten '
                       'in one loop and used in another: fit.py:3649. Fixing this is a science change.')
def test_ttv_shift_of_a_planet_moves_that_planet():
    base = {'dt_p1_A_0': 0., 'dt_p1_A_1': 0., 'dt_p2_A_0': 0.}
    moved = ttv_model(**dict(base, dt_p1_A_0=0.02)).model['A']['p1']
    assert np.max(np.abs(moved - ttv_model(**base).model['A']['p1'])) > 1e-4


@xfail(AssertionError, 'Issue #135: same cause as test_ttv_shift_of_a_planet_moves_that_planet.')
def test_ttv_shift_of_a_planet_leaves_other_planets_alone():
    base = {'dt_p1_A_0': 0., 'dt_p1_A_1': 0., 'dt_p2_A_0': 0.}
    moved = ttv_model(**dict(base, dt_p2_A_0=0.02)).model['A']['p1']
    np.testing.assert_array_equal(moved, ttv_model(**base).model['A']['p1'])


@xfail(AssertionError, 'non_linear_functions is ignored for RVs, with no warning.')
def test_rv_non_linear_function_changes_the_model():
    params = rv_parameters()
    data = rv_data()
    _, without = build(params, 'rv', **data)
    function = {'function': lambda X, p: 1000. * np.ones(len(X)), 'regressor': data['t_rv']['A']}
    _, using = build(params, 'rv', non_linear_functions={'A': function}, **data)
    without.generate(params)
    using.generate(params)
    assert using.get_log_likelihood(params) != without.get_log_likelihood(params)


@xfail(AssertionError, 'phaseoffset only acts together with a transit (TranEclFit). For an eclipse-only model, it has '
                       'no effect and raises no warning: fit.py:3974.')
def test_phase_offset_changes_an_eclipse_only_model():
    base = {k: v for k, v in transit_parameters().items() if k not in ('q1_A', 'q2_A')}
    base.update(fp_p1=5e-4, t_secondary_p1=2458460.5 + 0.85)
    results = []
    for offset in (0., 60.):
        params = dict(base, phaseoffset_p1=offset)
        _, model = build(params, 'lc', **lc_data())
        model.generate(params)
        results.append(model.model['A']['M'])
    assert np.max(np.abs(results[0] - results[1])) > 1e-6


@xfail(AssertionError, 'CeleriteDoubleSHOKernel says it follows celerite2\'s RotationTerm, where '
                       'omega2 = 8 pi Q2 / (P sqrt(4 Q2^2 - 1)). juliet uses Q1 in that expression: fit.py:5452. '
                       'Fixing this is a science change.')
def test_double_sho_matches_celerite2_rotation_term():
    p = KERNELS['CeleriteDoubleSHOKernel']['rv']
    params = rv_parameters()
    for name, value in p.items():
        params['GP_{}_A'.format(name)] = value
    data = rv_data()
    data['GP_regressors_rv'] = {'A': data['t_rv']['A']}
    dataset, model = build(params, 'rv', **data)
    model.generate(params)
    noise = dataset.rv_options['A']['noise_model']
    noise.set_parameter_vector(params)
    Q2 = 0.5 + p['Q0']
    expected_omega2 = 8. * np.pi * Q2 / (p['period'] * np.sqrt(4. * Q2**2 - 1.))
    assert noise.parameter_vector[5] == pytest.approx(np.log(expected_omega2))


@xfail(AssertionError, 'george\'s ExpSine2Kernel gamma is linear, but juliet sets it to log(GP_Gamma), so the '
                       'coefficient in the kernel is ln(Gamma): fit.py:5379. A Gamma below 1 gives a negative one. '
                       'Fixing this is a science change, and would invalidate existing priors.')
def test_exp_sine_squared_gamma_is_the_documented_coefficient():
    params = rv_parameters()
    for name, value in KERNELS['ExpSineSquaredSEKernel']['rv'].items():
        params['GP_{}_A'.format(name)] = value
    data = rv_data()
    data['GP_regressors_rv'] = {'A': data['t_rv']['A']}
    dataset, model = build(params, 'rv', **data)
    noise = dataset.rv_options['A']['noise_model']
    noise.set_parameter_vector(params)
    names = noise.GP.get_parameter_names()
    assert noise.GP.get_parameter_vector()[names.index('kernel:k2:gamma')] == pytest.approx(params['GP_Gamma_A'])


# ---------------------------------------------------------------- Crashes in documented features (section 2.3)
def test_rv_linear_model_at_data_times_works():
    # Issue #128 was fixed for this case only. The next test is the part that's left.
    params = dict(rv_parameters(), theta0_A=1.)
    data = rv_data()
    data['linear_regressors_rv'] = {'A': np.ones((len(data['t_rv']['A']), 1))}
    _, model = build(params, 'rv', posterior=True, **data)
    assert model.evaluate('A', all_samples=True).shape == data['t_rv']['A'].shape


@xfail(ValueError, 'The RV t= path does not update ndatapoints_per_instrument: fit.py:2623.')
def test_rv_linear_model_at_new_times():
    params = dict(rv_parameters(), theta0_A=1.)
    data = rv_data()
    data['linear_regressors_rv'] = {'A': np.ones((len(data['t_rv']['A']), 1))}
    _, model = build(params, 'rv', posterior=True, **data)
    t = np.linspace(data['t_rv']['A'][0], data['t_rv']['A'][-1], 37)
    assert model.evaluate('A', t=t, LMregressors=np.ones((37, 1)), all_samples=True).shape == t.shape


LTT = {'light_travel_delay': True, 'stellar_radius': 1.2}


@xfail(ValueError, 'Light-travel correction uses self.times, not the times given: fit.py:4104. Related to issue #132.')
def test_light_travel_delay_at_new_times():
    params = {k: v for k, v in transit_parameters().items()}
    params['fp_p1'] = 5e-4
    data = lc_data()
    _, model = build(params, 'lc', posterior=True, model_kwargs=LTT, **data)
    t = np.linspace(data['t_lc']['A'][0], data['t_lc']['A'][-1], 37)
    assert model.evaluate('A', t=t, all_samples=True).shape == t.shape


@xfail(ValueError, 'TTV time shifts start from self.times, not the times given: fit.py:4394. Related to issue #132.')
def test_ttvs_at_new_times():
    params = dict(transit_parameters(), dt_p1_A_0=0.002)
    data = lc_data()
    _, model = build(params, 'lc', posterior=True, **data)
    t = np.linspace(data['t_lc']['A'][0], data['t_lc']['A'][-1], 37)
    assert model.evaluate('A', t=t, all_samples=True).shape == t.shape


@xfail(ValueError, 'The non-linear function\'s regressor can\'t be replaced at evaluation: fit.py:4560.')
def test_non_linear_function_at_new_times():
    params = dict(transit_parameters(), nl_amp_A=2e-3, nl_tau_A=0.5)
    data = dict(lc_data(), non_linear_functions={'A': {'function': additive_nonlinear, 'regressor': lc_times(0)}})
    _, model = build(params, 'lc', posterior=True, **data)
    t = np.linspace(data['t_lc']['A'][0], data['t_lc']['A'][-1], 37)
    assert model.evaluate('A', t=t, all_samples=True).shape == t.shape


@xfail(KeyError, 'return_components assumes every instrument has a transit: fit.py:3194.')
def test_components_for_an_instrument_without_a_transit():
    params = {k: v for k, v in transit_parameters(('A', 'B')).items() if k not in ('q1_B', 'q2_B')}
    data = lc_data(('A', 'B'))
    _, model = build(params, 'lc', posterior=True, **data)
    median, components = model.evaluate('B', all_samples=True, return_components=True)
    assert median.shape == data['t_lc']['B'].shape


@xfail(TypeError, 'A global model with no GP has no return in get_log_likelihood, so it returns None: fit.py:4592.')
def test_global_model_without_gp_has_a_likelihood():
    params = dict(transit_parameters(('A', 'B')), theta0_lc=0.)
    _, model = build(params, 'lc', **lc_data(('A', 'B')))
    assert model.global_model
    model.generate(params)
    assert np.isfinite(model.get_log_likelihood(params))


@pytest.mark.parametrize('kind', ['lc'])
def test_shared_jitter_without_gp_works(kind):
    params = {k: v for k, v in transit_parameters(('A', 'B')).items() if k not in ('sigma_w_A', 'sigma_w_B')}
    params['sigma_w_A_B'] = 60.
    _, model = build(params, 'lc', **lc_data(('A', 'B')))
    model.generate(params)
    assert np.isfinite(model.get_log_likelihood(params))


def shared_jitter_gp(mode):
    params = {k: v for k, v in instrument_parameters(('X', 'Y')).items() if k not in ('sigma_w_X', 'sigma_w_Y')}
    params['sigma_w_X_Y'] = 60.
    data = lc_data(('X', 'Y'), n=40)
    if mode == 'global':
        data['GP_regressors_lc'] = {'lc': stack(data['t_lc'], ('X', 'Y'))}
        params.update(gp_parameters('CeleriteMaternKernel', 'lc', 'lc'))
    else:
        data['GP_regressors_lc'] = {name: data['t_lc'][name] for name in ('X', 'Y')}
        params.update(gp_parameters('CeleriteMaternKernel', 'lc', 'X_Y'))
    return params, data


@pytest.mark.parametrize('mode,line', [('instrument', 'fit.py:5411'), ('global', 'fit.py:4573')])
@xfail(KeyError, 'A shared sigma_w_A_B is not understood once a GP is involved.')
def test_shared_jitter_with_gp(mode, line):
    params, data = shared_jitter_gp(mode)
    _, model = build(params, 'lc', **data)
    model.generate(params)
    assert np.isfinite(model.get_log_likelihood(params))


@pytest.mark.parametrize('remove,add', [(('mu_A', 'mu_B'), {'mu_A_B': 2.}),
                                        (('sigma_w_A', 'sigma_w_B'), {'sigma_w_A_B': 1.})],
                         ids=['mu', 'sigma_w'])
@xfail(KeyError, 'RVs don\'t support shared instrument parameters, unlike photometry: fit.py:2614, 2617.')
def test_rv_shared_instrument_parameters(remove, add):
    params = {k: v for k, v in rv_parameters(('A', 'B')).items() if k not in remove}
    params.update(add)
    _, model = build(params, 'rv', **rv_data(('A', 'B')))
    model.generate(params)
    assert np.isfinite(model.get_log_likelihood(params))


@xfail(IndexError, 'sort_GP needs two columns, but one-dimensional regressors are accepted elsewhere: fit.py:479.')
def test_one_dimensional_unsorted_global_regressor():
    params = rv_parameters(('X', 'Y'))
    params.update(gp_parameters('CeleriteMaternKernel', 'rv', 'rv'))
    data = rv_data(('X', 'Y'))
    data['GP_regressors_rv'] = {'rv': stack(data['t_rv'], ('X', 'Y'))[:, 0]}
    build(params, 'rv', **data)


def unknown_gp_input():
    params = dict(rv_parameters(), GP_nonsense_A=1.)
    data = rv_data()
    data['GP_regressors_rv'] = {'A': data['t_rv']['A']}
    return params, data


@xfail(AssertionError, 'Unknown GP parameter names give "KeyError: None" instead of naming the problem: fit.py:5624.')
def test_unknown_gp_parameter_has_a_useful_message():
    params, data = unknown_gp_input()
    with pytest.raises(Exception, match='nonsense'):
        build(params, 'rv', **data)


@xfail(KeyError, 'An instrument named like a GP parameter (B, C, L, Q...) is mistaken for it, because get_kernel_name '
                 'looks for the instrument anywhere in the prior name: fit.py:5624.')
def test_instrument_named_like_a_gp_parameter():
    params = rv_parameters(('A', 'B'))
    for name in ('A', 'B'):
        params.update(gp_parameters('CeleriteQPKernel', 'rv', name))
    data = rv_data(('A', 'B'))
    data['GP_regressors_rv'] = {name: data['t_rv'][name] for name in ('A', 'B')}
    _, model = build(params, 'rv', **data)
    model.generate(params)
    assert np.isfinite(model.get_log_likelihood(params))


@xfail(ImportError, 'plots.py imports mpl_toolkits.axes_grid, which matplotlib has removed, and seaborn: plots.py:2.')
def test_plots_module_imports():
    import juliet.plots  # noqa: F401


# ---------------------------------------------------------------- Options that never reach dynesty (issue #138)
class StopBeforeSampling(Exception):
    pass


@pytest.fixture
def dynesty_call(monkeypatch):
    """Stops juliet.fit at the moment it creates the sampler, and records what the sampler was given."""
    seen = {}

    def record(cls, loglikelihood, prior_transform, ndim, **kwargs):
        seen.update(kwargs, loglikelihood=loglikelihood)
        raise StopBeforeSampling

    monkeypatch.setattr(dynesty.NestedSampler, '__new__', record)
    return seen


def small_fit(**load_kwargs):
    params = transit_parameters()
    return build(params, 'lc', **dict(lc_data(), **load_kwargs))[0]


@xfail(AssertionError, 'Issue #138: arguments are matched against vars(DynestySampler)["__init__"], which doesn\'t '
                       'exist on NestedSampler, so nothing matches: fit.py:2043. No seeded dynesty run is possible.')
@pytest.mark.parametrize('name,value', [('bound', 'single'), ('sample', 'unif'), ('walks', 7)])
def test_dynesty_keywords_reach_the_sampler(dynesty_call, name, value):
    with pytest.raises(StopBeforeSampling):
        small_fit().fit(sampler='dynesty', **{name: value})
    assert dynesty_call.get(name) == value


@xfail(AssertionError, 'Issue #138: see test_dynesty_keywords_reach_the_sampler.')
def test_dynesty_rstate_reaches_the_sampler(dynesty_call):
    rstate = np.random.default_rng(1)
    with pytest.raises(StopBeforeSampling):
        small_fit().fit(sampler='dynesty', rstate=rstate)
    assert dynesty_call.get('rstate') is rstate


def test_dynesty_default_options(dynesty_call):
    with pytest.raises(StopBeforeSampling):
        small_fit().fit(sampler='dynesty', n_live_points=123)
    assert (dynesty_call['bound'], dynesty_call['sample'], dynesty_call['nlive']) == ('multi', 'rwalk', 123)


@xfail(NameError, 'loglike uses extra_loglikelihood, missing self.: fit.py:1693.')
def test_extra_loglikelihood(dynesty_call):
    extra = {'loglikelihood': lambda parameters: -1.5}
    with pytest.raises(StopBeforeSampling):
        small_fit(extra_loglikelihood=extra).fit(sampler='dynesty')
    loglike = dynesty_call['loglikelihood']
    values = transit_parameters()
    assert np.isfinite(loglike([values[name] for name in loglike.__self__.model_parameters]))
