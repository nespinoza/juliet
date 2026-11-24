"""
Defects found in the full code review (section 2.6 of docs/dev/refactoring/fable_claude_refactoring.md).

Like test_known_bugs.py: each test states the correct behaviour and is a strict xfail, so the fix has to remove the
mark. No sampler runs here. Where a test needs a finished fit, a results file is written first (see helpers.py).
"""
import numpy as np
import pytest
from scipy import stats

import juliet
from contract.helpers import fit_without_sampling, free_vector, write_results
from support.build import make_priors, make_samples, rv_parameters, transit_parameters
from support.datasets import lc_data, rv_data

pytestmark = pytest.mark.contract


def xfail(exception, reason):
    return pytest.mark.xfail(strict=True, raises=exception, reason=reason)


class SamplerRan(Exception):
    pass


@pytest.fixture
def sampler_calls(monkeypatch):
    """Replaces dynesty's sampler by one that only records that it was asked to run."""
    import dynesty
    calls = []

    def record(*args, **kwargs):
        calls.append(1)
        raise SamplerRan()

    monkeypatch.setattr(dynesty.NestedSampler, '__new__', record)
    return calls


def fit_recording_the_sampler(dataset):
    try:
        return dataset.fit(sampler='dynesty', n_live_points=20)
    except SamplerRan:
        return None


# ---------------------------------------------------------------- Stale results (step 1.7b, decision D11)
STALE_CWD = 'fit sets out_folder to the working directory when there is none, and loads any results file it finds ' \
            'there: fit.py:1922. The second dataset gets the first one\'s posterior.'
STALE_FOLDER = 'load.save() never overwrites lc.dat, rvs.dat or priors.dat, and fit never overwrites the results ' \
               'file: fit.py:1086, 1924.'


@pytest.mark.writes_to_cwd
@xfail(AssertionError, STALE_CWD)
def test_a_fit_without_out_folder_ignores_results_of_another_dataset(tmp_path, sampler_calls):
    # What an earlier fit of other data, also without out_folder, left in the working directory:
    params = transit_parameters()
    write_results(str(tmp_path) + '/', make_samples(params, nsamples=20))
    dataset = juliet.load(priors=make_priors(params), **lc_data(seed=12))
    results = fit_recording_the_sampler(dataset)
    assert sampler_calls and results is None, 'The fit returned the posterior of a different dataset.'


@xfail(AssertionError, STALE_FOLDER)
def test_changed_priors_in_the_same_out_folder_are_not_ignored(out_folder, sampler_calls):
    params = transit_parameters()
    fit_without_sampling(params, out_folder, **lc_data())
    # The same data again, now with the period fixed. D11: this must raise or refit, not return the old posterior.
    dataset = juliet.load(priors=make_priors(params, fixed=('P_p1',)), out_folder=out_folder, **lc_data())
    try:
        results = fit_recording_the_sampler(dataset)
    except Exception:
        return
    assert sampler_calls and results is None, 'The fit returned the posterior of the earlier priors.'


@xfail(AssertionError, STALE_FOLDER)
def test_changed_data_in_the_same_out_folder_are_not_ignored(out_folder, sampler_calls):
    params = transit_parameters()
    fit_without_sampling(params, out_folder, **lc_data(seed=11))
    dataset = juliet.load(priors=make_priors(params), out_folder=out_folder, **lc_data(seed=12))
    try:
        results = fit_recording_the_sampler(dataset)
    except Exception:
        return
    assert sampler_calls and results is None, 'The fit returned the posterior of the earlier data.'


def test_the_same_fit_in_the_same_out_folder_is_read_back(out_folder, sampler_calls):
    """The documented way to get results again, which D11 keeps."""
    params = transit_parameters()
    first = fit_without_sampling(params, out_folder, **lc_data())
    again = juliet.load(priors=make_priors(params), out_folder=out_folder, **lc_data()).fit(sampler='dynesty')
    assert not sampler_calls
    np.testing.assert_array_equal(again.posteriors['posterior_samples']['p_p1'],
                                  first.posteriors['posterior_samples']['p_p1'])


# ---------------------------------------------------------------- evaluate with few samples (step 1.5)
@pytest.mark.parametrize('kind,parameters,data', [('lc', transit_parameters, lc_data), ('rv', rv_parameters, rv_data)])
@xfail(AssertionError, 'evaluate allocates nsamples (1000) rows, fills only the available samples and takes the median '
                       'over all rows, so with fewer samples the model is zero: fit.py:2925.')
def test_evaluate_with_fewer_samples_than_nsamples(kind, parameters, data, out_folder):
    results = fit_without_sampling(parameters(), out_folder, nsamples=400, **data())
    model = getattr(results, kind)
    np.testing.assert_allclose(model.evaluate('A'), model.evaluate('A', all_samples=True), rtol=1e-10)


def test_evaluate_with_more_samples_than_nsamples_draws_a_subset(out_folder):
    """Frozen: above nsamples, evaluate() uses a random subset, which is close to all the samples but not equal."""
    results = fit_without_sampling(transit_parameters(), out_folder, nsamples=1500, **lc_data())
    subset, everything = results.lc.evaluate('A'), results.lc.evaluate('A', all_samples=True)
    np.testing.assert_allclose(subset, everything, rtol=1e-4)


# ---------------------------------------------------------------- posteriors.dat (step 1.6)
def written_inclination(out_folder, fixed):
    """posteriors.dat has a derived inclination for the (r1, r2) parametrization."""
    params = {name: value for name, value in transit_parameters().items() if name not in ('b_p1', 'p_p1')}
    params.update(r1_p1=0.4, r2_p1=0.1, ecc_p1=0.3, omega_p1=60.)
    fit_without_sampling(params, out_folder, fixed=fixed, **lc_data())
    with open(out_folder + 'posteriors.dat') as f:
        rows = {line.split()[0]: float(line.split()[1]) for line in f if not line.startswith('#')}
    ecc, omega = params['ecc_p1'], np.radians(params['omega_p1'])
    expected = np.degrees(np.arccos(rows['b_p1'] / params['a_p1'] * (1. + ecc * np.sin(omega)) / (1. - ecc**2)))
    return rows['inc_p1'], expected


def test_inclination_in_posteriors_dat_with_ecc_and_omega_free(out_folder):
    written, expected = written_inclination(out_folder, fixed=())
    assert written == pytest.approx(expected, abs=1e-3)


@pytest.mark.parametrize('fixed', ['omega_p1', 'ecc_p1'])
@xfail(AssertionError, 'With one of ecc, omega fixed, writepp uses omega in degrees inside np.sin: utils.py:1100.')
def test_inclination_in_posteriors_dat_with_one_of_ecc_omega_fixed(fixed, out_folder):
    written, expected = written_inclination(out_folder, fixed=(fixed,))
    assert written == pytest.approx(expected, abs=1e-3)


# ---------------------------------------------------------------- Inputs (steps 1.6 and 1.7)
@xfail(ValueError, 'readpriors unpacks every line that isn\'t a comment, so a blank line fails: utils.py:755.')
def test_prior_file_with_a_blank_line(tmp_path):
    path = tmp_path / 'priors.dat'
    path.write_text('P_p1 normal 1.7,0.1\n\nb_p1 uniform 0.,1.\n')
    assert set(juliet.utils.readpriors(str(path))[0]) == {'P_p1', 'b_p1'}


def underscore_dataset():
    data = lc_data()
    renamed = {key: {'LCO_i': values['A']} for key, values in data.items()}
    params = {name.replace('_A', '_LCO_i'): value for name, value in transit_parameters().items()}
    return juliet.load(priors=make_priors(params), **renamed)


@xfail(AssertionError, 'Parameter names are split on "_" to find their instruments, so q1_LCO_i is read as instrument '
                       '"i" and LCO_i gets no transit, with no error. Issue #2 is closed but this remains.')
def test_instrument_name_with_an_underscore_keeps_its_transit():
    try:
        dataset = underscore_dataset()
    except Exception:
        return  # Refusing the name with a clear error would also be correct.
    assert dataset.lc_options['LCO_i']['TransitFit']


def hyphen_inputs():
    data = lc_data(('A', 'B'))
    renamed = {key: {'TESS-S1': values['A'], 'B': values['B']} for key, values in data.items()}
    params = {name.replace('_A', '_TESS-S1'): value for name, value in transit_parameters(('A', 'B')).items()}
    return params, renamed


@xfail(ValueError, 'ld_laws is split on "-" to separate instrument from law, so an instrument with a "-" fails.')
def test_ld_laws_for_an_instrument_name_with_a_hyphen():
    params, data = hyphen_inputs()
    dataset = juliet.load(priors=make_priors(params), ld_laws='TESS-S1-linear,B-quadratic', **data)
    assert dataset.lc_options['TESS-S1']['ldlaw'] == 'linear'


def test_instrument_name_with_a_hyphen_works_without_ld_laws():
    params, data = hyphen_inputs()
    assert juliet.load(priors=make_priors(params), **data).lc_options['TESS-S1']['TransitFit']


@xfail(ImportError, '__all__ lists "plots", which isn\'t imported by the package (and can\'t be: see '
                       'test_plots_module_imports).')
def test_star_import():
    namespace = {}
    exec('from juliet import *', namespace)
    assert 'load' in namespace and 'fit' in namespace


# ---------------------------------------------------------------- The Exponential prior (decision D12)
def test_exponential_prior_is_a_gamma_distribution_in_a_fit(tmp_path):
    """Frozen, not a defect to fix: D12 keeps the Gamma(shape) behaviour and corrects the documentation."""
    out_folder = str(tmp_path / 'fit') + '/'
    (tmp_path / 'fit').mkdir()
    # From a prior file, as the docs describe. (The same prior in a dictionary fails: see the next test.)
    params = transit_parameters()
    lines = ['{} fixed {}'.format(name, value) for name, value in params.items() if name != 'sigma_w_A']
    (tmp_path / 'fit' / 'priors.dat').write_text('\n'.join(lines + ['sigma_w_A exponential 2.0']) + '\n')
    write_results(out_folder, {'sigma_w_A': np.linspace(40., 60., 20)})
    data = lc_data()
    results = juliet.load(priors=out_folder + 'priors.dat', out_folder=out_folder, **data).fit(sampler='dynesty')
    assert results.paramnames == ['sigma_w_A']
    assert float(np.squeeze(results.prior_transform_r(np.array([0.5]))[0])) == pytest.approx(stats.gamma(2.).ppf(0.5))


@xfail(TypeError, 'save_priorfile joins the hyperparameters as a list, but an exponential prior has a single number: '
                  'fit.py:413. So this prior can\'t be given in a dictionary together with out_folder.')
def test_exponential_prior_in_a_dictionary_with_out_folder(out_folder):
    params = transit_parameters()
    priors = make_priors(params, fixed=tuple(params))
    priors['sigma_w_A'] = {'distribution': 'exponential', 'hyperparameters': 2.}
    juliet.load(priors=priors, out_folder=out_folder, **lc_data())


# ---------------------------------------------------------------- Found while writing step 0.3
@xfail(KeyError, 'fit() ends by replacing its dictionary of current parameter values with the results, which have no '
                 'fixed parameters, so loglike can\'t be called on the returned object: fit.py:2488.')
def test_loglike_of_a_finished_fit_with_fixed_parameters(out_folder):
    params = transit_parameters()
    results = fit_without_sampling(params, out_folder, fixed=('ecc_p1', 'omega_p1'), **lc_data())
    assert np.isfinite(results.loglike(free_vector(results, params)))
