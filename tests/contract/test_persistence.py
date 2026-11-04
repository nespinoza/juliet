"""Reading results written by juliet. The fixture was written by the code at baseline-2.2.11 (see make_fixture.py)."""
import os
import pickle
import re
import shutil
from pathlib import Path

import dynesty
import numpy as np
import pytest

import juliet
from support.build import make_priors, transit_parameters
from support.datasets import lc_data, rv_data

pytestmark = pytest.mark.contract

FIXTURE = Path(__file__).parent / 'fixtures' / 'transit_fit'


@pytest.fixture
def fixture_folder(tmp_path):
    folder = tmp_path / 'fit'
    shutil.copytree(FIXTURE, folder)
    return str(folder) + '/'


def stored():
    with open(FIXTURE / '_dynesty_NS_posteriors.pkl', 'rb') as f:
        return pickle.load(f)


def test_pickle_has_the_documented_keys():
    out = stored()
    assert {'posterior_samples', 'lnZ', 'lnZerr'} <= set(out)
    assert {'unnamed', 'loglike', 't0_p1', 'p_p1', 'mflux_A'} == set(out['posterior_samples'])


def test_results_are_read_from_an_input_folder_without_sampling(fixture_folder, monkeypatch):
    def no_sampling(*args, **kwargs):
        raise AssertionError('Results exist, so no sampler should run.')

    monkeypatch.setattr(dynesty.NestedSampler, '__new__', no_sampling)
    results = juliet.load(input_folder=fixture_folder).fit(sampler='dynesty')
    out = stored()
    assert results.posteriors['lnZ'] == out['lnZ']
    assert results.posteriors['lnZerr'] == out['lnZerr']
    for name, values in out['posterior_samples'].items():
        np.testing.assert_array_equal(results.posteriors['posterior_samples'][name], values)


def test_results_from_an_input_folder_can_be_evaluated(fixture_folder):
    results = juliet.load(input_folder=fixture_folder).fit(sampler='dynesty')
    model = results.lc.evaluate('A', all_samples=True)
    assert model.shape == (40,)
    assert np.all(np.isfinite(model))


def test_reading_results_is_stable_for_the_model_value(fixture_folder):
    # The stored median model, so that a change in how results are read and applied is noticed:
    results = juliet.load(input_folder=fixture_folder).fit(sampler='dynesty')
    model = results.lc.evaluate('A', all_samples=True)
    with np.load(FIXTURE.parent / 'transit_fit_model.npz') as saved:
        np.testing.assert_allclose(model, saved['model'], rtol=1e-10)


POSTERIORS_HEADER = re.compile(r'^# Parameter Name\s+Median\s+Upper 68 CI\s+Lower 68 CI\s*$')
POSTERIORS_ROW = re.compile(r'^(\S+)\s+(-?\d+\.\d{10})\s+(-?\d+\.\d{10})\s+(-?\d+\.\d{10})\s*$')


def parse_posteriors_dat(path):
    with open(path) as f:
        lines = f.read().splitlines()
    assert POSTERIORS_HEADER.match(lines[0]), lines[0]
    rows = {}
    for line in lines[1:]:
        match = POSTERIORS_ROW.match(line)
        assert match, line
        rows[match.group(1)] = [float(x) for x in match.groups()[1:]]
    return rows


def test_posteriors_dat_format_in_the_fixture():
    assert set(parse_posteriors_dat(FIXTURE / 'posteriors.dat')) == {'t0_p1', 'p_p1', 'mflux_A'}


def test_posteriors_dat_is_rewritten_identically(fixture_folder):
    os.remove(fixture_folder + 'posteriors.dat')
    juliet.load(input_folder=fixture_folder).fit(sampler='dynesty')
    with open(fixture_folder + 'posteriors.dat') as f, open(FIXTURE / 'posteriors.dat') as g:
        assert f.read() == g.read()


def test_posteriors_dat_values_are_the_quantiles():
    rows = parse_posteriors_dat(FIXTURE / 'posteriors.dat')
    samples = stored()['posterior_samples']
    for name, (median, upper, lower) in rows.items():
        value, up, down = juliet.utils.get_quantiles(samples[name])
        assert (median, upper, lower) == pytest.approx((value, up - value, value - down), abs=1e-9)


def test_load_writes_inputs_and_reads_them_back(out_folder):
    params = transit_parameters()
    data = lc_data()
    juliet.load(priors=make_priors(params), out_folder=out_folder, **data)
    reloaded = juliet.load(input_folder=out_folder)
    np.testing.assert_allclose(reloaded.times_lc['A'], data['t_lc']['A'], rtol=1e-12)
    np.testing.assert_allclose(reloaded.data_lc['A'], data['y_lc']['A'], rtol=1e-8)
    np.testing.assert_allclose(reloaded.errors_lc['A'], data['yerr_lc']['A'], rtol=1e-8)
    assert set(reloaded.priors) == set(params)


def test_reading_prior_files(tmp_path):
    path = tmp_path / 'priors.dat'
    path.write_text('P_p1 Normal 1.7,0.1\nb_p1 Uniform 0.,1.\necc_p1 FIXED 0.0\nsigma_w_A loguniform 1.,100.\n'
                    'sigma_w_rv_R Jeffreys 0.1,10.\n# a comment\n')
    output = juliet.utils.readpriors(str(path))
    priors = output[0]
    assert priors['P_p1']['distribution'] == 'normal'
    assert priors['P_p1']['hyperparameters'] == [1.7, 0.1]
    assert priors['ecc_p1']['distribution'] == 'fixed'
    assert priors['ecc_p1']['hyperparameters'] == 0.
    # Old names for RV jitter keep working:
    assert 'sigma_w_R' in priors


def test_reading_data_files(tmp_path):
    path = tmp_path / 'data.dat'
    path.write_text('1.0 2.0 0.1 A\n1.5 2.5 0.2 B\n2.0 3.0 0.3 A\n')
    t, y, yerr, instruments, indexes, n_instruments, names = juliet.utils.read_data(str(path))[:7]
    np.testing.assert_array_equal(t, [1., 1.5, 2.])
    np.testing.assert_array_equal(y, [2., 2.5, 3.])
    np.testing.assert_array_equal(yerr, [0.1, 0.2, 0.3])
    np.testing.assert_array_equal(instruments, ['A', 'B', 'A'])
    np.testing.assert_array_equal(indexes['A'], [0, 2])
    assert n_instruments == 2
    assert names == ['A', 'B']


def test_dictionaries_and_rv_only_inputs_load(out_folder):
    data = rv_data()
    params = {'P_p1': 10., 't0_p1': 2458461., 'K_p1': 5., 'ecc_p1': 0., 'omega_p1': 90., 'mu_A': 10., 'sigma_w_A': 1.}
    dataset = juliet.load(priors=make_priors(params), out_folder=out_folder, **data)
    assert dataset.inames_rv == ['A']
    reloaded = juliet.load(input_folder=out_folder)
    np.testing.assert_allclose(reloaded.data_rv['A'], data['y_rv']['A'], rtol=1e-6)
