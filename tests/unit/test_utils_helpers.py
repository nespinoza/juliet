"""get_phases, get_quantiles and bin_data."""
import numpy as np
import pytest

from juliet import utils

pytestmark = pytest.mark.unit


def test_get_phases_array_range_and_values():
    t = np.array([0., 0.25, 0.5, 0.75, 1.25, 10.])
    phases = utils.get_phases(t, 1., 0.)
    np.testing.assert_allclose(phases, [0., 0.25, -0.5, -0.25, 0.25, 0.], atol=1e-14)


def test_get_phases_scalar_matches_array():
    t = np.array([0.8])
    assert utils.get_phases(0.8, 1., 0.) == pytest.approx(utils.get_phases(t, 1., 0.)[0])


def test_get_phases_uses_medians_of_samples():
    t = np.linspace(0., 3., 7)
    expected = utils.get_phases(t, 1.5, 0.2)
    np.testing.assert_allclose(utils.get_phases(t, np.array([1.4, 1.5, 1.6]), np.array([0.1, 0.2, 0.3])), expected)


def test_get_phases_phmin_moves_the_wrap():
    assert utils.get_phases(np.array([0.7]), 1., 0., phmin=0.8)[0] == pytest.approx(0.7)
    assert utils.get_phases(np.array([0.9]), 1., 0., phmin=0.8)[0] == pytest.approx(-0.1)


def test_get_quantiles_odd_count():
    dist = np.arange(1001.)
    np.random.RandomState(1).shuffle(dist)
    median, upper, lower = utils.get_quantiles(dist)
    assert median == 500.
    assert upper == 500. + int(1001 * 0.34 + 1)
    assert lower == 500. - int(1001 * 0.34 + 1)


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason='get_quantiles even-count median averages ordered[n/2] and ordered[n/2+1]; the median is '
                          'ordered[n/2-1] and ordered[n/2]. Off by half a sample: utils.py:995.')
def test_get_quantiles_even_count_median():
    assert utils.get_quantiles(np.arange(1000.))[0] == np.median(np.arange(1000.))


def test_get_quantiles_even_count_current_values():
    # Pins today's output, so that the fix to the median above is a visible, deliberate change:
    assert utils.get_quantiles(np.arange(1000.)) == (500.5, 842., 159.)


def test_get_quantiles_matches_percentiles_on_a_normal():
    dist = np.random.RandomState(2).normal(size=200001)
    median, upper, lower = utils.get_quantiles(dist)
    np.testing.assert_allclose([median, upper, lower], np.percentile(dist, [50., 84., 16.]), atol=2e-3)


@pytest.mark.parametrize('method,function', [('median', np.median), ('mean', np.mean)])
def test_bin_data_values_current_behaviour(method, function):
    # Each bin spans n_bin points but only uses its first n_bin - 1 (see the xfail below).
    x, y = np.arange(12.), np.arange(12.)**2
    xb, yb, _ = utils.bin_data(x, y, 4, method=method)
    assert len(xb) == 3
    np.testing.assert_allclose(xb, [function(x[i:i + 3]) for i in (0, 4, 8)])
    np.testing.assert_allclose(yb, [function(y[i:i + 3]) for i in (0, 4, 8)])


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason='bin_data slices [i:i + n_bin - 1], so the last point of every bin is dropped: utils.py:1017.')
def test_bin_data_uses_every_point():
    x = np.arange(4.)
    assert utils.bin_data(x, x, 4, method='mean')[0][0] == np.mean(x)


def test_bin_data_errors_current_behaviour():
    y = np.array([1., 2., 3., 4., 10., 20., 30., 40.])
    _, _, err = utils.bin_data(np.arange(8.), y, 4, method='mean')
    np.testing.assert_allclose(err, [np.std(y[:3]) / np.sqrt(3), np.std(y[4:7]) / np.sqrt(3)])
    yerr = np.full(8, 0.3)
    _, _, err = utils.bin_data(np.arange(8.), y, 4, yerr=yerr, method='mean')
    np.testing.assert_allclose(err, 0.3 * np.sqrt(3) / 3.)


def test_bin_data_rejects_unknown_method():
    with pytest.raises(Exception, match='not recognized'):
        utils.bin_data(np.arange(4.), np.arange(4.), 2, method='mode')
