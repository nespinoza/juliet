"""Prior transforms against scipy.stats ppf, and log-priors against scipy.stats logpdf."""
import numpy as np
import pytest
from scipy import stats

from juliet import utils

pytestmark = pytest.mark.unit

QUANTILES = np.array([1e-4, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1. - 1e-4])


def truncnorm(mu, sigma, a, b):
    return stats.truncnorm((a - mu) / sigma, (b - mu) / sigma, loc=mu, scale=sigma)


# (transform, log-density, hyperparameters, scipy distribution, points inside the support)
DISTRIBUTIONS = {
    'uniform': (utils.transform_uniform, utils.evaluate_uniform, [2., 5.], stats.uniform(2., 3.)),
    'loguniform': (utils.transform_loguniform, utils.evaluate_loguniform, [0.1, 100.], stats.loguniform(0.1, 100.)),
    'normal': (utils.transform_normal, utils.evaluate_normal, [1.5, 0.3], stats.norm(1.5, 0.3)),
    'truncated_normal': (utils.transform_truncated_normal, utils.evaluate_truncated_normal, [1., 0.5, 0.2, 1.6],
                         truncnorm(1., 0.5, 0.2, 1.6)),
}


@pytest.mark.parametrize('name', DISTRIBUTIONS)
def test_transform_matches_ppf(name):
    transform, _, hyperparameters, distribution = DISTRIBUTIONS[name]
    computed = [transform(x, hyperparameters) for x in QUANTILES]
    np.testing.assert_allclose(computed, distribution.ppf(QUANTILES), rtol=1e-10)


@pytest.mark.parametrize('name', DISTRIBUTIONS)
def test_log_prior_matches_logpdf(name):
    _, evaluate, hyperparameters, distribution = DISTRIBUTIONS[name]
    points = distribution.ppf(QUANTILES)
    computed = [evaluate(x, hyperparameters) for x in points]
    np.testing.assert_allclose(computed, distribution.logpdf(points), rtol=1e-10)


@pytest.mark.parametrize('name', ['uniform', 'loguniform', 'truncated_normal'])
def test_log_prior_is_minus_infinity_outside_support(name):
    _, evaluate, hyperparameters, distribution = DISTRIBUTIONS[name]
    low, high = distribution.support()
    assert evaluate(low - 1e-3, hyperparameters) == -np.inf
    assert evaluate(high + 1e-3, hyperparameters) == -np.inf


def test_beta_transform_matches_ppf():
    computed = [utils.transform_beta(x, [2., 3.]) for x in QUANTILES]
    np.testing.assert_allclose(computed, stats.beta(2., 3.).ppf(QUANTILES), rtol=1e-10)


@pytest.mark.parametrize('a,b', [(0.5, 0.5)])
def test_beta_log_prior_matches_logpdf_when_a_positive_and_b_below_one(a, b):
    points = np.array([0.1, 0.5, 0.9])
    computed = [utils.evaluate_beta(x, [a, b]) for x in points]
    np.testing.assert_allclose(computed, stats.beta(a, b).logpdf(points), rtol=1e-10)


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason='evaluate_beta tests the hyperparameters (a > 0 and b < 1) instead of x, so any prior '
                          'with b >= 1 gives -inf everywhere: utils.py:661.')
def test_beta_log_prior_for_b_above_one():
    assert utils.evaluate_beta(0.5, [2., 3.]) == pytest.approx(stats.beta(2., 3.).logpdf(0.5))


def test_exponential_transform_is_gamma_ppf():
    # hyperparameters is the gamma shape, not the exponential's rate or scale. Shape 1 is the standard exponential:
    computed = [utils.transform_exponential(x, 1.) for x in QUANTILES]
    np.testing.assert_allclose(computed, stats.expon().ppf(QUANTILES), rtol=1e-10)
    computed = [utils.transform_exponential(x, 2.5) for x in QUANTILES]
    np.testing.assert_allclose(computed, stats.gamma(2.5).ppf(QUANTILES), rtol=1e-10)


def test_exponential_log_prior_is_gamma_logpdf():
    points = np.array([0.1, 1., 4.])
    np.testing.assert_allclose([utils.evaluate_exponential(x, 2.5) for x in points], stats.gamma(2.5).logpdf(points))
    assert utils.evaluate_exponential(-1., 2.5) == -np.inf


def modified_jeffreys_ppf(x, turn, high):
    return turn * (np.exp(x * np.log(high / turn + 1.)) - 1.)


def test_modified_jeffreys_transform():
    # The code offsets x by 1e-10 so the lower limit isn't exactly zero, hence the tolerance at low quantiles:
    computed = [utils.transform_modifiedjeffreys(x, [1., 100.]) for x in QUANTILES]
    np.testing.assert_allclose(computed, modified_jeffreys_ppf(QUANTILES, 1., 100.), rtol=1e-5)


def test_modified_jeffreys_log_prior_is_a_normalized_density():
    turn, high = 1., 100.
    grid = np.linspace(1e-6, high - 1e-6, 200001)
    density = np.exp([utils.evaluate_modifiedjeffreys(x, [turn, high]) for x in grid])
    assert np.sum(0.5 * (density[1:] + density[:-1]) * np.diff(grid)) == pytest.approx(1., abs=1e-4)
    assert utils.evaluate_modifiedjeffreys(-0.5, [turn, high]) == -np.inf
    assert utils.evaluate_modifiedjeffreys(high + 1., [turn, high]) == -np.inf


def test_transforms_and_log_priors_are_consistent():
    # A density computed from each transform's numerical derivative should match its log-prior:
    for name, (transform, evaluate, hyperparameters, _) in DISTRIBUTIONS.items():
        x = 0.4
        step = 1e-6
        derivative = (transform(x + step, hyperparameters) - transform(x - step, hyperparameters)) / (2. * step)
        assert evaluate(transform(x, hyperparameters), hyperparameters) == pytest.approx(-np.log(derivative), abs=1e-6)
