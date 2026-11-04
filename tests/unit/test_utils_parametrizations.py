"""Limb-darkening and (r1, r2) parametrizations against the formulas in the literature."""
import numpy as np
import pytest

from juliet import utils

pytestmark = pytest.mark.unit

TWO_PARAMETER_LAWS = ['quadratic', 'squareroot', 'logarithmic']


def test_reverse_q_coeffs_quadratic_values():
    # Moved from juliet/test_utils.py:
    assert utils.reverse_q_coeffs('quadratic', 2., 4.) == (36., 0.16666666666666666)


@pytest.mark.parametrize('law', TWO_PARAMETER_LAWS)
@pytest.mark.parametrize('q1,q2', [(0.3, 0.35), (0.01, 0.9), (0.81, 0.1)])
def test_q_to_u_round_trip(law, q1, q2):
    u1, u2 = utils.reverse_ld_coeffs(law, q1, q2)
    assert utils.reverse_q_coeffs(law, u1, u2) == pytest.approx((q1, q2), rel=1e-12)


@pytest.mark.parametrize('law', TWO_PARAMETER_LAWS)
def test_convert_ld_coeffs_matches_reverse_q_coeffs(law):
    assert utils.convert_ld_coeffs(law, 0.2, 0.3) == pytest.approx(utils.reverse_q_coeffs(law, 0.2, 0.3), rel=1e-14)


def test_reverse_ld_coeffs_quadratic_is_kipping_2013():
    q1, q2 = 0.3, 0.35
    assert utils.reverse_ld_coeffs('quadratic', q1, q2) == pytest.approx(
        (2. * np.sqrt(q1) * q2, np.sqrt(q1) * (1. - 2. * q2)), rel=1e-14)


def test_reverse_ld_coeffs_linear_passes_through():
    assert utils.reverse_ld_coeffs('linear', 0.3, 0.35) == (0.3, 0.35)


def espinoza_2018(r1, r2, pl, pu):
    """Eqs. in Sec. 2 of Espinoza (2018) and Sec. 2.1 of Espinoza et al. (2019), written independently."""
    Ar = (pu - pl) / (2. + pl + pu)
    if r1 > Ar:
        return (1. + pl) * (1. + (r1 - 1.) / (1. - Ar)), (1. - r2) * pl + r2 * pu
    return (1. + pl) + np.sqrt(r1 / Ar) * r2 * (pu - pl), pu + (pl - pu) * np.sqrt(r1 / Ar) * (1. - r2)


@pytest.mark.parametrize('pl,pu', [(0., 1.), (0.05, 0.2), (0., 0.5)])
def test_reverse_bp_matches_formula(pl, pu):
    grid = np.linspace(0.005, 0.995, 25)
    r1, r2 = (values.ravel() for values in np.meshgrid(grid, grid))
    b, p = utils.reverse_bp(r1, r2, pl, pu)
    expected = np.array([espinoza_2018(x, y, pl, pu) for x, y in zip(r1, r2)])
    np.testing.assert_allclose(b, expected[:, 0], rtol=1e-14)
    np.testing.assert_allclose(p, expected[:, 1], rtol=1e-14)


@pytest.mark.parametrize('pl,pu', [(0., 1.), (0.05, 0.2)])
def test_reverse_bp_stays_in_the_allowed_triangle(pl, pu):
    # The point of the parametrization: every (r1, r2) gives 0 < b < 1 + p and pl < p < pu.
    r1, r2 = np.random.RandomState(1).uniform(size=(2, 5000))
    b, p = utils.reverse_bp(r1, r2, pl, pu)
    assert np.all((b >= 0.) & (b <= 1. + p + 1e-12))
    assert np.all((p >= pl - 1e-12) & (p <= pu + 1e-12))
