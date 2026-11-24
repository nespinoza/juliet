"""
The light-travel-time correction of eclipses (fit(light_travel_delay=True, stellar_radius=...)).

The planet is closer to the observer at transit than at eclipse, so the eclipse is seen later than a clock set at
transit predicts, by the time light takes to cross the line-of-sight distance between the two positions. Here that
distance is computed from the orbit equation, independently of juliet's KeplerOrbit.
"""
import astropy.constants as const
import batman
import numpy as np
import pytest

from juliet.utils import correct_light_travel_time
from support.build import build, transit_parameters

pytestmark = pytest.mark.unit

A, INC, RS, PERIOD = 3.6, 88., 1.2, 1.
RS_LIGHT_DAYS = RS * const.R_sun.value / const.c.value / 86400.


def orbit(ecc=0., omega=90.):
    params = batman.TransitParams()
    params.t0, params.per, params.a, params.rp, params.inc = 0., PERIOD, A, 0.1, INC
    params.ecc, params.w, params.Rs = ecc, omega, RS
    params.limb_dark, params.u = 'quadratic', [0.2, 0.3]
    return params


def delay(times, params):
    """How much later than `times` the events are seen, in days."""
    return times - correct_light_travel_time(times, params)


def line_of_sight(true_anomaly, ecc, omega):
    """Distance of the planet behind the plane of the sky through the star, in stellar radii (negative: in front)."""
    r = A * (1. - ecc**2) / (1. + ecc * np.cos(true_anomaly))
    return -r * np.sin(np.radians(omega) + true_anomaly) * np.sin(np.radians(INC))


def test_circular_orbit_delay_follows_the_orbit():
    full = 2. * A * np.sin(np.radians(INC)) * RS_LIGHT_DAYS
    # About 20 seconds for these values:
    assert 15. < full * 86400. < 25.
    quarters = np.array([0., 0.25, 0.5, 0.75]) * PERIOD
    np.testing.assert_allclose(delay(quarters, orbit()), full * np.array([0., 0.5, 1., 0.5]), rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize('ecc,omega', [(0.3, 90.), (0.3, 0.), (0.3, 200.), (0.6, -40.)])
def test_eccentric_orbit_delay_at_transit_and_eclipse(ecc, omega):
    params = orbit(ecc, omega)
    t_eclipse = batman.TransitModel(params, np.array([0., 0.5])).get_t_secondary(params)
    at_transit, at_eclipse = delay(np.array([params.t0, t_eclipse]), params)
    f_transit, f_eclipse = np.pi / 2. - np.radians(omega), 3. * np.pi / 2. - np.radians(omega)
    expected = (line_of_sight(f_eclipse, ecc, omega) - line_of_sight(f_transit, ecc, omega)) * RS_LIGHT_DAYS
    assert abs(at_transit) < 1e-12
    np.testing.assert_allclose(at_eclipse, expected, rtol=1e-8)


def test_nearly_circular_orbit_matches_the_circular_branch():
    # The two branches of correct_light_travel_time (see PR #124):
    times = np.linspace(0., PERIOD, 50)
    np.testing.assert_allclose(delay(times, orbit(ecc=1e-9)), delay(times, orbit()), rtol=0., atol=1e-11)


# ---------------------------------------------------------------- In juliet's light-curve model
T0, P_JULIET, FP = 2458460.5, 1.7, 0.01
LTT = {'light_travel_delay': True, 'stellar_radius': RS}
TRANSIT_WINDOW = T0 + np.linspace(-0.1, 0.1, 120)
ECLIPSE_WINDOW = T0 + P_JULIET / 2. + np.linspace(-0.1, 0.1, 120)


def juliet_orbit(values):
    params = orbit()
    params.t0, params.per, params.a = values['t0_p1'], values['P_p1'], values['a_p1']
    params.inc = np.degrees(np.arccos(values['b_p1'] / values['a_p1']))
    return params


def model_at(times, values, **model_kwargs):
    data = {'t_lc': {'A': times}, 'y_lc': {'A': np.ones(len(times))}, 'yerr_lc': {'A': np.full(len(times), 1e-4)}}
    _, model = build(values, 'lc', model_kwargs=model_kwargs, **data)
    model.generate(values)
    assert model.modelOK
    return np.copy(model.model['A']['deterministic'])


@pytest.mark.parametrize('with_transit', [False, True], ids=['eclipse', 'transit_and_eclipse'])
def test_model_with_delay_is_the_model_without_it_at_earlier_times(with_transit):
    values = dict(transit_parameters(), fp_p1=FP)
    if not with_transit:
        del values['q1_A'], values['q2_A']
    no_delay = dict(values, t_secondary_p1=T0 + P_JULIET / 2.)
    times = np.concatenate([TRANSIT_WINDOW, ECLIPSE_WINDOW])
    # Only the eclipse is corrected; the transit defines the clock:
    earlier = np.concatenate([TRANSIT_WINDOW, correct_light_travel_time(ECLIPSE_WINDOW, juliet_orbit(values))])

    delayed = model_at(times, values, **LTT)
    np.testing.assert_allclose(delayed, model_at(earlier, no_delay), rtol=0., atol=1e-12)
    # The correction is visible on the eclipse's ingress and egress, and nowhere near the transit:
    difference = delayed - model_at(times, no_delay)
    assert np.max(np.abs(difference[len(TRANSIT_WINDOW):])) > 1e-5
    assert np.max(np.abs(difference[:len(TRANSIT_WINDOW)])) < 1e-12
