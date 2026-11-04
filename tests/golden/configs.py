"""
The configurations covered by the frozen-numerics layer, as data.

Add a configuration by appending to CONFIGS and running pytest with --update-golden (see tests/README.md).
Combinations are pairwise, not a full product, except that every GP kernel runs in global and per-instrument mode
for both photometry and RVs.
"""
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from scientific.targets import PAPER, data_arguments, paper_parameters
from support.build import instrument_parameters, rv_parameters, transit_parameters
from support.datasets import lc_data, lc_times, rv_data, stack
from support.kernels import KERNELS, gp_parameters

PLAIN, COMPONENTS, AT_T, TRANSIT = 'plain', 'components', 'at_t', 'transit'
PER_INSTRUMENT_OPS = (PLAIN, COMPONENTS, AT_T)
# These raise for the reasons in tests/contract/test_known_bugs.py:
NO_AT_T = (PLAIN, COMPONENTS)
NO_COMPONENTS = (PLAIN, AT_T)


@dataclass
class Config:
    id: str
    modeltypes: tuple
    data: Callable
    params: dict
    fixed: tuple = ()
    model_kwargs: dict = field(default_factory=dict)
    # Which evaluate calls to freeze. Those left out of a configuration raise today: see tests/contract.
    ops: tuple = PER_INSTRUMENT_OPS
    requires: Optional[str] = None
    # Samples that juliet.fit adds to a posterior without them having a prior, such as the t0 and P of (T) TTVs:
    extra_samples: dict = field(default_factory=dict)
    # Store only the likelihood, for large real datasets:
    loglike_only: bool = False


CONFIGS = []


def lc(id, params, data=None, instruments=('A',), **kwargs):
    build_data = data or (lambda: lc_data(instruments))
    CONFIGS.append(Config(id, ('lc',), build_data, params, **kwargs))


def rv(id, params, data=None, instruments=('A',), **kwargs):
    build_data = data or (lambda: rv_data(instruments))
    CONFIGS.append(Config(id, ('rv',), build_data, params, **kwargs))


def with_changes(base, remove=(), **changes):
    out = {k: v for k, v in base.items() if k not in remove}
    out.update(changes)
    return out


BASE = transit_parameters()
TWO_INSTRUMENTS = ('A', 'B')
# 'B' and 'C' are also names of GP parameters, which juliet confuses with instruments (see tests/contract):
GP_INSTRUMENTS = ('X', 'Y')
THREE_INSTRUMENTS = ('A', 'B', 'C')
ORBIT_REMOVE = ('ecc_p1', 'omega_p1')

# ---------------------------------------------------------------- Transit parametrizations
lc('lc_transit_ab', BASE)
lc('lc_transit_rho', with_changes(BASE, remove=('a_p1',), rho=1500.))
lc('lc_transit_r1r2', with_changes(BASE, remove=('b_p1', 'p_p1'), r1_p1=0.4, r2_p1=0.15))
lc('lc_transit_r1r2_low_r1', with_changes(BASE, remove=('b_p1', 'p_p1'), r1_p1=0.02, r2_p1=0.5))
lc('lc_transit_r1r2_pl_pu', with_changes(BASE, remove=('b_p1', 'p_p1'), r1_p1=0.4, r2_p1=0.4),
   model_kwargs={'pl': 0.05, 'pu': 0.2})
lc('lc_transit_r1r2_rho', with_changes(BASE, remove=('a_p1', 'b_p1', 'p_p1'), rho=1500., r1_p1=0.4, r2_p1=0.15))
lc('lc_transit_fixed_priors', BASE, fixed=('ecc_p1', 'omega_p1', 'mdilution_A', 'P_p1'))

# ---------------------------------------------------------------- Eccentricity
lc('lc_ecc_omega', with_changes(BASE, ecc_p1=0.2, omega_p1=40.))
lc('lc_ecosomega', with_changes(BASE, remove=ORBIT_REMOVE, ecosomega_p1=0.15, esinomega_p1=0.1))
lc('lc_secosomega', with_changes(BASE, remove=ORBIT_REMOVE, secosomega_p1=0.3, sesinomega_p1=0.25))
lc('lc_ecc_rho_r1r2', with_changes(BASE, remove=('a_p1', 'b_p1', 'p_p1'), rho=1500., r1_p1=0.4, r2_p1=0.15,
                                   ecc_p1=0.1, omega_p1=130.))
# Above ecclim the model refuses the point, so nothing is stored but the flag:
lc('lc_model_refused_above_ecclim', with_changes(BASE, ecc_p1=0.5), model_kwargs={'ecclim': 0.3}, ops=())

# ---------------------------------------------------------------- Limb darkening
NO_Q = ('q1_A', 'q2_A')
lc('lc_ld_quadratic_u', with_changes(BASE, remove=NO_Q, u1_A=0.3, u2_A=0.2))
lc('lc_ld_linear_q', with_changes(BASE, remove=('q2_A',)))
lc('lc_ld_linear_u', with_changes(BASE, remove=NO_Q, u1_A=0.4))


def _with_ld(law):
    return lambda: dict(lc_data(), ld_laws=law)


lc('lc_ld_squareroot_q', BASE, data=_with_ld('squareroot'))
lc('lc_ld_squareroot_u', with_changes(BASE, remove=NO_Q, u1_A=0.3, u2_A=0.2), data=_with_ld('squareroot'))
lc('lc_ld_logarithmic_q', BASE, data=_with_ld('logarithmic'))
lc('lc_ld_power2_u', with_changes(BASE, remove=NO_Q, u1_A=0.6, u2_A=0.7), data=_with_ld('power2'))
lc('lc_ld_nonlinear_c', with_changes(BASE, remove=NO_Q, c1_A=0.5, c2_A=-0.2, c3_A=0.4, c4_A=-0.1))
lc('lc_ld_per_instrument_laws', with_changes(transit_parameters(TWO_INSTRUMENTS), remove=('q2_B',)),
   data=lambda: dict(lc_data(TWO_INSTRUMENTS), ld_laws='A-quadratic,B-linear'))

# ---------------------------------------------------------------- Eclipses and phase curves
ECLIPSE = with_changes(BASE, remove=NO_Q, fp_p1=5e-4, t_secondary_p1=2458460.5 + 0.85)
TRANSIT_ECLIPSE = with_changes(BASE, fp_p1=5e-4, t_secondary_p1=2458460.5 + 0.85)
LTT = {'light_travel_delay': True, 'stellar_radius': 1.2}


def no_secondary_time(params):
    return with_changes(params, remove=('t_secondary_p1',))


lc('lc_eclipse', ECLIPSE)
lc('lc_transit_eclipse', TRANSIT_ECLIPSE)
lc('lc_transit_eclipse_sin_phase_curve', with_changes(TRANSIT_ECLIPSE, phaseoffset_p1=20.))
lc('lc_eclipse_sin_phase_curve', with_changes(ECLIPSE, phaseoffset_p1=20.))
lc('lc_cowan_agol', with_changes(TRANSIT_ECLIPSE, C1_p1=2e-4, D1_p1=5e-5, C2_p1=3e-5, D2_p1=1e-5))
lc('lc_lambert', with_changes(no_secondary_time(BASE), remove=(), aglambert_p1=0.3, t_secondary_p1=2458460.5 + 0.85))
lc('lc_lambert_plus_sin', with_changes(TRANSIT_ECLIPSE, aglambert_p1=0.3, phaseoffset_p1=20.))
lc('lc_eclipse_light_travel', no_secondary_time(ECLIPSE), model_kwargs=LTT, ops=NO_AT_T)
lc('lc_transit_eclipse_light_travel', no_secondary_time(TRANSIT_ECLIPSE), model_kwargs=LTT, ops=NO_AT_T)
lc('lc_cowan_agol_light_travel',
   no_secondary_time(with_changes(TRANSIT_ECLIPSE, C1_p1=2e-4, D1_p1=5e-5, C2_p1=3e-5, D2_p1=1e-5)), model_kwargs=LTT, ops=NO_AT_T)
lc('lc_lambert_light_travel', with_changes(BASE, aglambert_p1=0.3), model_kwargs=LTT, ops=NO_AT_T)

# ---------------------------------------------------------------- catwoman
CATWOMAN = with_changes(BASE, remove=('p_p1',), p1_p1=0.09, p2_p1=0.11, phi_p1=30.)
lc('lc_catwoman', CATWOMAN, requires='requires_catwoman')
lc('lc_catwoman_rho', with_changes(CATWOMAN, remove=('a_p1',), rho=1500.), requires='requires_catwoman')
lc('lc_catwoman_supersampled', CATWOMAN, requires='requires_catwoman',
   data=lambda: dict(lc_data(), lc_instrument_supersamp=['A'], lc_n_supersamp=[5], lc_exptime_supersamp=[0.02]))

# ---------------------------------------------------------------- kelp
KELP_FILTER = {'kelp_filt_wav': {'A': np.linspace(0.6e-6, 1e-6, 20)}, 'kelp_filt_trans': {'A': np.ones(20)}}
lc('lc_kelp_homogeneous', with_changes(BASE, singlescat_p1=0.5, g_p1=0.2), model_kwargs=LTT, requires='requires_kelp')
lc('lc_kelp_inhomogeneous', with_changes(BASE, w0_p1=0.5, wp_p1=0.5, agkelp_p1=0.3, x1_p1=-30., x2_p1=30.),
   model_kwargs=LTT, requires='requires_kelp')
lc('lc_kelp_inhomogeneous_morris', with_changes(BASE, w0_p1=0.5, wp_p1=0.5, agkelp_p1=0.3, x1prime_p1=-0.5,
                                                x2prime_p1=0.5), model_kwargs=LTT, requires='requires_kelp')
lc('lc_kelp_thermal', with_changes(BASE, hotspotoff_p1=20., wdrag_p1=4.5, alpha_p1=0.6, cml11_p1=0.1, fprime_p1=0.7),
   model_kwargs=dict(LTT, stellar_teff=5500., **KELP_FILTER), requires='requires_kelp')


# ---------------------------------------------------------------- Supersampling
def _supersampled(instruments=('A',)):
    return dict(lc_data(instruments), lc_instrument_supersamp=['A'], lc_n_supersamp=[5], lc_exptime_supersamp=[0.02])


lc('lc_supersampled', BASE, data=_supersampled)
lc('lc_supersampled_transit_eclipse', TRANSIT_ECLIPSE, data=_supersampled)
lc('lc_supersampled_one_of_two', transit_parameters(TWO_INSTRUMENTS), data=lambda: _supersampled(TWO_INSTRUMENTS))

# ---------------------------------------------------------------- TTVs (single planet; see below for two planets)
lc('lc_ttv_dt', with_changes(BASE, **{'dt_p1_A_0': 0.002, 'dt_p1_A_1': -0.003, 'dt_p1_A_2': 0.001}), ops=NO_AT_T)
T_TTV = with_changes(BASE, remove=('t0_p1', 'P_p1'), **{'T_p1_A_0': 2458460.502, 'T_p1_A_1': 2458462.197,
                                                        'T_p1_A_2': 2458463.901})
lc('lc_ttv_T', T_TTV, ops=NO_AT_T, extra_samples={'P_p1': 1.7, 't0_p1': 2458460.5})

# ---------------------------------------------------------------- Several instruments and planets
lc('lc_two_instruments', transit_parameters(TWO_INSTRUMENTS), instruments=TWO_INSTRUMENTS)
lc('lc_three_instruments', transit_parameters(THREE_INSTRUMENTS), instruments=THREE_INSTRUMENTS)
lc('lc_two_planets', transit_parameters(planets=(1, 2)))
lc('lc_two_planets_two_instruments', transit_parameters(TWO_INSTRUMENTS, planets=(1, 2)), instruments=TWO_INSTRUMENTS)
lc('lc_two_planets_mixed_ecc', with_changes(transit_parameters(planets=(1, 2)), remove=('ecc_p2', 'omega_p2'),
                                            ecosomega_p2=0.1, esinomega_p2=0.05))
lc('lc_one_instrument_no_transit', with_changes(transit_parameters(TWO_INSTRUMENTS), remove=('q1_B', 'q2_B')),
   instruments=TWO_INSTRUMENTS, ops=NO_COMPONENTS)

_TWO = transit_parameters(TWO_INSTRUMENTS)
lc('lc_shared_ld', with_changes(_TWO, remove=('q1_A', 'q2_A', 'q1_B', 'q2_B'), q1_A_B=0.3, q2_A_B=0.35),
   instruments=TWO_INSTRUMENTS)
lc('lc_shared_dilution', with_changes(_TWO, remove=('mdilution_A', 'mdilution_B'), mdilution_A_B=0.95),
   instruments=TWO_INSTRUMENTS)
lc('lc_shared_mflux', with_changes(_TWO, remove=('mflux_A', 'mflux_B'), mflux_A_B=1e-5), instruments=TWO_INSTRUMENTS)
lc('lc_shared_jitter', with_changes(_TWO, remove=('sigma_w_A', 'sigma_w_B'), sigma_w_A_B=60.),
   instruments=TWO_INSTRUMENTS)
lc('lc_per_instrument_depth', with_changes(_TWO, remove=('p_p1',), p_p1_A=0.1, p_p1_B=0.11),
   instruments=TWO_INSTRUMENTS)
lc('lc_per_instrument_eclipse_depth',
   with_changes(_TWO, fp_p1_A=4e-4, fp_p1_B=6e-4, t_secondary_p1=2458460.5 + 0.85), instruments=TWO_INSTRUMENTS)
lc('lc_shared_eclipse_depth', with_changes(_TWO, fp_p1_A_B=4e-4, t_secondary_p1=2458460.5 + 0.85),
   instruments=TWO_INSTRUMENTS)


# ---------------------------------------------------------------- Linear and non-linear terms
def linear_regressors(times):
    return np.column_stack([np.sin(times - times[0]), np.cos(2. * (times - times[0]))])


def _linear_lc():
    return dict(lc_data(), linear_regressors_lc={'A': linear_regressors(lc_times(0))})


def additive_nonlinear(X, parameter_values):
    return parameter_values['nl_amp_A'] * np.exp(-(X - X[0]) / parameter_values['nl_tau_A'])


def multiplicative_nonlinear(X, parameter_values):
    return 1. + parameter_values['nl_amp_A'] * np.exp(-(X - X[0]) / parameter_values['nl_tau_A'])


def _nonlinear_lc(function, multiplicative=None):
    entry = {'function': function, 'regressor': lc_times(0)}
    if multiplicative is not None:
        entry['multiplicative'] = multiplicative
    return dict(lc_data(), non_linear_functions={'A': entry})


lc('lc_linear_model', with_changes(BASE, theta0_A=1e-3, theta1_A=-5e-4), data=_linear_lc,
   ops=(PLAIN, COMPONENTS, AT_T, TRANSIT))
_NL = with_changes(BASE, nl_amp_A=2e-3, nl_tau_A=0.5)
lc('lc_nonlinear_additive', _NL, data=lambda: _nonlinear_lc(additive_nonlinear), ops=NO_AT_T)
lc('lc_nonlinear_additive_explicit', _NL, data=lambda: _nonlinear_lc(additive_nonlinear, False), ops=(PLAIN,))
lc('lc_nonlinear_multiplicative', _NL, data=lambda: _nonlinear_lc(multiplicative_nonlinear, True), ops=NO_AT_T)

# ---------------------------------------------------------------- RVs
RV = rv_parameters()
rv('rv_one_planet', RV)
rv('rv_fixed_priors', RV, fixed=('ecc_p1', 'omega_p1'))
rv('rv_ecc_omega', with_changes(RV, ecc_p1=0.2, omega_p1=40.))
rv('rv_ecosomega', with_changes(RV, remove=ORBIT_REMOVE, ecosomega_p1=0.15, esinomega_p1=0.1))
rv('rv_secosomega', with_changes(RV, remove=ORBIT_REMOVE, secosomega_p1=0.3, sesinomega_p1=0.25))
rv('rv_two_planets', rv_parameters(planets=(1, 2)))
rv('rv_two_instruments', rv_parameters(TWO_INSTRUMENTS), instruments=TWO_INSTRUMENTS, ops=NO_COMPONENTS)
rv('rv_three_instruments', rv_parameters(THREE_INSTRUMENTS), instruments=THREE_INSTRUMENTS, ops=NO_COMPONENTS)
rv('rv_two_planets_two_instruments', rv_parameters(TWO_INSTRUMENTS, planets=(1, 2)), instruments=TWO_INSTRUMENTS,
   ops=NO_COMPONENTS)
rv('rv_trend_line', with_changes(RV, rv_intercept=1.5, rv_slope=0.02))
rv('rv_trend_quadratic', with_changes(RV, rv_intercept=1.5, rv_slope=0.02, rv_quad=-1e-3))
rv('rv_trend_custom_ta', with_changes(RV, rv_intercept=1.5, rv_slope=0.02), model_kwargs={'ta': 2458000.})
rv('rv_linear_model', with_changes(RV, theta0_A=0.5, theta1_A=-0.3),
   data=lambda: dict(rv_data(), linear_regressors_rv={'A': linear_regressors(rv_data()['t_rv']['A'])}),
   ops=NO_AT_T)


# ---------------------------------------------------------------- Joint photometry and RVs
def _joint(instruments_lc=('A',), instruments_rv=('R',)):
    return dict(lc_data(instruments_lc), **rv_data(instruments_rv))


_JOINT = with_changes(transit_parameters(), P_p1=10., t0_p1=2458461., K_p1=5., mu_R=10., sigma_w_R=0.5)
CONFIGS.append(Config('joint_one_planet', ('lc', 'rv'), _joint, _JOINT))
CONFIGS.append(Config('joint_two_instruments', ('lc', 'rv'), lambda: _joint(TWO_INSTRUMENTS, ('R', 'S')),
                      with_changes(transit_parameters(TWO_INSTRUMENTS), P_p1=10., t0_p1=2458461., K_p1=5.,
                                   mu_R=10., sigma_w_R=0.5, mu_S=-5., sigma_w_S=0.7), ops=NO_COMPONENTS))


# ---------------------------------------------------------------- GP kernels: per-instrument and global, LC and RV
def _kernel_data(kind, kernel, mode, george_hodlr=True):
    if kind == 'lc':
        data = lc_data(GP_INSTRUMENTS, n=50)
        times, key = data['t_lc'], 'GP_regressors_lc'
    else:
        data = rv_data(GP_INSTRUMENTS)
        times, key = data['t_rv'], 'GP_regressors_rv'
    if mode == 'global':
        data[key] = {kind: stack(times, GP_INSTRUMENTS)}
    else:
        data[key] = {name: times[name] for name in GP_INSTRUMENTS}
    data['george_hodlr'] = george_hodlr
    return data


def _kernel_params(kind, kernel, mode):
    params = (instrument_parameters(GP_INSTRUMENTS) if kind == 'lc' else rv_parameters(GP_INSTRUMENTS))
    if mode == 'global':
        params.update(gp_parameters(kernel, kind, kind))
    else:
        for name in GP_INSTRUMENTS:
            params.update(gp_parameters(kernel, kind, name))
    return params


def _instrument_ops(kind):
    return PER_INSTRUMENT_OPS if kind == 'lc' else NO_COMPONENTS


def _add_kernel_configs():
    for kernel in KERNELS:
        for kind in ('lc', 'rv'):
            for mode in ('instrument', 'global'):
                if mode == 'instrument':
                    ops = _instrument_ops(kind)
                else:
                    # Global RV only works through the t= route, and global LC not at all. See tests/contract.
                    ops = (AT_T,) if kind == 'rv' else ()
                params = _kernel_params(kind, kernel, mode)
                CONFIGS.append(Config('gp_{}_{}_{}'.format(kind, mode, kernel), (kind,),
                                      lambda kind=kind, kernel=kernel, mode=mode: _kernel_data(kind, kernel, mode),
                                      params, ops=ops))


_add_kernel_configs()


# Other GP options, one dimension at a time:
def _exact_solver(kind, kernel, mode):
    return lambda: _kernel_data(kind, kernel, mode, george_hodlr=False)


for _kind in ('lc', 'rv'):
    for _mode, _ops in (('instrument', _instrument_ops(_kind)), ('global', (AT_T,) if _kind == 'rv' else ())):
        CONFIGS.append(Config('gp_{}_{}_SEKernel_exact_solver'.format(_kind, _mode), (_kind,),
                              _exact_solver(_kind, 'SEKernel', _mode), _kernel_params(_kind, 'SEKernel', _mode),
                              ops=_ops))


def _matern_eps():
    return dict(_kernel_data('rv', 'CeleriteMaternKernel', 'instrument'), matern_eps=0.05)


CONFIGS.append(Config('gp_rv_instrument_matern_eps', ('rv',), _matern_eps,
                      _kernel_params('rv', 'CeleriteMaternKernel', 'instrument'), ops=NO_COMPONENTS))


def _two_column_gp(kind, mode):
    data = lc_data(GP_INSTRUMENTS, n=50) if kind == 'lc' else rv_data(GP_INSTRUMENTS)
    times = data['t_' + kind]
    columns = {name: np.column_stack([times[name], np.sin(times[name])]) for name in GP_INSTRUMENTS}
    if mode == 'global':
        data['GP_regressors_' + kind] = {kind: np.concatenate([columns[name] for name in GP_INSTRUMENTS])}
    else:
        data['GP_regressors_' + kind] = columns
    return data


for _kernel, _alpha in (('SEKernel', 'alpha'), ('M32Kernel', 'malpha')):
    for _kind in ('lc', 'rv'):
        for _mode, _ops in (('instrument', _instrument_ops(_kind)), ('global', (AT_T,) if _kind == 'rv' else ())):
            _params = instrument_parameters(GP_INSTRUMENTS) if _kind == 'lc' else rv_parameters(GP_INSTRUMENTS)
            for _suffix in ([_kind] if _mode == 'global' else GP_INSTRUMENTS):
                _params.update(gp_parameters(_kernel, _kind, _suffix))
                _params['GP_{}1_{}'.format(_alpha, _suffix)] = 0.7
            CONFIGS.append(Config('gp_{}_{}_{}_two_regressors'.format(_kind, _mode, _kernel), (_kind,),
                                  lambda kind=_kind, mode=_mode: _two_column_gp(kind, mode), _params, ops=_ops))


# A GP with transit: the usual science case, photometry and RVs, per instrument and global.
def _gp_on_transit(mode):
    data = _kernel_data('lc', 'CeleriteMaternKernel', mode)
    return data


for _mode, _ops in (('instrument', (PLAIN, COMPONENTS, AT_T, TRANSIT)), ('global', ())):
    _params = transit_parameters(GP_INSTRUMENTS)
    for _suffix in (['lc'] if _mode == 'global' else GP_INSTRUMENTS):
        _params.update(gp_parameters('CeleriteMaternKernel', 'lc', _suffix))
    CONFIGS.append(Config('gp_lc_{}_on_transit'.format(_mode), ('lc',), lambda mode=_mode: _gp_on_transit(mode),
                          _params, ops=_ops))


def _gp_on_planet(mode):
    return _kernel_data('rv', 'CeleriteMaternKernel', mode)


for _mode, _ops in (('instrument', NO_COMPONENTS), ('global', (AT_T,))):
    _params = rv_parameters(GP_INSTRUMENTS, planets=(1, 2))
    for _suffix in (['rv'] if _mode == 'global' else GP_INSTRUMENTS):
        _params.update(gp_parameters('CeleriteMaternKernel', 'rv', _suffix))
    CONFIGS.append(Config('gp_rv_{}_on_planets'.format(_mode), ('rv',), lambda mode=_mode: _gp_on_planet(mode),
                          _params, ops=_ops))

# A GP shared between some instruments by name, without a global GP:
_shared = instrument_parameters(GP_INSTRUMENTS)
_shared.update(gp_parameters('CeleriteMaternKernel', 'lc', 'X_Y'))
CONFIGS.append(Config('gp_lc_shared_hyperparameters', ('lc',),
                      lambda: _kernel_data('lc', 'CeleriteMaternKernel', 'instrument'), _shared))

# ---------------------------------------------------------------- The paper's data, at the paper's values
for _problem in PAPER:
    CONFIGS.append(Config('paper_' + _problem.replace('-', '_'), ('lc', 'rv'), lambda problem=_problem: data_arguments(problem),
                          paper_parameters(_problem), ops=(), loglike_only=True))

BY_ID = {config.id: config for config in CONFIGS}
if len(BY_ID) != len(CONFIGS):
    raise ValueError('Duplicate configuration ids in tests/golden/configs.py')
