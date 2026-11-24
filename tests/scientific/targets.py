"""
The K2-140 and K2-32 analyses of Espinoza et al. (2019, arXiv:1812.08549), from the paper's PDF: Tables 8 and 10.

Each row is parameter: (median, upper error, lower error) in juliet's units, which differ from the tables' in places:
the RVs in tutorial1/ are in km/s, and M is relative flux rather than ppm.

Left out on purpose:
- Derived parameters, and the GP hyperparameters of K2-32. Those are passed to celerite in units that are still being
  checked against what Table 10 says (section 6 of claude_refactoring.md).
- Table 9's priors for K2-32, as its P and t0 rows are a copy of K2-140's. The prior files in tutorial2/ are used instead.
"""
from pathlib import Path

ROOT = Path(__file__).parents[2]

# The Kepler long-cadence exposure time. The paper says that K2 integrations were accounted for, without giving these
# two numbers, so this is an assumption of this test:
K2_LONG_CADENCE = {'lc_instrument_supersamp': ['K2'], 'lc_n_supersamp': [20], 'lc_exptime_supersamp': [0.020434]}

PROBLEMS = {
    'k2-140': dict(priors=str(ROOT / 'tutorial1/priors/K2-140_priors_fullcircular.dat'),
                   lcfilename=str(ROOT / 'tutorial1/data/K2-140_lc.dat'),
                   rvfilename=str(ROOT / 'tutorial1/data/K2-140_rvs.dat'),
                   ld_laws='K2-quadratic,LCOGT-linear', **K2_LONG_CADENCE),
    'k2-32': dict(priors=str(ROOT / 'tutorial2/priors/K2-32_priors.dat'),
                  lcfilename=str(ROOT / 'tutorial2/data/K2-32_lc.dat'),
                  rvfilename=str(ROOT / 'tutorial2/data/K2-32_rvs.dat'),
                  GPlceparamfile=str(ROOT / 'tutorial2/data/K2-32_lceparams.dat'), **K2_LONG_CADENCE),
}

PAPER = {
    # Table 8. RVs in km/s.
    'k2-140': {
        'P_p1': (6.569298, 0.000026, 0.000027), 't0_p1': (2457588.28381, 0.00024, 0.00024),
        'a_p1': (15.24, 0.13, 0.16), 'r1_p1': (0.402, 0.054, 0.047), 'r2_p1': (0.11437, 0.00083, 0.00087),
        'K_p1': (0.1038, 0.0048, 0.0045),
        'mflux_K2': (-35.5e-6, 8.4e-6, 8.0e-6), 'sigma_w_K2': (381.7, 5.7, 5.7),
        'q1_K2': (0.213, 0.127, 0.075), 'q2_K2': (0.57, 0.22, 0.18),
        'mflux_LCOGT': (230e-6, 230e-6, 220e-6), 'sigma_w_LCOGT': (18.8, 122.5, 16.3), 'q1_LCOGT': (0.586, 0.064, 0.065),
        'mu_CORALIE': (1.2149, 0.0075, 0.0079), 'sigma_w_CORALIE': (0.00116, 0.0053, 0.0009),
        'mu_HARPS': (1.2466, 0.0073, 0.0080), 'sigma_w_HARPS': (0.0135, 0.0103, 0.0093),
        'mu_FIES': (1.1318, 0.0036, 0.0036), 'sigma_w_FIES': (0.0016, 0.0067, 0.0014),
    },
    # Table 10. RVs in m/s.
    'k2-32': {
        'rho': (2122.6, 56.9, 62.3),
        'P_p1': (8.991854, 0.000089, 0.000093), 't0_p1': (2456909.91797, 0.00031, 0.00028),
        'r1_p1': (0.421, 0.049, 0.051), 'r2_p1': (0.05496, 0.00051, 0.00054), 'K_p1': (5.63, 0.78, 0.76),
        'P_p2': (20.65974, 0.00087, 0.00085), 't0_p2': (2456961.4058, 0.0017, 0.0016),
        'r1_p2': (0.538, 0.032, 0.037), 'r2_p2': (0.03108, 0.00054, 0.00056), 'K_p2': (1.68, 0.83, 0.82),
        'P_p3': (31.7143, 0.0013, 0.0011), 't0_p3': (2456903.7905, 0.0016, 0.0018),
        'r1_p3': (0.63156, 0.02001, 0.02124), 'r2_p3': (0.03673, 0.00059, 0.00064), 'K_p3': (2.43, 0.94, 0.87),
        'mflux_K2': (-3500e-6, 860e-6, 870e-6), 'sigma_w_K2': (83.6, 1.1, 1.1),
        'q1_K2': (0.334, 0.155, 0.104), 'q2_K2': (0.61, 0.19, 0.16),
        'mu_HIRES': (-1.722, 0.714, 0.705), 'sigma_w_HIRES': (3.65, 0.65, 0.56),
        'mu_HARPS': (1.092, 0.701, 0.706), 'sigma_w_HARPS': (3.92, 0.67, 0.58),
        'mu_PFS': (-6.5, 2.0, 2.0), 'sigma_w_PFS': (4.6, 2.2, 1.8),
    },
}

# Table 10's GP values, as printed, to freeze a likelihood on real data. Whether T is a time scale or a rate is open.
K2_32_GP_AS_PRINTED = {'GP_sigma_K2': 10.46, 'GP_timescale_K2': 16647., 'GP_rho_K2': 49.}
FIXED = {
    'k2-140': {'ecc_p1': 0., 'omega_p1': 90., 'mdilution_K2': 1., 'mdilution_LCOGT': 1.},
    'k2-32': {'ecc_p1': 0., 'omega_p1': 90., 'ecc_p2': 0., 'omega_p2': 90., 'ecc_p3': 0., 'omega_p3': 90.,
              'mdilution_K2': 1.},
}


def paper_parameters(problem):
    """The paper's medians, plus the fixed values and (K2-32) the GP, as a parameter dictionary."""
    params = {name: row[0] for name, row in PAPER[problem].items()}
    params.update(FIXED[problem])
    if problem == 'k2-32':
        params.update(K2_32_GP_AS_PRINTED)
    return params


def data_arguments(problem):
    return {key: value for key, value in PROBLEMS[problem].items() if key != 'priors'}
