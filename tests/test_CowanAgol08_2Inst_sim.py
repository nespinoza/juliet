import numpy as np
import matplotlib.pyplot as plt
import batman
import juliet
import os
import corner
import matplotlib.gridspec as gd
import multiprocessing
multiprocessing.set_start_method('fork')

np.random.seed(5)

# Cowan & Agol (2008) phase curve model simulation for 2 instrument
# Set this variable to True if you want to fit the same phase curve model to both instruments
# Set this variable to False if you want to fit the diff phase curve model to both instruments
SamePC_2Inst = False
instruments = ['MIRI1', 'MIRI2']
pout = os.getcwd() + '/juliet/juls/cowan_agol_08_pc_sim_DiffPC_2Inst'
if SamePC_2Inst:
    ## Instrument 1 and 2 both have same phase curve parameters
    Es, C1s, D1s = [379e-6, 379e-6], [127e-6, 127e-6], [-139e-6, -139e-6]
    C2s, D2s = [46e-6, 46e-6], [-15e-6, -15e-6]
else:
    ## Instrument 1 and 2 have different phase curve parameters
    Es, C1s, D1s = [379e-6, 300e-6], [127e-6, 125e-6], [-139e-6, -109e-6]
    C2s, D2s = [46e-6, 25e-6], [-15e-6, -35e-6]


# ---------------------------------------------------------------------
#
#                    Generating simulated data
#
# ---------------------------------------------------------------------

# We will simulate GJ-1214 b phase curve with planetary and phase curve parameters 
# from Kempton et al. 2023 (https://ui.adsabs.harvard.edu/abs/2023Natur.620...67K/abstract)
# and Mahajan et al. 2024 (https://ui.adsabs.harvard.edu/abs/2024ApJ...963L..37M/abstract)
# (Phase curve parameters are from the former paper and the planetary parameters are from the latter)
# For simplicity we will assume circular orbit

per, per_err = 1.580404531, np.sqrt(0.000000018**2 + 0.000000017**2)
tc, tc_err = 2459639.7812619, np.sqrt(0.0000090**2 + 0.0000089**2)
bb, bb_err = 0.264, np.sqrt(0.020**2 + 0.023**2)
inc, inc_err = 88.980, np.sqrt(0.094**2 + 0.085**2)
rprs, rprs_err = 0.11589, 0.00016
rho, rho_err = 25.41*1e3, np.sqrt(0.62**2 + 0.71**2) * 1e3
rst = 0.2162
ar, ar_err = 14.97, np.sqrt(0.12**2 + 0.14**2)

tsecondary = tc + (per/2)
sigma = 200e-6

# Batman transit model
pars = batman.TransitParams()
pars.per = per
pars.t0 = tc
pars.a = ar
pars.rp = rprs
pars.inc = np.rad2deg(np.arccos(bb/ar))
pars.ecc = 0.
pars.w = 0.
pars.u = [0.1, 0.3]
pars.limb_dark = 'quadratic'

# ------------- Full dataset
## Making the dataset: Full Dataset
tim, fl, fle = {}, {}, {}
total_models = {}
for ins in range(len(instruments)):
    # Simulated time
    times = np.linspace(tc-(0.55*per), tc+(0.55*per), 10000) + ( ((ins+1)**3) * per)

    m1 = batman.TransitModel(pars, times)
    flx_tra = m1.light_curve(pars)

    # Batman occultation model
    pars.fp = Es[ins]
    pars.t_secondary = tsecondary
    m2 = batman.TransitModel(params=pars, t=times, transittype='secondary')
    flx_ecl = m2.light_curve(pars)

    # Cowan & Agol (2008) phase curve model
    omega_t = 2 * np.pi * (times - tsecondary) / per
    pc_CA08 = Es[ins] + ( C1s[ins] * (np.cos( omega_t ) - 1.) ) + ( D1s[ins] * np.sin( omega_t ) ) + ( C2s[ins] * (np.cos( 2*omega_t ) - 1.) ) + ( D2s[ins] * np.sin( 2*omega_t ) )
    sine_model = 1. + pc_CA08 * ((flx_ecl - 1.) / Es[ins])

    # Total model
    total_model = flx_tra * sine_model
    total_models[instruments[ins]] = total_model

    # ------------------------------------------------------
    # Simulated data
    tim7 = times
    fl7 = total_model + np.random.normal(0., sigma, len(times))
    fle7 = np.ones(len(times))*sigma*0.1

    # ------------- Full dataset
    ## Saving the dataset
    tim[instruments[ins]], fl[instruments[ins]], fle[instruments[ins]] = tim7, fl7, fle7


# ---------------------------------------------------------------------
#
#                    And now, the fitting!
#
# ---------------------------------------------------------------------

# And priors
## Planetary priors
par_P = ['P_p1', 't0_p1', 'p_p1', 'b_p1', 'q1_' + '_'.join(instruments), 'q2_' + '_'.join(instruments), 'ecc_p1', 'omega_p1', 'a_p1']
dist_P = ['fixed', 'fixed', 'uniform', 'uniform', 'uniform', 'uniform', 'fixed', 'fixed', 'normal']
hyper_P = [per, tc, [0., 1.], [0., 1.], [0., 1.], [0., 1.], 0., 90., [ar, ar_err]]

if SamePC_2Inst:
    par_pc = ['fp_p1', 'C1_p1', 'D1_p1', 'C2_p1', 'D2_p1']
    dist_pc = ['uniform', 'uniform', 'uniform', 'uniform', 'uniform']
    hyper_pc = [[0., 500.e-6], [-1., 1.], [-1., 1.], [-1., 1.], [-1., 1.]]
else:
    par_pc, dist_pc, hyper_pc = [], [], []
    for ins in range(len(instruments)):
        par_pc = par_pc + ['fp_p1_' + instruments[ins], 'C1_p1_' + instruments[ins], 'D1_p1_' + instruments[ins], 'C2_p1_' + instruments[ins], 'D2_p1_' + instruments[ins]]
        dist_pc = dist_pc + ['uniform', 'uniform', 'uniform', 'uniform', 'uniform']
        hyper_pc = hyper_pc + [[0., 500.e-6], [-1., 1.], [-1., 1.], [-1., 1.], [-1., 1.]]


## Instrumental priors
par_ins, dist_ins, hyper_ins = [], [], []
for ins in range(len(instruments)):
    par_ins = par_ins + ['mdilution_' + instruments[ins], 'mflux_' + instruments[ins], 'sigma_w_' + instruments[ins]]
    dist_ins = dist_ins + ['fixed', 'normal', 'loguniform']
    hyper_ins = hyper_ins + [1., [0., 0.1], [0.1, 10000.]]

### Total priros
par_tot = par_P + par_pc + par_ins
dist_tot = dist_P + dist_pc + dist_ins
hyper_tot = hyper_P + hyper_pc + hyper_ins

priors_tot = juliet.utils.generate_priors(par_tot, dist_tot, hyper_tot)

# And fitting
dataset = juliet.load(priors=priors_tot, t_lc=tim, y_lc=fl, yerr_lc=fle, out_folder=pout, verbose=True)
res = dataset.fit(sampler = 'dynesty', nthreads=8, light_travel_delay=True, stellar_radius=rst)

# ---------------------------------------------------------------------
#
#                           Some plotting
#
# ---------------------------------------------------------------------

for ins in range(len(instruments)):
    instrument = instruments[ins]
    # Let's plot some cool results!
    model = res.lc.evaluate(instrument)

    # Let's make sure that it works:
    fig = plt.figure(figsize=(16,9))
    gs = gd.GridSpec(2,1, height_ratios=[2,1])

    # Top panel
    ax1 = plt.subplot(gs[0])
    ax1.errorbar(tim[instrument], fl[instrument], yerr=fle[instrument], fmt='.', alpha=0.3)
    ax1.plot(tim[instrument], model, c='k', zorder=100, label='Fitted model')
    ax1.plot(tim[instrument], total_models[instrument], c='r', lw=2.5, alpha=0.5, zorder=50, label='Ingested model')
    ax1.set_ylabel('Relative Flux')
    ax1.set_xlim(np.min(tim[instrument]), np.max(tim[instrument]))
    ax1.xaxis.set_major_formatter(plt.NullFormatter())
    ax1.legend()

    # Bottom panel
    ax2 = plt.subplot(gs[1])
    ax2.errorbar(tim[instrument], (fl[instrument]-model)*1e6, yerr=fle[instrument]*1e6, fmt='.', alpha=0.3)
    ax2.axhline(y=0.0, c='black', ls='--', zorder=100)
    ax2.set_ylabel('Residuals (ppm)')
    ax2.set_xlabel('Time (BJD)')
    ax2.set_xlim(np.min(tim[instrument]), np.max(tim[instrument]))
    #plt.show()
    plt.savefig(pout + '/full_model_' + instrument + '.png')

# ----------------------------------------------------
#  Corner plot
# ----------------------------------------------------
post1 = res.posteriors['posterior_samples']
q1, q2 = juliet.utils.convert_ld_coeffs('quadratic', 0.1, 0.3)

if SamePC_2Inst:
    data = np.vstack([ post1['p_p1'], post1['b_p1'], post1['q1_' + '_'.join(instruments)], post1['q2_' + '_'.join(instruments)],\
                    post1['a_p1'], post1['fp_p1'], post1['C1_p1'], post1['D1_p1'], post1['C2_p1'], post1['D2_p1'] ])
    data = np.transpose(data)
    lbls = np.array([ 'Rp/R*', 'b', 'q1', 'q2', 'a/R*', 'fp', 'C1', 'D1', 'C2', 'D2' ])
    truths = np.array([ rprs, bb, q1, q2, ar, Es[0], C1s[0], D1s[0], C2s[0], D2s[0]])
else:
    data = np.vstack([ post1['p_p1'], post1['b_p1'], post1['q1_' + '_'.join(instruments)], post1['q2_' + '_'.join(instruments)], post1['a_p1'],\
                       post1['fp_p1_' + instruments[0]], post1['C1_p1_' + instruments[0]], post1['D1_p1_' + instruments[0]], post1['C2_p1_' + instruments[0]], post1['D2_p1_' + instruments[0]],\
                       post1['fp_p1_' + instruments[1]], post1['C1_p1_' + instruments[1]], post1['D1_p1_' + instruments[1]], post1['C2_p1_' + instruments[1]], post1['D2_p1_' + instruments[1]] ])
    data = np.transpose(data)
    lbls = np.array([ 'Rp/R*', 'b', 'q1', 'q2', 'a/R*', 'fp0', 'C10', 'D10', 'C20', 'D20', 'fp1', 'C11', 'D11', 'C21', 'D21' ])
    truths = np.array([ rprs, bb, q1, q2, ar, Es[0], C1s[0], D1s[0], C2s[0], D2s[0], Es[1], C1s[1], D1s[1], C2s[1], D2s[1]])

fig = corner.corner(data, labels=lbls, show_titles=True, truths=truths)
plt.savefig(pout + '/corner.png')#"""