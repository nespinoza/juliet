import numpy as np
import matplotlib.pyplot as plt
import batman
import juliet
import os
import corner
import matplotlib.gridspec as gd
import multiprocessing
multiprocessing.set_start_method('fork')

np.random.seed(3)

# Lambertian + Sine phase curve model simulation for 2 instrument
# Set this variable to True if you want to fit the same phase curve model to both instruments
# Set this variable to False if you want to fit the diff phase curve model to both instruments
# For single instrument, simply put name of only one instrument in instruments array
SamePC_2Inst = False
instruments = ['CHEOPS1', 'CHEOPS2']
pout = os.getcwd() + '/juliet/juls/LambertSin_2Inst_DiffPC'
if SamePC_2Inst:
    ## Instrument 1 and 2 both have same phase curve parameters (Only geometric albedo for Lambertian fitting)
    Ags, fps = [0.15, 0.15], [350e-6, 350e-6]
    ph_off = [60., 60.]
else:
    ## Instrument 1 and 2 have different phase curve parameters
    Ags, fps = [0.15, 0.25], [350e-6, 450e-6]
    ph_off = [50., 60.]


# ---------------------------------------------------------------------
#
#                    Generating simulated data
#
# ---------------------------------------------------------------------

# We will simulate WASP-103 b phase curve with planetary and phase curve parameters 
# from 	Gillon et al. 2014 (https://ui.adsabs.harvard.edu/abs/2014A%26A...562L...3G/abstract)

per, per_err = 0.925542, 0.000019
tc, tc_err = 2456459.59957, 0.00075
bb, bb_err = 0.19, 0.13
inc, inc_err = 86.3, 2.7
rprs, rprs_err = 0.1093, np.sqrt(0.0019**2 + 0.0017**2)
rho, rho_err = 0.583*1e3, np.sqrt(0.030**2 + 0.055**2) * 1e3
rst = 1.436
ar, ar_err = 2.978, np.sqrt(0.050**2 + 0.096**2)

q1, q2 = juliet.utils.convert_ld_coeffs('quadratic', 0.1, 0.3)

tsecondary = tc + (per/2)
sigma = 100e-6

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
    times = np.linspace(tc-(0.65*per), tc+(0.65*per), 10000) + ( ((ins+1)**3) * per)

    m1 = batman.TransitModel(pars, times)
    flx_tra = m1.light_curve(pars)

    # Batman occultation model
    pars.fp = 100e-6
    pars.t_secondary = tsecondary
    m2 = batman.TransitModel(params=pars, t=times, transittype='secondary')
    flx_ecl = m2.light_curve(pars)
    
    # ------------------------------------------------------------------------------
    # Lambertian phase curve model (Deline et al. 2022; see their section 4.4.3)
    # ------------------------------------------------------------------------------
    ## First find true anomaly and other angles
    true_anomaly = m2.get_true_anomaly()
    argument_peri = np.radians( m2.w )
    inclination = np.radians( m2.inc )

    ## Now, computing alpha
    alpha = np.arccos( -np.sin( argument_peri + true_anomaly ) * np.sin( inclination ) )

    ## Now, the phase curve model
    ecc_fac = ( 1 + m2.ecc * np.cos(true_anomaly) ) / ( 1 - m2.ecc**2 )
    term1 = ( m2.rp * ecc_fac / m2.a )**2
    term2 = ( np.sin(alpha) + (np.pi - alpha)*np.cos(alpha) ) / np.pi
    lambert_model = Ags[ins] * term1 * term2

    # ------------------------------------------------------------------------------
    # Sinusoidal phase curve
    # ------------------------------------------------------------------------------

    orbital_phase = ( ( ( times - tc ) / per ) % 1 )
    center_phase = - np.pi / 2.

    # Build model. First, the basis sine function:
    sine_model = np.sin(2. * np.pi * (orbital_phase) + center_phase + ph_off[ins] * (np.pi / 180.) )
    # Scale to be 1 at secondary eclipse, 0 at transit:
    sine_model = (sine_model + 1) * 0.5
    # Amplify by phase-amplitude:
    sine_model = (fps[ins]) * sine_model
    
    # ------------------------------------------------------------------------------
    # Total phase curve
    # ------------------------------------------------------------------------------

    phase_curve_model = lambert_model + sine_model
    phase_curve_model = 1. + phase_curve_model * ((flx_ecl - 1.) / m2.fp)

    # Total model
    total_model = flx_tra * phase_curve_model
    total_models[instruments[ins]] = total_model

    # ------------------------------------------------------
    # Simulated data
    tim7 = times
    fl7 = total_model + np.random.normal(0., sigma, len(times))
    fle7 = np.ones(len(times))*sigma*0.1

    # ------------- Full dataset
    ## Saving the dataset
    tim[instruments[ins]], fl[instruments[ins]], fle[instruments[ins]] = tim7, fl7, fle7

    """plt.errorbar(tim7, fl7, yerr=fle7, fmt='.')
    plt.plot(tim7, 1. + lambert_model * ((flx_ecl - 1.) / m2.fp), 'k-', zorder=10)
    plt.plot(tim7, 1. + sine_model * ((flx_ecl - 1.) / m2.fp), 'r-', zorder=10)
    plt.plot(tim7, total_model, 'b-', zorder=10)
    
    plt.show()

"""

    
# ---------------------------------------------------------------------
#
#                    And now, the fitting!
#
# ---------------------------------------------------------------------

# And priors
## Planetary priors
par_P = ['P_p1', 't0_p1', 'p_p1', 'b_p1', 'q1_' + '_'.join(instruments), 'q2_' + '_'.join(instruments), 'ecc_p1', 'omega_p1', 'a_p1']
dist_P = ['fixed', 'fixed', 'uniform', 'uniform', 'uniform', 'uniform', 'fixed', 'fixed', 'normal']
hyper_P = [per, tc, [0., 0.15], [0.1, 0.3], [0., 1.], [0., 1.], 0., 90., [ar, ar_err]]

if SamePC_2Inst:
    par_pc = ['aglambert_p1', 'fp_p1', 'phaseoffset_p1']
    dist_pc = ['uniform', 'uniform', 'uniform']
    hyper_pc = [[0., 0.5], [0e-6, 600e-6], [0., 80.]]
else:
    par_pc, dist_pc, hyper_pc = [], [], []
    for ins in range(len(instruments)):
        par_pc = par_pc + ['aglambert_p1_' + instruments[ins], 'fp_p1_' + instruments[ins], 'phaseoffset_p1_' + instruments[ins]]
        dist_pc = dist_pc + ['uniform', 'uniform', 'uniform']
        hyper_pc = hyper_pc + [[0., 0.5], [0.e-6, 600e-6], [0., 80.]]


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
    ax1.plot(tim[instrument], total_models[instrument], c='r', lw=3.5, alpha=0.5, zorder=50, label='Ingested model')
    ax1.set_ylabel('Relative Flux')
    ax1.set_xlim(np.min(tim[instrument]), np.max(tim[instrument]))
    ax1.set_ylim([1-300e-6, 1+500e-6])
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

if SamePC_2Inst:
    data = np.vstack([ post1['p_p1'], post1['b_p1'], post1['q1_' + '_'.join(instruments)], post1['q2_' + '_'.join(instruments)],\
                    post1['a_p1'], post1['aglambert_p1'], post1['fp_p1'], post1['phaseoffset_p1'] ])
    data = np.transpose(data)
    lbls = np.array([ 'Rp/R*', 'b', 'q1', 'q2', 'a/R*', 'Ag', 'fp', 'PhOff'])
    truths = np.array([ rprs, bb, q1, q2, ar, Ags[0], fps[0], ph_off[0] ])
else:
    data = np.vstack([ post1['p_p1'], post1['b_p1'], post1['q1_' + '_'.join(instruments)], post1['q2_' + '_'.join(instruments)], post1['a_p1'],\
                       post1['aglambert_p1_' + instruments[0]], post1['aglambert_p1_' + instruments[1]],\
                       post1['fp_p1_' + instruments[0]], post1['fp_p1_' + instruments[1]],\
                       post1['phaseoffset_p1_' + instruments[0]], post1['phaseoffset_p1_' + instruments[1]] ])
    data = np.transpose(data)
    lbls = np.array([ 'Rp/R*', 'b', 'q1', 'q2', 'a/R*', 'Ag0', 'Ag1', 'fp0', 'fp1', 'PhOff0', 'PhOff1' ])
    truths = np.array([ rprs, bb, q1, q2, ar, Ags[0], Ags[1], fps[0], fps[1], ph_off[0], ph_off[1] ])

fig = corner.corner(data, labels=lbls, show_titles=True, truths=truths)
plt.savefig(pout + '/corner.png')#"""