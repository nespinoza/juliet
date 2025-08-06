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

E, C1, D1 = 379e-6, 127e-6, -139e-6
C2, D2 = 46e-6, -15e-6

sigma = 200e-6

# Simulated time
times = np.linspace(tc-(0.55*per), tc+(0.55*per), 10000)

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

m1 = batman.TransitModel(pars, times)
flx_tra = m1.light_curve(pars)

# Batman occultation model
pars.fp = E
pars.t_secondary = tsecondary
m2 = batman.TransitModel(params=pars, t=times, transittype='secondary')
flx_ecl = m2.light_curve(pars)

# Cowan & Agol (2008) phase curve model
omega_t = 2 * np.pi * (times - tsecondary) / per
pc_CA08 = E + ( C1 * (np.cos( omega_t ) - 1.) ) + ( D1 * np.sin( omega_t ) ) + ( C2 * (np.cos( 2*omega_t ) - 1.) ) + ( D2 * np.sin( 2*omega_t ) )
sine_model = 1. + pc_CA08 * ((flx_ecl - 1.) / E)

# Total model
total_model = flx_tra * sine_model

# ------------------------------------------------------
# Simulated data
tim7 = times
fl7 = total_model + np.random.normal(0., sigma, len(times))
fle7 = np.ones(len(times))*sigma*0.1

# ---------------------------------------------------------------------
#
#                    And now, the fitting!
#
# ---------------------------------------------------------------------
instrument = 'MIRI'
pout = 'cowan_agol_08_pc_sim'

# ------------- Full dataset
## Making the dataset: Full Dataset
tim, fl, fle = {}, {}, {}
tim[instrument], fl[instrument], fle[instrument] = tim7, fl7, fle7

# And priors
## Planetary priors
par_P = ['P_p1', 't0_p1', 'p_p1', 'b_p1', 'q1_' + instrument, 'q2_' + instrument, 'ecc_p1', 'omega_p1', 'a_p1']
dist_P = ['fixed', 'fixed', 'uniform', 'uniform', 'uniform', 'uniform', 'fixed', 'fixed', 'normal']
hyper_P = [per, tc, [0., 1.], [0., 1.], [0., 1.], [0., 1.], 0., 90., [ar, ar_err]]

par_pc = ['fp_p1', 'C1_p1', 'D1_p1', 'C2_p1', 'D2_p1']
dist_pc = ['uniform', 'uniform', 'uniform', 'uniform', 'uniform']
hyper_pc = [[0., 500.e-6], [-1., 1.], [-1., 1.], [-1., 1.], [-1., 1.]]

## Instrumental priors
par_ins = ['mdilution_' + instrument, 'mflux_' + instrument, 'sigma_w_' + instrument]
dist_ins = ['fixed', 'normal', 'loguniform']
hyper_ins = [1., [0., 0.1], [0.1, 10000.]]

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

# Let's plot some cool results!
model = res.lc.evaluate(instrument)

# Let's make sure that it works:
fig = plt.figure(figsize=(16,9))
gs = gd.GridSpec(2,1, height_ratios=[2,1])

# Top panel
ax1 = plt.subplot(gs[0])
ax1.errorbar(tim[instrument], fl[instrument], yerr=fle[instrument], fmt='.', alpha=0.3)
ax1.plot(tim[instrument], model, c='k', zorder=100, label='Fitted model')
ax1.plot(times, total_model, c='r', lw=2.5, alpha=0.5, zorder=50, label='Ingested model')
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
plt.savefig(pout + '/full_model.png')

# ----------------------------------------------------
#  Corner plot
# ----------------------------------------------------
post1 = res.posteriors['posterior_samples']
q1, q2 = juliet.utils.convert_ld_coeffs('quadratic', 0.1, 0.3)

data = np.vstack([ post1['p_p1'], post1['b_p1'], post1['q1_' + instrument], post1['q2_' + instrument],\
                  post1['a_p1'], post1['fp_p1'], post1['C1_p1'], post1['D1_p1'], post1['C2_p1'], post1['D2_p1'] ])
data = np.transpose(data)
lbls = np.array([ 'Rp/R*', 'b', 'q1', 'q2', 'a/R*', 'fp', 'C1', 'D1', 'C2', 'D2' ])
truths = np.array([ rprs, bb, q1, q2, ar, E, C1, D1, C2, D2])

fig = corner.corner(data, labels=lbls, show_titles=True, truths=truths)
plt.savefig(pout + '/corner.png')