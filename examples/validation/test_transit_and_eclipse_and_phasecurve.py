import matplotlib.pyplot as plt
import time
import numpy as np
import random

import juliet
import batman

random.seed(42)

# Create dataset:
def get_eclipse_model(n = 300, fp = 0.01, t0 = 0., per = 1.0):
    params = batman.TransitParams()
    params.t0 = t0   # time of inferior conjunction
    params.per = per  # orbital period (days)
    params.a = 3.6    # semi-major axis (in units of stellar radii)
    params.rp = 0.1   # rp/rs
    params.inc = 90.  # orbital inclination (in degrees)
    params.ecc = 0.   # eccentricity
    params.w = 90.    # longitude of periastron (in degrees) p
    params.limb_dark = 'quadratic' # limb darkening profile to use
    params.u = [0.2,0.3] # limb darkening coefficients
    params.fp = fp
    params.ac = 0.0
    params.t_secondary = params.t0 + params.per*0.5 + params.ac
    print('True secondary time:', params.t_secondary)
    t = np.linspace(t0-0.1,t0+per+0.1,n) + t0 
    tmodel = batman.TransitModel(params, t.astype('float64'), transittype='secondary')
    return t, tmodel.light_curve(params)

def get_transit_model(n = 300, p = 0.1, t0 = 0., per = 1.0):
    params = batman.TransitParams()
    params.t0 = t0   # time of inferior conjunction
    params.per = per  # orbital period (days)
    params.a = 3.6    # semi-major axis (in units of stellar radii)
    params.rp = p   # rp/rs
    params.inc = 90.  # orbital inclination (in degrees)
    params.ecc = 0.   # eccentricity
    params.w = 90.    # longitude of periastron (in degrees) p
    params.limb_dark = 'quadratic' # limb darkening profile to use
    params.u = [0.2,0.3] # limb darkening coefficients
    t = np.linspace(t0-0.1,t0+per+0.1,n) + t0 
    tmodel = batman.TransitModel(params, t.astype('float64'))
    return t, tmodel.light_curve(params)

# Set number of datapoints the lightcurve will have; define dictionaries 
# that will host the data:
per = 0.5
t0 = 0.0
tsec = t0 + per*0.5
n = 1000
times, fluxes, errors = {}, {}, {}

# Get data, add noise, put it in the dictionaries. Note sigma is underestimated:
fp = 0.01
te, emodel = get_eclipse_model(n, fp = fp, per = per, t0 = t0)
#emodel = emodel - fp

tt, tmodel = get_transit_model(n, per = per, t0 = t0)

# Generate sinusoid:
A = fp * 1e6#1000#3.*fp*1e6#100 # in ppm
phase_offset = 0.#50.0 # in degrees

print('Period = {p}, t0 = {t0}'.format(p = per, t0 = t0))
orbital_phase = ( ( ( tt - t0 ) / per ) % 1 )
center_phase = - np.pi / 2. 
#sine_model = 1. + (A*1e-6) * np.sin(2. * np.pi * (orbital_phase) + center_phase + phase_offset * (np.pi / 180.) )

sine_model = np.sin(2. * np.pi * (orbital_phase) + center_phase + phase_offset * (np.pi / 180.) )

# Scale to be 1 at secondary eclipse, 0 at transit:
sine_model = (sine_model + 1) * 0.5

# Amplify by A in ppm:
sine_model = (A*1e-6) * sine_model

# Multiply by normed eclipse model; add one:
sine_model = 1. + sine_model * ((emodel - 1.) / fp)

# Now make way for eclipse model:
#idx = np.where(emodel != 1. + fp) 
#sine_model[idx] = emodel[idx]# / np.mean(sine_model[idx])
# Find points not-in-eclipse:
#idx = np.where(emodel != 1.)

# Re-scale sine model to match eclipse model depth:
planet_model = tmodel * sine_model #(emodel-1.0) * sine_model
#sine_model
#emodel[idx] = ( emodel[idx] / (1. + fp) ) * np.mean(sine_model[idx])
#sine_model[idx] = 0.

times['inst'], tmodel = tt, planet_model

plt.plot(times['inst'], tmodel, 'r.-')
plt.plot(times['inst'], sine_model, 'b-')
plt.plot(times['inst'], emodel, 'g-')

plt.show()

tmodel = planet_model

sigma = 100*1e-6
fluxes['inst'], errors['inst'] = tmodel + np.random.normal(0., sigma, len(tmodel)), np.ones(len(tmodel))*sigma*0.1

# Define priors:
params = ['P_p1','t0_p1', 'b_p1','ecc_p1','omega_p1', 'p_p1', 'q1_inst', 'q2_inst',\
          'a_p1', 'mdilution_inst', 'mflux_inst', 'sigma_w_inst', 'fp_p1', 't_secondary_p1', 'phaseoffset_p1']

# Distributions:
dists = ['fixed','fixed','fixed','fixed','fixed', 'uniform','uniform','uniform',\
         'normal', 'fixed', 'normal', 'loguniform', 'uniform', 'normal', 'fixed']

# Hyperparameters
hyperps = [per, t0, 0.0, 0.0, 90., [0.,0.3], [0., 1.], [0., 1.],\
           [3.6,0.1], 1.0, [0.,0.1], [0.1, 1000.], [0.,0.1], [tsec,0.1], 0.]

# Join priors:
priors = juliet.generate_priors(params,dists,hyperps)

print('Priors:')
print('---------')
print(priors)
# And fit:
sampler = 'multinest'
    
dataset = juliet.load(priors = priors, t_lc = times, y_lc = fluxes, yerr_lc = errors, out_folder = 'transit-and-eclipse-pc-test', verbose = True)
results = dataset.fit(sampler = sampler, progress = True)

print('START \n testing:')
print(results.lc.dictionary['inst']['EclipseFit'])
print(results.lc.posteriors.keys())
plt.hist(results.lc.posteriors['t_secondary_p1'], bins = 100)
plt.show()
print(np.median(results.lc.posteriors['t_secondary_p1']))
print('END')
model = results.lc.evaluate('inst')
plt.errorbar(times['inst'], fluxes['inst'], errors['inst'], fmt = '.')
plt.plot(times['inst'], model, 'r--', label = 'Fitted model, inst')
plt.plot(times['inst'], tmodel, 'r-', label = 'Input model')

plt.legend(fontsize=22)
plt.show()
