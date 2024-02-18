import matplotlib.pyplot as plt
import time
import numpy as np
import random

import juliet
import batman

random.seed(42)

# Create dataset:
def get_eclipse_model(n = 300, fp = 0.01):
    params = batman.TransitParams()
    params.t0 = 0.0   # time of inferior conjunction
    params.per = 1.0  # orbital period (days)
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
    t = np.linspace(-0.1,0.1,n) + 0.5
    tmodel = batman.TransitModel(params, t.astype('float64'), transittype='secondary')
    return t, tmodel.light_curve(params)

def get_transit_model(n = 300, p = 0.1, t0 = 0.):
    params = batman.TransitParams()
    params.t0 = t0   # time of inferior conjunction
    params.per = 1.0  # orbital period (days)
    params.a = 3.6    # semi-major axis (in units of stellar radii)
    params.rp = p   # rp/rs
    params.inc = 90.  # orbital inclination (in degrees)
    params.ecc = 0.   # eccentricity
    params.w = 90.    # longitude of periastron (in degrees) p
    params.limb_dark = 'quadratic' # limb darkening profile to use
    params.u = [0.2,0.3] # limb darkening coefficients
    t = np.linspace(-0.1,0.1,n) + t0 
    tmodel = batman.TransitModel(params, t.astype('float64'))
    return t, tmodel.light_curve(params)

# Set number of datapoints the lightcurve will have; define dictionaries 
# that will host the data:
n = 300
times, fluxes, errors = {}, {}, {}

# Get data, add noise, put it in the dictionaries. Note sigma is underestimated:
fp = 0.01
te, emodel = get_eclipse_model(n, fp = fp)
emodel = emodel - fp

tt, tmodel = get_transit_model(n)

times['inst'], tmodel = np.append(tt,te), np.append(tmodel, emodel)

# Generate yet another dataset but with a different depth and transit time:
times['inst2'], tmodel2 = get_transit_model(n, p = 0.2, t0 = 1.0)

sigma = 100*1e-6
fluxes['inst'], errors['inst'] = tmodel + np.random.normal(0., sigma, len(tmodel)), np.ones(len(tmodel))*sigma*0.1
fluxes['inst2'], errors['inst2'] = tmodel2 + np.random.normal(0., sigma, len(tmodel2)), np.ones(len(tmodel2))*sigma*0.1

# Define priors:
params = ['P_p1','t0_p1', 'b_p1','ecc_p1','omega_p1', 'p_p1_inst', 'p_p1_inst2', 'q1_inst_inst2', 'q2_inst_inst2',\
          'a_p1', 'mdilution_inst_inst2', 'mflux_inst', 'sigma_w_inst', 'fp_p1', 't_secondary_p1', \
          'mflux_inst2', 'sigma_w_inst2']

# Distributions:
dists = ['fixed','fixed','fixed','fixed','fixed', 'uniform','uniform','uniform','uniform',\
         'normal', 'fixed', 'normal', 'loguniform', 'uniform', 'normal', \
         'normal', 'loguniform']

# Hyperparameters
hyperps = [1.0, 0.0, 0.0, 0.0, 90., [0.,0.3], [0.,0.3], [0., 1.], [0., 1.],\
           [3.6,0.1], 1.0, [0.,0.1], [0.1, 1000.], [0.,0.1], [0.5,0.1],\
           [0.,0.1], [0.1, 1000.]]

# Join priors:
priors = juliet.generate_priors(params,dists,hyperps)

print('Priors:')
print('---------')
print(priors)
# And fit:
sampler = 'multinest'
    
dataset = juliet.load(priors = priors, t_lc = times, y_lc = fluxes, yerr_lc = errors, out_folder = 'transit-and-eclipse-test', verbose = True)
results = dataset.fit(sampler = sampler, progress = True)

print('START \n testing:')
print(results.lc.dictionary['inst']['EclipseFit'])
print(results.lc.posteriors.keys())
plt.hist(results.lc.posteriors['t_secondary_p1'], bins = 100)
plt.show()
print(np.median(results.lc.posteriors['t_secondary_p1']))
print('END')
model = results.lc.evaluate('inst')
model2 = results.lc.evaluate('inst2')
plt.errorbar(times['inst'], fluxes['inst'], errors['inst'], fmt = '.')
plt.errorbar(times['inst2'], fluxes['inst2'], errors['inst2'], fmt = '.')
plt.plot(times['inst'], model, 'r--', label = 'Fitted model, inst')
plt.plot(times['inst2'], model2, 'r--', label = 'Fitted model, inst2')
plt.plot(times['inst'], tmodel, 'r-', label = 'Input model')
plt.plot(times['inst2'], tmodel2, 'r-', label = 'Input model')

plt.legend(fontsize=22)
plt.show()
