import os
import numpy as np

import matplotlib.pyplot as plt

import juliet

import batman

# Create dataset:
def get_transit_model(t):
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
    tmodel = batman.TransitModel(params, t.astype('float64'))
    return tmodel.light_curve(params)

def standarize_variable(x):
    
    return (x - np.mean(x)) / np.sqrt(np.var(x))

# Generate times (over an hour) and fake flat fluxes:
times = np.linspace(-0.1, 0.1, 300)
fluxes = get_transit_model(times)

# Add noise (if not already added):
if not os.path.exists('transit_test.txt'):

    sigma = 100 # ppm
    noise = np.random.normal(0., sigma*1e-6, len(times))

    dataset = fluxes + noise

    fout = open('transit_test.txt', 'w')
    for i in range(len(dataset)):

        fout.write('{0:.10f}\n'.format(dataset[i]))

else:

    dataset = np.loadtxt('transit_test.txt', unpack=True)

# Fit:
jtimes, jfluxes, jfluxes_error = {}, {}, {}
jtimes['instrument'], jfluxes['instrument'], jfluxes_error['instrument'] = times, dataset, np.ones(len(dataset))*1e-6

# Try now a Matern 3/2:
params = ['mdilution_instrument', 'mflux_instrument', 'sigma_w_instrument',  
          'P_p1', 't0_p1', 'p_p1', 'a_p1', 'b_p1', 'u1_instrument', 'u2_instrument', 'ecc_p1', 'omega_p1']

# Distributions:
dists = ['fixed', 'normal', 'loguniform', 
         'fixed', 'fixed', 'uniform', 'fixed', 'fixed', 'uniform', 'uniform', 'fixed', 'fixed']

# Hyperparameters:
hyperps = [1.0, [0., 0.1], [10., 1000.], 
           1., 0., [0, 0.2], 3.6, 0., [-3.,3.], [-3.,3.], 0., 90.]

priors = juliet.generate_priors(params, dists, hyperps)

# Fit with hodlr:
jdataset = juliet.load(priors=priors, t_lc=jtimes, y_lc=jfluxes, \
                       yerr_lc=jfluxes_error,\
                       out_folder = 'transit-u1u2-power2', \
                       ld_laws = 'power2',\
                       verbose = True)

results = jdataset.fit(sampler = 'dynamic_dynesty', progress = True)

# Plot:
model = results.lc.evaluate('instrument')

plt.plot(times, dataset, '.', label = 'data')
plt.plot(times, model, label = 'Fitted model (transit)')
plt.plot(times, fluxes, label = 'True model')

plt.legend()
plt.show()
