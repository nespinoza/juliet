import time
import numpy as np
import random

import juliet
import batman

random.seed(42)

# Create dataset:
def get_transit_model(n = 300):
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
    t = np.linspace(-0.1,0.1,n)
    tmodel = batman.TransitModel(params, t.astype('float64'))
    return t, tmodel.light_curve(params)

# Set number of datapoints the lightcurve will have; define dictionaries 
# that will host the data:
n = 300
times, fluxes, errors = {}, {}, {}

# Get data, add noise, put it in the dictionaries. Note sigma is underestimated:
times['inst'], tmodel = get_transit_model(n)
sigma = 100*1e-6
fluxes['inst'], errors['inst'] = tmodel + np.random.normal(0., sigma, n), np.ones(300)*sigma*0.1

# Define priors:
params = ['P_p1','t0_p1','p_p1','b_p1','q1_inst','q2_inst','ecc_p1','omega_p1',\
              'a_p1', 'mdilution_inst', 'mflux_inst', 'sigma_w_inst']

# Distributions:
dists = ['fixed','normal','uniform','fixed','uniform','uniform','fixed','fixed',\
                 'normal', 'fixed', 'normal', 'loguniform']

# Hyperparameters
hyperps = [1.0, [0.0,0.1], [0.,1], 0.0, [0., 1.], [0., 1.], 0.0, 90.,\
                   [3.6,0.1], 1.0, [0.,0.1], [0.1, 1000.]]

# Join priors:
priors = juliet.generate_priors(params,dists,hyperps)

# Define starting point:
starting_point = {}
starting_point['t0_p1'] = 0.
starting_point['p_p1'] = 0.1
starting_point['q1_inst'] = 0.5
starting_point['q2_inst'] = 0.5
starting_point['a_p1'] = 3.6
starting_point['mflux_inst'] = 0.
starting_point['sigma_w_inst'] = 100.

# And fit:
samplers = ['slicesampler_ultranest', 'emcee', 'dynesty', 'dynamic_dynesty', 'ultranest', 'multinest']
all_times = {}
for sampler in samplers:
    dataset = juliet.load(priors = priors, t_lc = times, y_lc = fluxes, yerr_lc = errors, out_folder = sampler+'-test', starting_point = starting_point)
    t0 = time.time()
    dataset.fit(sampler = sampler, progress = True)
    t1 = time.time()
    total = t1 - t0
    all_times[sampler] = total
    print(sampler,' took ',total,' seconds to run.')
print('timing results (in seconds):')
print(all_times)
