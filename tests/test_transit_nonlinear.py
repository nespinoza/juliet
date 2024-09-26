import os
import numpy as np

import matplotlib.pyplot as plt

import juliet

import batman

def non_linear_function_instrument1(X, parameter_values):

    return parameter_values['A_instrument'] * np.exp( -X / parameter_values['tau_instrument'] )

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

A, tau = 0.01, 0.1

plt.plot(times, dataset, '.')
dataset = dataset + A * np.exp( - times / tau ) 
jtimes['instrument'], jfluxes['instrument'], jfluxes_error['instrument'] = times, dataset, np.ones(len(dataset))*1e-6

plt.plot(times, dataset, '.')
plt.show()

# Try now a Matern 3/2:
params = ['mdilution_instrument', 'mflux_instrument', 'sigma_w_instrument',  
          'P_p1', 't0_p1', 'p_p1', 'a_p1', 'b_p1', 'q1_instrument', 'q2_instrument', 'ecc_p1', 'omega_p1', \
          'A_instrument', 'tau_instrument']

# Distributions:
dists = ['fixed', 'normal', 'loguniform', 
         'fixed', 'fixed', 'uniform', 'fixed', 'fixed', 'uniform', 'uniform', 'fixed', 'fixed', \
         'uniform', 'uniform']

# Hyperparameters:
hyperps = [1.0, [0., 0.1], [10., 1000.], 
           1., 0., [0, 0.2], 3.6, 0., [0., 1.], [0., 1.], 0., 90., 
           [-10,10], [0,10]]

priors = juliet.generate_priors(params, dists, hyperps)

non_linear_functions = {}
non_linear_functions['instrument'] = {}
non_linear_functions['instrument']['function'] = non_linear_function_instrument1
non_linear_functions['instrument']['regressor'] = times

# Fit with hodlr:
jdataset = juliet.load(priors=priors, t_lc=jtimes, y_lc=jfluxes, \
                       yerr_lc=jfluxes_error,\
                       out_folder = 'transit-nonlinear',\
                       non_linear_functions = non_linear_functions,\
                       verbose = True)

results = jdataset.fit(sampler = 'dynamic_dynesty', progress = True)

# Plot:
model = results.lc.evaluate('instrument')

plt.plot(times, dataset, '.', label = 'data')
plt.plot(times, model, label = 'Fitted model (transit)')
plt.plot(times, fluxes, label = 'True model')

plt.legend()
plt.show()
