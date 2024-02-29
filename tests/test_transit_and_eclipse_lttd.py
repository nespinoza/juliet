import matplotlib.pyplot as plt
import time
import numpy as np
import random

import juliet
import batman

random.seed(42)

activate_lttd = True

times, fluxes, errors = {}, {}, {}

# Load dataset:
tt, tmodel = np.loadtxt('lttd-transit1.dat', unpack = True, usecols = (0,1))
te, emodel = np.loadtxt('lttd-eclipse1.dat', unpack = True, usecols = (0,1))
note, noemodel = np.loadtxt('NOlttd-eclipse1.dat', unpack = True, usecols = (0,1))
tt2, tmodel2 = np.loadtxt('lttd-transit2.dat', unpack = True, usecols = (0,1))

times['inst'], tmodel = np.append(tt,te), np.append(tmodel, emodel)

# Generate yet another dataset but with a different depth and transit time:
times['inst2'] = tt2

sigma = 50*1e-6
fluxes['inst'], errors['inst'] = tmodel + np.random.normal(0., sigma, len(tmodel)), np.ones(len(tmodel))*sigma*0.1
fluxes['inst2'], errors['inst2'] = tmodel2 + np.random.normal(0., sigma, len(tmodel2)), np.ones(len(tmodel2))*sigma*0.1

# Define priors:
params = ['P_p1','t0_p1', 'b_p1','ecc_p1','omega_p1', 'p_p1_inst', 'p_p1_inst2', 'q1_inst_inst2', 'q2_inst_inst2',\
          'a_p1', 'mdilution_inst_inst2', 'mflux_inst', 'sigma_w_inst', 'fp_p1', \
          'mflux_inst2', 'sigma_w_inst2']

# Distributions:
dists = ['fixed','normal','fixed','uniform','uniform', 'uniform','uniform','uniform','uniform',\
         'uniform', 'fixed', 'normal', 'loguniform', 'uniform', \
         'normal', 'loguniform']

# Hyperparameters
hyperps = [1.0, [0.0,0.1], 0.0, [0.,0.5], [-180., 180.], [0.,0.3], [0.,0.3], [0., 1.], [0., 1.],\
           [1.,10.], 1.0, [0.,0.1], [0.1, 1000.], [0.,0.1],\
           [0.,0.1], [0.01, 1000.]]

# Join priors:
priors = juliet.generate_priors(params,dists,hyperps)

print('Priors:')
print('---------')
print(priors)
# And fit:
sampler = 'multinest'
    
if activate_lttd:

    dataset = juliet.load(priors = priors, t_lc = times, y_lc = fluxes, yerr_lc = errors, out_folder = 'transit-and-eclipse-lttd-test-activated', verbose = True)
    results = dataset.fit(sampler = sampler, progress = True, light_travel_delay = True, stellar_radius = 1.)

else:

    dataset = juliet.load(priors = priors, t_lc = times, y_lc = fluxes, yerr_lc = errors, out_folder = 'transit-and-eclipse-lttd-test-noactivated', verbose = True)
    results = dataset.fit(sampler = sampler, progress = True)

print('START \n testing:')
print(results.lc.dictionary['inst']['EclipseFit'])
print(results.lc.posteriors.keys())
plt.hist(results.lc.posteriors['ecc_p1'], bins = 100)
plt.show()
print(np.median(results.lc.posteriors['ecc_p1']))
print('END')
model = results.lc.evaluate('inst')
model2 = results.lc.evaluate('inst2')
plt.errorbar(times['inst'], fluxes['inst'], errors['inst'], fmt = '.')
plt.errorbar(times['inst2'], fluxes['inst2'], errors['inst2'], fmt = '.')
plt.plot(times['inst'], model, 'r--', label = 'Fitted model, inst')
plt.plot(times['inst2'], model2, 'r--', label = 'Fitted model, inst2')
plt.plot(te, noemodel, 'b-', label = 'Input model, without time delay')
plt.plot(times['inst'], tmodel, 'r-', label = 'Input model')
plt.plot(times['inst2'], tmodel2, 'r-', label = 'Input model')

plt.legend(fontsize=22)
plt.show()

plt.errorbar(times['inst'], (fluxes['inst'] - model)*1e6, errors['inst'], fmt = '.')
plt.show()
