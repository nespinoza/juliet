.. _rvfits:

Fitting radial-velocities
===================

In ``juliet``, the radial-velocity model is essentially the same as the one already introduced for the lightcurve in 
the :ref:`transitfit` tutorial, i.e., in the absence of extra linear terms (see :ref:`linearmodels`), is of the form 
(see Section 2 of the `juliet paper <https://arxiv.org/abs/1812.08549>`_)

:math:`\mathcal{M}_{i}(t) + \epsilon_i(t)`,

where :math:`\epsilon_i(t)` is a noise model for instrument :math:`i` (which as for the :ref:`transitfit` tutorial, 
here we assume is white-gaussian noise --- i.e., we assume :math:`\epsilon_i(t)\sim \mathcal{N}(0,\sqrt{\sigma(t)^2 + \sigma_{w,i}})`, 
where :math:`\sigma^2_{w,i}` is a jitter term added to each instrument --- we extend this to gaussian processes in the 
:ref:`gps` tutorial), and :math:`\mathcal{M}_{i}(t)` is the deterministic part of the radial-velocity model for the instrument. 
The form of this deterministic part of the model is given by

:math:`\mathcal{M}_{i}(t) = \mathcal{K}(t) + \mu_i + Q(t-t_a)^2 + A(t-t_a) + B`.

Here, :math:`\mathcal{K}(t)` is a Keplerian model which models the RV perturbations on the star due to the planets orbiting 
around it, :math:`\mu_i` is the RV of the star as measured by instrument :math:`i` and the coefficients :math:`Q, A` and 
:math:`B` define an additional long-term trend useful for modelling long-period signals in the RVs that might not be well 
modelled by an additional Keplerian signal --- :math:`t_a` is just an arbitrary value substracted to the input times for 
numerical stability of the coefficients (by default :math:`t_a = 2458460` --- but this can be defined by the user). By default, 
no long-term trend is incorporated in the models (i.e., :math:`Q = A = B = 0`).


RV fits
-------

To showcase the capabilities ``juliet`` has for radial-velocity fitting, here we will analyze the radial-velocities of the 
TOI-141 system (`Espinoza et al. (2019) <https://arxiv.org/abs/1903.07694>`_). We already analyzed the transits of this 
object in the :ref:`quicktest` tutorial; here we use the radial-velocities (RVs) of this system as it was shown that not 
only the signal of the transiting planet was present in the RVs, but there is also evidence for _another_ planet in the system. 
We have uploaded the dataset in a ``juliet``-friendly format [`here <https://github.com/nespinoza/juliet/blob/master/docs/tutorials/rvs_toi141.dat>`_].

Let us first try to find the RV signature of the transiting planet analyzed in the :ref:`transitfit` tutorial in this dataset. 
From that analysis, the period is :math:`P = 1.007917 \pm 0.000073` days and the time-of-transit center is 
:math:`t0 = 2458325.5386 \pm 0.0011`. Let us use these as priors for a first fit to the data --- let us in turn assume uniform wide 
priors for the systemic velocities for each instrument :math:`\mu_i`, jitter terms and RV semi-amplitude; let us also fix the eccentricity 
to zero for now:

.. code-block:: python
 
    import juliet
    priors = {}

    # Name of the parameters to be fit:
    params = ['P_p1','t0_p1','mu_CORALIE14', \
              'mu_CORALIE07','mu_HARPS','mu_FEROS',\
              'K_p1', 'ecc_p1', 'omega_p1', 'sigma_w_CORALIE14','sigma_w_CORALIE07',\
               'sigma_w_HARPS','sigma_w_FEROS']

    # Distributions:
    dists = ['normal','normal','uniform', \
             'uniform','uniform','uniform',\
             'uniform','fixed', 'fixed', 'loguniform', 'loguniform',\
             'loguniform', 'loguniform']

    # Hyperparameters
    hyperps = [[1.007917,0.000073], [2458325.5386,0.0011], [-100,100], \
               [-100,100], [-100,100], [-100,100], \
               [0.,100.], 0., 90., [1e-3, 100.], [1e-3, 100.], \
               [1e-3, 100.], [1e-3, 100.]]

    # Populate the priors dictionary:
    for param, dist, hyperp in zip(params, dists, hyperps):
        priors[param] = {}
        priors[param]['distribution'], priors[param]['hyperparameters'] = dist, hyperp
        print param,priors[param]

    dataset = juliet.load(priors = priors, rvfilename='rvs_toi141.dat', out_folder = 'toi141_rvs')
    results = dataset.fit(n_live_points = 300)   

To plot the data, one can extract the models in an analogous fashion as we did for the :ref:`transitfit` tutorial: we 
use the ``results.rv.evaluate()`` function. As with the ``results.lc.evaluate()`` function presented in the 
:ref:`transitfit` tutorial, the function receives an instrument name and optionally times in which one wants to evaluate the 
model. Because each of the RV model parts are additive, it is easy to extract, e.g., the systemic-velocity corrected keplerian 
signal by simply evaluating the model in an arbitrary instrument and substracting the median of the systemic-velocity for 
that instrument. Let us do this to plot the above defined fit to see how we did --- we'll only plot the HARPS and FEROS 
data, as the CORALIE data is not very constraining:

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt 
    
    # Plot HARPS and FEROS datasets in the same panel. For this, first select any 
    # of the two and substract the systematic velocity to get the Keplerian signal. 
    # Let's do it with FEROS. First generate times on which to evaluate the model:
    min_time, max_time = np.min(dataset.times_rv['FEROS'])-30,\
                         np.max(dataset.times_rv['FEROS'])+30

    model_times = np.linspace(min_time,max_time,1000)

    # Now evaluate the model in those times, and substract the systemic-velocity to 
    # get the Keplerian signal:
    keplerian = results.rv.evaluate('FEROS', t = model_times) - \
                np.median(results.posteriors['posterior_samples']['mu_FEROS'])

    # Now plot the (systematic-velocity corrected) RVs:
    fig = plt.figure(figsize=(12,5))
    instruments = ['FEROS','HARPS']
    colors = ['cornflowerblue','orangered']
    for i in range(len(instruments)):
        instrument = instruments[i]
        # Evaluate the median jitter for the instrument:
        jitter = np.median(results.posteriors['posterior_samples']['sigma_w_'+instrument])
        # Evaluate the median systemic-velocity:
        mu = np.median(results.posteriors['posterior_samples']['mu_'+instrument])
        # Plot original data with original errorbars:
        plt.errorbar(dataset.times_rv[instrument]-2457000,dataset.data_rv[instrument]-mu,\
                     yerr = dataset.errors_rv[instrument],fmt='o',\
                     mec=colors[i], ecolor=colors[i], elinewidth=3, mfc = 'white', \
                     ms = 7, label=instrument, zorder=10)

        # Plot original errorbars + jitter (added in quadrature):
        plt.errorbar(dataset.times_rv[instrument]-2457000,dataset.data_rv[instrument]-mu,\
                     yerr = np.sqrt(dataset.errors_rv[instrument]**2+jitter**2),fmt='o',\
                     mec=colors[i], ecolor=colors[i], mfc = 'white', label=instrument,\
                     alpha = 0.5, zorder=5)

    # Plot Keplerian model:
    plt.plot(model_times-2457000, keplerian,color='black',zorder=1)
    plt.ylabel('RV (m/s)')
    plt.xlabel('Time (BJD - 2457000)')
    plt.title('1 Planet Fit | Log-evidence: {0:.3f} $\pm$ {1:.3f}'.format(results.posteriors['lnZ'],\
           results.posteriors['lnZerr']))
    plt.ylim([-20,20])
    plt.xlim([1365,1435]) 

.. figure:: rvfit.png
   :alt: Results for the 1-planet fit.

Interesting. We have plotted both the original data with the original errorbars, and the errorbars 
enlarged by the best-fit jitter term. Note how the jitter is large (specially for HARPS)? This is to 
explain the large variations that appear in this 1-planet-fit result. Could this be due to an additional 
planet? To test this hypothesis, let's try another fit but now fitting for *two* planets: the 1-day transiting one, 
and an additional one with an unknown period from, say, 1 to 10 days. To do this, add the extra priors for this model first: 

.. code-block:: python

    # Add second planet to the prior:
    params = params + ['P_p2',   't0_p2',  'K_p2',    'ecc_p2','omega_p2']
    dists = dists +   ['uniform','uniform','uniform', 'fixed', 'fixed']
    hyperps = hyperps + [[1.,10.],[2458325.,2458330.],[0.,100.], 0., 90.]

    # Repopulate priors dictionary:
    priors = {}

    for param, dist, hyperp in zip(params, dists, hyperps):
        priors[param] = {}
        priors[param]['distribution'], priors[param]['hyperparameters'] = dist, hyperp
        print param,priors[param]

And let's perform the second ``juliet`` fit with this two-planet system:

.. code-block:: python

    dataset = juliet.load(priors = priors, rvfilename='rvs_toi141.dat', out_folder = 'toi141_rvs_2planets')
    results2 = dataset.fit(n_live_points = 300)

Repeating the same plot as above we find:

.. figure:: rvfit_2pl.png
   :alt: Results for the 2-planet fit.

Woah! Much better fit to the data. Note also that we have plotted the log-evidences that ``juliet`` gives for these 
models --- and the log-evidence for the 2-planet model is much larger than the one for the 1-planet model, 
:math:`\Delta \ln Z = 114.4` which is a `huge` odds ratio in favor of the two-planet model. Let's plot the posterior distributions 
for the parameters of this fit using Daniel Foreman-Mackey's `corner <https://corner.readthedocs.io>`_ package:

.. code-block:: python

    import corner

    posterior_names = [r"$K_1$ (m/s)", r"$P_2$ (days)", r"$K_2$ (m/s)"]
    first_time = True
    for i in range(len(params)):
        if dists[i] != 'fixed' and params[i] != 'P_p1' and 't0' not in params[i] and \
        params[i][0:2] != 'mu' and params[i][0:5] != 'sigma':
            if first_time:
                posterior_data = results2.posteriors['posterior_samples'][params[i]]
                first_time = False
            else:
                posterior_data  = np.vstack((posterior_data, results2.posteriors['posterior_samples'][params[i]]))
    posterior_data = posterior_data.T    
    figure = corner.corner(posterior_data, labels = posterior_names)

.. figure:: corner-2planet.png
   :alt: Corner plot for results for the 2-planet fit.

Best-fit period of this second planet is at 4.76 days: just like in the paper! The semi-amplitudes mostly agree as well. 
We are only missing the addition of Gaussian Processes to this fit, which maybe might explain some of the extra variance 
observed in the best-fit model above --- we touch on this problem in the :ref:`gps` tutorial.
