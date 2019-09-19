.. _jointfits:

Fitting transits and radial-velocities
===================

We have dealt so far separately between fitting transit lightcurves in the :ref:`transitfit` tutorial and with fitting 
radial-velocity data in the :ref:`rvfits` tutorial. Here, we simply join what we have learned in those tutorials in order 
to showcase the ability of ``juliet`` to fit both dataset simultaneously. 

In the background, ``juliet`` simply assumes both of these datasets are independant but that they can have common 
parameters. For example, the period and time-of-transit center are common to both datasets, but the radial-velocity 
semi-amplitude is only constrained by the radial-velocity dataset. Performing joint fits, thus, one can jointly extract 
information for common parameters between those datasets simultaneously in order to properly propagate that into the 
uncertainties and correlations between all the parameters being constrained.

Here, we use the TOI-141 dataset whose transit information was already presented in the :ref:`quickstart` section, and 
whose radial-velocity data was already presented in the :ref:`rvfits` section.

A joint fit to the TOI-141 system
----------------------------------

In the :ref:`rvfits` tutorial, we have already seen how the RV data (which you can download from [`here <https://github.com/nespinoza/juliet/blob/master/docs/tutorials/rvs_toi141.dat>`_]) support the presence of at least two planets in the system, while in the :ref:`quickstart` section we have already seen 
how to fit a transit lightcurve for this system. Let us then simply join the prior distributions and data from these two sections into one. Let's 
first define the joint prior distribution:

.. code-block:: python

    # Define the master prior dictionary. First define the TRANSIT priors:
    priors = {}

    # Name of the parameters to be fit:
    params = ['P_p1','t0_p1','r1_p1','r2_p1','q1_TESS','q2_TESS','ecc_p1','omega_p1',\
                  'rho', 'mdilution_TESS', 'mflux_TESS', 'sigma_w_TESS']

    # Distribution for each of the parameters:
    dists = ['normal','normal','uniform','uniform','uniform','uniform','fixed','fixed',\
                     'loguniform', 'fixed', 'normal', 'loguniform']

    # Hyperparameters of the distributions (mean and standard-deviation for normal 
    # distributions, lower and upper limits for uniform and loguniform distributions, and 
    # fixed values for fixed "distributions", which assume the parameter is fixed)
    hyperps = [[1.,0.1], [1325.55,0.1], [0.,1], [0.,1.], [0., 1.], [0., 1.], 0.0, 90.,\
                       [100., 10000.], 1.0, [0.,0.1], [0.1, 1000.]]

    # Populate the priors dictionary:
    for param, dist, hyperp in zip(params, dists, hyperps):
        priors[param] = {}
        priors[param]['distribution'], priors[param]['hyperparameters'] = dist, hyperp

    # Now define the RV priors:
    params = ['mu_CORALIE14', 'mu_CORALIE07','mu_HARPS','mu_FEROS', 'K_p1', 'sigma_w_CORALIE14','sigma_w_CORALIE07',\
               'sigma_w_HARPS','sigma_w_FEROS','P_p2',   't0_p2',  'K_p2', 'ecc_p2', 'omega_p2']

    # Distributions:
    dists = ['uniform', 'uniform','uniform','uniform', 'uniform', 'loguniform', 'loguniform',\
             'loguniform', 'loguniform', 'uniform','uniform','uniform', 'fixed', 'fixed']

    # Hyperparameters
    hyperps = [[-100,100], [-100,100], [-100,100], [-100,100], [0.,100.], [1e-3, 100.], [1e-3, 100.], \
               [1e-3, 100.], [1e-3, 100.], [1.,10.], [2458325.,2458330.], [0.,100.], 0., 90.]

    # Populate the priors dictionary:
    for param, dist, hyperp in zip(params, dists, hyperps):
        priors[param] = {}
        priors[param]['distribution'], priors[param]['hyperparameters'] = dist, hyperp

Now let's get the transit data, load the radial-velocity data and priors into ``juliet`` and run the fit:

.. code-block:: python

   import juliet
   import numpy as np

   # First get TESS photometric data:
   t,f,ferr  = juliet.get_TESS_data('https://archive.stsci.edu/hlsps/tess-data-alerts/'+\
                                 'hlsp_tess-data-alerts_tess_phot_00403224672-'+\
                                 's01_tess_v1_lc.fits')

   times, fluxes, fluxes_error = {},{},{}
   times['TESS'], fluxes['TESS'], fluxes_error['TESS'] = t,f,ferr
  
   # RV data is given in a file, so let's just pass the filename to juliet and load the dataset:
   dataset = juliet.load(priors=priors, t_lc = times, y_lc = fluxes, \
                      yerr_lc = fluxes_error, rvfilename='rvs_toi141.dat', \
                      out_folder = 'toi141_jointfit')

   # And now let's fit it!
   results = dataset.fit(n_live_points = 500)

We first should note that this fit has 21 (!) free parameters. Consequently, we have increased the number of live-points 
(with respect to other tutorials were we defined it to be 300) as there is a larger parameter space the live-points 
have to explore (for details on this, check Section 2.5 of the `juliet paper <https://arxiv.org/abs/1812.08549>`_ and 
references therein): having more live-points improves our resolution of the parameter space to be explored. As a rule-of-thumb, 
live-points :math:`n_\textrm{live}` should scale with about the square of the number of parameters :math:`n_p`. In our case, 
:math:`n_p = 21` so :math:`n_\textrm{live}\sim n_p^2 = 440` --- we set it to 500 just to be on the safe side. Given the enlarged 
parameter space and number of live-points, the run will of course take longer to finish --- in my laptop, this fit took about 
15 minutes. 
