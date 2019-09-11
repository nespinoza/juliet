.. _quicktest:

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


