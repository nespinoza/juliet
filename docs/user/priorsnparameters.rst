.. _priorsnparameters:

Models, priors and outputs
===================

As it was shown in the :ref:`quicktest` section, a typical ``juliet`` run will make use of two objects which form the core of 
the library: the ``load`` object and the ``fit`` object. The former is used to load a dataset, and the second is used to fit that 
dataset using the samplers supported within ``juliet``, which in turn spits out the result of the fit including posterior 
distributions, fits, etc. In general, a dataset can be loaded to ``juliet`` by the simple call 

.. code-block:: python

    import juliet
    dataset = juliet.load(priors=priors, t_lc = times, y_lc = fluxes, \
                          yerr_lc = fluxes_error, t_rv = rvtimes, y_rv = rvs, \
                          yerr_rv = rv_errs, out_folder = yourfolder)

Here ``times``, ``fluxes`` and ``fluxes_error`` are dictionaries containing the lightcurve data and 
``rvtimes``, ``rvs`` and ``rv_errs`` contain the radial-velocity data, where each key should have the 
instrument name and under each of those instruments an array should be given with the corresponding 
data. Alternatively, you might give paths to files that contain your data given they have times in the 
first column, data in the second, errors in the data in the third column and instrument names in the fourth 
via the ``lcfilename`` and ``rvfilename`` options (e.g., ``juliet.load(...,lcfilename = path_to_lc)``).

The ``priors`` variable, on the other hand, is either a dictionary or a filename containing the prior distribution 
information for each parameter in the model (see below) and ``yourfolder`` is a user-defined folder 
that is used to save the results (and the data!). 

Once a ``dataset`` is loaded it can be fit by doing ``dataset.fit()``. The options of the fit can of 
course be modified --- we refer the users to the API on this documentation for details on this front. 

Exoplanets with juliet, pt. I: planetary parameters
--------------------------------

To date, ``juliet`` is able to model transit and radial-velocities (RVs), each of which have their own set of 
parameters. We have divided the types of parameters into what we call the **planetary parameters** and 
the **instrument parameters**. Within ``juliet``, the former set of parameters are always of the form 
``parameter_pN``, where ``N`` is a number identifier for a given planet (yes, ``juliet`` handles 
multiple-planet systems!). The instrument parameters, on the other hand, are always of the form 
``parameter_instrument``, where ``instrument`` is an instrument name.

The (basic) **planetary parameters** currently supported by ``juliet`` are:

+------------------+-----------------------------------------------------------------------+
| Parameter name   |           Description                                                 |
+==================+=======================================================================+
| ``P_p1``         | The planetary period of the planet under study (days).                |
+------------------+-----------------------------------------------------------------------+
| ``t0_p1``        | The time-of-transit center of the planet under study (days).          |
+------------------+-----------------------------------------------------------------------+
| ``p_p1``         | Planet-to-star radius ratio (Rp/Rs).                                  |
+------------------+-----------------------------------------------------------------------+
| ``b_p1``         | Impact parameter of the orbit.                                        |
+------------------+-----------------------------------------------------------------------+
| ``a_p1``         | Scaled semi-major axis of the orbit (a/R*).                           |
+------------------+-----------------------------------------------------------------------+
| ``ecc_p1``       | Eccentricity of the orbit.                                            |
+------------------+-----------------------------------------------------------------------+
| ``omega_p1``     | Argument of periastron passage of the orbit (in degrees).             |
+------------------+-----------------------------------------------------------------------+
| ``K_p1``         | RV semi-amplitude of the orbit of the planet (same units as RV data). |
+------------------+-----------------------------------------------------------------------+

Within ``juliet``, it is very important that the periods of the planets are in chronological order, 
i.e., that ``P_p1 < P_p2 < ....``. This is to avoid solutions in which the periods of the planets 
can be exchanged between the variables. When fitting for transit data, all of the above but ``K`` 
have to be defined for each planet. When fitting radial-velocities, only ``P``, ``t0``, ``ecc``, ``omega`` 
and ``K`` have to be defined. When fitting both, all of these have to be defined.

Although the above are the basic planetary parameters allowed by ``juliet``, the library 
allows to perform three more advanced and efficient parametrizations for some of its 
parameters:

- **The first** is the one proposed by `Espinoza (2018) <https://ui.adsabs.harvard.edu/abs/2018RNAAS...2d.209E/abstract>`_, in which instead of fitting for ``p`` and ``b``, one fits for the parameters ``r1`` and ``r2`` which, if sampled with uniform priors between 0 and 1, are able to allow only physically plausible values for ``p`` and ``b`` (i.e., ``b < 1 + p``). This parametrization needs one to define the smallest planet-to-star radius ratio to be considered, ``pl`` and the maximum planet-to-star radius ratio to be considered, ``pu``. For a coarse search, one could set ``pl`` to zero and ``pu`` to 1 --- these are the default values within ``juliet``.

- **The second parametrization** allowed by ``juliet`` is to define a prior for the stellar density, ``rho`` (in kg/m^3) instead of the scaled semi-major axis of the planets, ``a``. This is useful because setting this for a system, using Kepler's third law one can recover ``a`` for each planet using only the period, ``P``, which is a mandatory parameter for any ``juliet`` run. In this way, instead of fitting for ``a`` for different planetary systems, a single value of ``rho`` can be defined for the system.

- **The third parametrization** has to do with the eccentricity and the argument of periastron. ``juliet`` allows either to (1) fit for them directly (via the ``ecc`` and ``omega`` parameters), (2) to fit for the parameters ``esinomega`` = ``ecc*sin(omega*pi/180)`` and ``ecosomega`` = ``ecc*cos(omega*pi/180)`` or (3) to fit for the parameters ``sesinomega`` = ``sqrt(ecc)*sin(omega*pi/180)`` and ``secosomega`` = ``sqrt(ecc)*cos(omega*pi/180)``. The latter two are typically defined between -1 and 1, and within ``juliet`` it is always ensured that the eccentricity is smaller than 1.

Finally, for RVs there are three additional "planetary parameters" that can be passed, which are helpful to model long-period planets for 
which no full cycles have been observed in the data yet. These are the ``rv_intercept``, ``rv_slope`` and ``rv_quad``. These fit a long-term 
trend to the RVs which is added to the Keplerian model and is of the form ``rv_intercept + (t-ta)*rv_slope + (t-ta)**2*rv_quad``. ``ta`` is 
an arbitrary time, which within ``juliet`` is defined to be ``2458460`` --- this arbitrary time can of course be changed by the user. To 
do it, when fitting a ``dataset`` simply do ``dataset.fit(..., ta = yourdate)``.

Exoplanets with juliet, pt. II: instrumental parameters
--------------------------------

The **instrument parameters** currently supported by ``juliet`` are:

+----------------------------+-------------------------------------------------------------------------------------+
| Parameter name             |           Description                                                               |   
+============================+=====================================================================================+
| ``mdilution_instrument``   | The dilution factor for the photometric `instrument`.                               |
+----------------------------+-------------------------------------------------------------------------------------+
| ``mflux_instrument``       | The offset relative flux for the photometric `instrument`.                          |
+----------------------------+-------------------------------------------------------------------------------------+
| ``sigma_w_instrument``     | A jitter (in ppm or RV units) added in quadrature to the errorbars of `instrument`. |
+----------------------------+-------------------------------------------------------------------------------------+
| ``q1_instrument``          | Limb-darkening parametrization for photometric `instrument`.                        |
+----------------------------+-------------------------------------------------------------------------------------+
| ``q2_instrument``          | Limb-darkening parametrization for photometric `instrument`.                        |
+----------------------------+-------------------------------------------------------------------------------------+
| ``mu_instrument``          | Systemic radial-velocity for a radial-velocity `instrument` (same units as data).   |
+----------------------------+-------------------------------------------------------------------------------------+

Here, ``q1`` and ``q2`` are the limb-darkening parametrizations of `Kipping (2013) <https://ui.adsabs.harvard.edu/#abs/arXiv:1308.0009>`_ 
for two-parameter limb-darkening laws for all laws except for the logarithmic, where they correspond to the transformations in 
`Espinoza & Jordan (2016) <http://adsabs.harvard.edu/abs/2016MNRAS.457.3573E>`_. If the linear law is to be used, the user has to only define 
``q1`` which will be interpreted directly as the limb-darkening coefficient of the linear law. For ``juliet`` analyses only using photometry, 
``mdilution, mflux, sigma_w, q1`` and ``q2`` have to defined. For analyses only using radial-velocity measurements, ``mu`` and ``sigma_w`` 
have to be defined. All must be defined in the case of joint fits. 

By default, the limb-darkening law assumed in the fits for all instruments is the quadratic law. However, one can define different 
limb-darkening laws for different instruments passing a string to the ``ld_laws`` input of the ``juliet.load()`` object, where the 
string defines the names and laws to be used for each instrument separated by commas (e.g., 
``juliet.load(...,ld_laws = 'TESS-quadratic,K2-logarithmic,LCOGT-linear')``. Limb-darkening coefficients and dilution factors can be 
common within instruments, too. To force this, simply give all the instruments that should be common to different instruments 
separated by underscores when passing the `priors` (see below) to ``juliet``, e.g., ``q1_TESS_K2``.

.. WARNING:: Because in `juliet` the internal parameters include underscores (`_`), the instrument names **should not** contain underscores. In this way, for example, instead of naming your instrument `My_Instrument` (as in, e.g., `mdilution_My_instrument`), prefer `My-Instrument` or `MyInstrument` instead.

Exoplanets with juliet, pt. III: linear models & gaussian processes
--------------------------------

There are additional instrument parameters that can be given to `juliet` to account for linear models in the data and/or gaussian-processes. 
For `linear models`, it is assumed each linear regressor ``X`` of instrument ``instrument`` will be weighted by a parameter ``thetaX_instrument``. There 
is no limit to the number of linear terms a given instrument can have, and the linear regressors can either be given directly as a dictionary through 
the ``juliet.load`` call (through the ``linear_regressors_lc`` input for lightcurve linear regressors and/or the ``linear_regressors_rv`` input for 
linear regressors for the radial-velocities), or as extra columns in any input lightcurve or radial-velocity file the user is giving as input to that 
same call. For details, check out the :ref:`linearmodels` tutorial.

For `Gaussian Processes` (GPs), the regressors can be given in a similar manner as for linear regressors when doing the ``juliet.load`` call (i.e., via the 
analogous ``GP_regressors_lc`` and ``GP_regressors_rv`` inputs). Alternatively, the name of a file which contains the different regressors on each column with the 
last column being the instrument name can be given through the same ``juliet.load`` call using the ``GPlceparamfile`` for the file defining the GP regressors 
for the lightcurves and ``GPrveparamfile`` for the file defining the GP regressors for the radial-velocities. 

``juliet`` automatically identifies which kernel the user wants to use for each instrument depending on the name of the GP hyperparameters in the `priors`. 
For instrument-by-instrument models (i.e., GP regressions which are individual to each instrument) the parameter names follow the ``pname_instrument`` form, 
where ``pname`` is any of the parameter names listed below and ``instrument`` is a given instrument (e.g., ``GP_sigma_TESS``). For so-called "global" models, 
which are models that are `not` instrument-specific (for more details on the difference between those types of models, check the ``juliet`` paper and/or the 
:ref:`gps` tutorial), the parameter names follow the ``pname_lc`` form for global lightcurve models, and ``pname_rv`` for radial-velocity global models. 

Below we list the GP kernels implemented so far within `juliet`. More kernels can be implemented upon request and/or via git push to the `juliet` repository --- 
again, for usage details, please check out the :ref:`gps` tutorial:

**Multi-dimensional squared-exponential kernel**

+------------------------------+--------------------------------------------------------------------------------------+
| Hyperparameters              |           Description                                                                |   
+==============================+======================================================================================+
| ``GP_sigma``                 |   Amplitude of the GP (in ppm for the photometry, units of measurements for RVs)     |
+------------------------------+--------------------------------------------------------------------------------------+
| ``GP_alpha0``                | Inverse (squared) length-scale/normalized amplitude of the first external parameter  |
+------------------------------+--------------------------------------------------------------------------------------+
| ``GP_alpha1``                | Inverse (squared) length-scale/normalized amplitude of the second external parameter |
+------------------------------+--------------------------------------------------------------------------------------+
| ...                          | ...                                                                                  |
+------------------------------+--------------------------------------------------------------------------------------+
| ``GP_alphan``                | Inverse (squared) length-scale/normalized amplitude of the n+1 external parameter    |
+------------------------------+--------------------------------------------------------------------------------------+

**Exp-sine-squared kernel**

+------------------------------+-------------------------------------------------------------------------------------+
| Hyperparameters              |           Description                                                               |
+==============================+=====================================================================================+
| ``GP_sigma``                 |   Amplitude of the GP (in ppm for the photometry, units of measurements for RVs)    |
+------------------------------+-------------------------------------------------------------------------------------+
| ``GP_alpha``                 |          Inverse (squared) length-scale of the external parameter                   |
+------------------------------+-------------------------------------------------------------------------------------+
| ``GP_Gamma``                 |                   Amplitude of the sine-part of the kernel                          |
+------------------------------+-------------------------------------------------------------------------------------+
| ``GP_Prot``                  |                   Period of the quasi-periodic kernel                               |
+------------------------------+-------------------------------------------------------------------------------------+

**celerite quasi-periodic kernel**

+------------------------------+-------------------------------------------------------------------------------------+
| Hyperparameters              |           Description                                                               |
+==============================+=====================================================================================+
| ``GP_B``                     |   Amplitude of the GP (in ppm for the photometry, units of measurements for RVs)    |
+------------------------------+-------------------------------------------------------------------------------------+
| ``GP_C``                     |                   Additive factor impacting on the amplitude of the GP              |
+------------------------------+-------------------------------------------------------------------------------------+
| ``GP_L``                     |                           Length-scale of exponential part of the GP                |
+------------------------------+-------------------------------------------------------------------------------------+
| ``GP_Prot``                  |                           Period of the quasi-periodic GP                           |
+------------------------------+-------------------------------------------------------------------------------------+

**celerite Simple Harmonic Oscillator (SHO) kernel**

+------------------------------+-------------------------------------------------------------------------------------+
| Hyperparameters              |           Description                                                               |
+==============================+=====================================================================================+
| ``GP_S0``                    |                          Characteristic power of the SHO                            |
+------------------------------+-------------------------------------------------------------------------------------+
| ``GP_omega0``                |                        Characteristic frequency of the SHO                          |
+------------------------------+-------------------------------------------------------------------------------------+
|         ``GP_Q``             |                              Quality factor of the SHO                              |
+------------------------------+-------------------------------------------------------------------------------------+

**celerite (approximate) Matern kernel**

+------------------------------+-------------------------------------------------------------------------------------+
| Hyperparameters              |           Description                                                               |
+==============================+=====================================================================================+
|       ``GP_sigma``           | Amplitude of the GP (in ppm for the photometry, units of measurements for RVs)      |
+------------------------------+-------------------------------------------------------------------------------------+
|          ``GP_rho``          |                          Time/length-scale of the GP                                |
+------------------------------+-------------------------------------------------------------------------------------+

**celerite exponential kernel**

+------------------------------+-------------------------------------------------------------------------------------+
| Hyperparameters              |           Description                                                               |
+==============================+=====================================================================================+
| ``GP_sigma``                 |   Amplitude of the GP (in ppm for the photometry, units of measurements for RVs)    |
+------------------------------+-------------------------------------------------------------------------------------+
| ``GP_timescale``             |                   Time/length-scale of the GP                                       |
+------------------------------+-------------------------------------------------------------------------------------+

**celerite (approximate) Matern multiplied by exponential kernel**

+----------------------------+-------------------------------------------------------------------------------------+
| Hyperparameters            |           Description                                                               |
+============================+=====================================================================================+
| ``GP_sigma``               | Amplitude of the GP (in ppm for the photometry, units of measurements for RVs)      |
+----------------------------+-------------------------------------------------------------------------------------+
| ``GP_rho``                 |                  Time/length-scale of the Matern part of the GP                     |
+----------------------------+-------------------------------------------------------------------------------------+
| ``GP_timescale``           |               Time/length-scale of the exponential part of the GP                   |
+----------------------------+-------------------------------------------------------------------------------------+


Priors
-------

As introduced at the beggining, a set of priors can be defined for the parameters under consideration via the ``priors`` variable, 
which can be either a filename containing a file with the priors as was done in the :ref:`quicktest` section, or a dictionary, as 
was also done in that section. Currently, `juliet` supports the following prior distributions to be defined for the parameters:

+---------------------+-----------------------------------------------------+----------------------+
|    Distribution     |     Description                                     | Hyperparameters      |
+=====================+=====================================================+======================+
| ``Uniform``         | A uniform distribution defined                      | ``a,b``              |
|                     | between a lower (``a``) and upper (``b``) limit.    |                      |
+---------------------+-----------------------------------------------------+----------------------+
| ``Normal``          |  A normal distribution defined by its mean ``mu``   | ``mu,sigma``         | 
|                     |  and standard-deviation ``sigma``.                  |                      |
+---------------------+-----------------------------------------------------+----------------------+
| ``TruncatedNormal`` |  A normal distribution defined by its mean ``mu``   |                      |
|                     |  and standard-deviation ``sigma``, along with a     | ``mu,sigma,a,b``     |
|                     |  lower (``a``) and upper (``b``) limit defining     |                      |
|                     |  its support.                                       |                      |
+---------------------+-----------------------------------------------------+----------------------+
| ``Jeffreys`` or     |  A log-uniform distribution defined between a       |  ``a,b``             |
| ``Loguniform``      |  lower (``a``) and upper (``b``) limit.             |                      |
+---------------------+-----------------------------------------------------+----------------------+
| ``Beta``            |  A beta distribution having support between 0 and 1 |  ``alpha,beta``      |
|                     |  defined by its ``alpha`` and ``beta`` parameters.  |                      |
+---------------------+-----------------------------------------------------+----------------------+

Note that the hyperparameters have to be passed on the order defined above in the prior file or dictionary. 
Further distributions can be made available for `juliet` upon request, as they are extremely easy to implement. 
If a parameter wants to be fixed to a known value, then the prior distribution can be set to `FIXED`. 

Outputs
-------

Once a ``juliet`` fit is ran (e.g., ``results = dataset.fit()``), this will generate a ``juliet.fit`` object which has several features 
the user can explore. The most important is the ``juliet.fit.posteriors`` dictionary, which contains three important keys: 
``posterior_samples``, which is a dictionary having the posterior samples for all the fitted parameters, ``lnZ``, which has the 
log-evidence for the current fit and ``lnZerr`` which has the error on the log-evidence. This same dictionary is also automatically 
saved to the output folder if there was one defined by the user as a .pkl file. 

In addition, a file called ``posteriors.dat`` file is also printed out if an output folder is given, which is of the form

.. code-block:: bash

   # Parameter Name                 Median                  Upper 68 CI             Lower 68 CI 
   q2_TESS                          0.4072409698            0.3509391055            0.2793487941
   P_p1                             1.0079166018            0.0000827690            0.0000545234
   a_p1                             4.5224665335            0.5972474545            1.3392152148
   q1_TESS                          0.2178116586            0.2583946746            0.1424332922
   r2_p1                            0.0146632299            0.0008468341            0.0006147659
   p_p1                             0.0146632299            0.0008468341            0.0006147659
   b_p1                             0.5122384103            0.2961574900            0.3206523210
   inc_p1                           83.5179400288           4.3439922509            8.1734713106
   mflux_TESS                       -0.0000154812           0.0000021394            0.0000020902
   rho                              1722.5385338667         776.2573107345          1121.9672108451
   t0_p1                            1325.5386166342         0.0008056050            0.0012949209
   r1_p1                            0.6748256069            0.1974383267            0.2137682140
   sigma_w_TESS                     127.3813413245          3.6857084428            3.3647860049

This contains on the first column the parameter name, in the second the median, in the third the upper 68% credibility band in 
the fourth column the 68% lower credibility band of the parameter, as extracted from the posterior distribution. For more output 
results (e.g., model evaluations, predictions, plots) check out the tutorials!
