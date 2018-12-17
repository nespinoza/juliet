flags within juliet 
---

Fun With Flags
------------
Below, we list the set of flags that can be used within `juliet`. For examples on how to use them, 
please read `juliet`'s wiki page.

`-lcfile lightcurve_filename.dat`

This flag tells `juliet` where to find the `lightcurve_filename.dat` file containing the times, 
relative fluxes, errors and instruments of the transit dataset. `juliet` expects that in the 
first column this file has time, in the second it has relative fluxes, in the third errors on those 
relative fluxes and in the fourth the instrument names.

`-rvfile rv_filename.dat`

This flag tells `juliet` where to find the `rv_filename.dat` file containing the times, 
radial-velocities, errors and instruments of the radial-velocity (RV) dataset. `juliet` expects that 
in the first column this file has time, in the second it has RV, in the third errors on the RVs 
and in the fourth the instrument names.

`-lceparamfile lc_eparam_filename.dat`

This flag tells `juliet` where to find a file with the external parameters to be used to "detrend" the data 
of a given instrument. `juliet` expects that the lightcurve file (e.g., `lightcurve_filename.dat`) is 
synchronized in the row number with this file for each instrument. For example, if there are two datapoints 
for instrument A in rows 1 and 2, the external parameters for instrument A have to have the external paramerters 
at the times of row 1 in the first row defining the external parameters for instrument A and the external parameters 
at the times of row 2 in the second row. 

`-rveparamfile`

Same as for `lceparamfile`, but for radial-velocities.

`-ofolder`

This flag reads an output folder:

`-ldlaw`

This flag defines the limb-darkening to be used. Can be either common to all instruments (e.g., give 'quadratic' as input), 
or it can be different for every instrument, in which case you must pass a comma separated list of instrument-ldlaw pair, e.g. 
'TESS-quadratic,CHAT-linear'.

`-lctimedef`

Lightcurve time definitions (e.g., 'TESS-TDB,CHAT-UTC', etc.). If not given, it is assumed all lightcurves are in TDB:

`-rvtimedef`

Radial-velocities time definitions (e.g., 'HARPS-TDB,CORALIE-UTC', etc.). If not given, it is assumed all RVs are in UTC:

`-priorfile`

This reads the prior file.

`-rvunits`

This defines if rv units are m/s (ms) or km/s (kms); useful for plotting. Default is m/s.

`-nrvchunk`

This defines the minimum chunk (in days) of RV data that activates multi-panel plots. Each panel will have data within nrvchunk days.

`--plotbinnedrvs`

Decide if binned RVs will be plotted at the end:

`-ecclime`

Allow user to change the maximum eccentricity for the fits; helps avoid issue that Batman can run into with high eccentricities

`-sdensity_mean`

Define stellar density mean.

`-sdensity_sigma`

Define stellar density stdev.

`-efficient_bp`

Define if the sampling for p and b in Espinoza (2018) wants to be used; define pl and pu (this assumes 
sampling parameters in prior file are r1 and r2):

`-pl`

pl for --efficient_bp

`-pu`

pu for --efficient_bp

`-nlive`

Number of live points.

`-nsims`

Number of samples to draw from posterior to compute models/plots.

`-n_supersamp`, `-exptime_supersamp` and `-instrument_supersamp`

Dealing with supersampling for long exposure times for LC. n_supersamp is the number of 
supersampled points, exptime_supersamp the exposure time and instrument_supersamp the instrument
for which you want to apply supersampling. If you need several instruments to have supersampling,
you can give these input as comma separated values, e.g., '-instrument_supersamp TESS,K2 -n_supersamp 20,30 -exptime_supersamp 0.020434,0.020434' 
will give values of n_supersamp of 20 and 30 to TESS and K2 lightcurves, respectively, and both of them with texp of 0.020434 days.

`--geroge_hodlr`

Define if HODLRSolver wants to be used for george. Only applied to photometric GPs:

`--dynamic`

Define if Dynamic Nested Sampling is to be used:

`--use_dynesty`

Define if dynesty will be used.

`-dynesty_bound`

Define some arguments for dynesty runs (see https://dynesty.readthedocs.io/en/latest/api.html); default is single. 

`-dynesty_sample`

Method used to sample uniformly within the likelihood constraint, conditioned on the provided bounds (default is rwalk).

`-dynesty_nthreads`

Number of threads to use within dynesty (giving a number here assumes one wants to perform multithreading):
