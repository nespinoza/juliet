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

`-lcrvfile rv_filename.dat`

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

# This reads the external parameters to fit to the photometry GP:
parser.add_argument('-lceparamfile', default=None)
# This reads the external parameters to fit to the RV GP:
parser.add_argument('-rveparamfile', default=None)
# This reads an output folder:
parser.add_argument('-ofolder', default='results')
# This defines the limb-darkening to be used. Can be either common to all instruments (e.g., give 'quadratic' as input), 
# or it can be different for every instrument, in which case you must pass a comma separated list of instrument-ldlaw pair, e.g.
# 'TESS-quadratic,CHAT-linear', etc.:
parser.add_argument('-ldlaw', default='quadratic')
# Lightcurve time definitions (e.g., 'TESS-TDB,CHAT-UTC', etc.). If not given, it is assumed all lightcurves are in TDB:
parser.add_argument('-lctimedef', default='TDB')
# Radial-velocities time definitions (e.g., 'HARPS-TDB,CORALIE-UTC', etc.). If not given, it is assumed all RVs are in UTC:
parser.add_argument('-rvtimedef', default='UTC')
# This reads the prior file:
parser.add_argument('-priorfile', default=None)
# This defines if rv units are m/s (ms) or km/s (kms); useful for plotting. Default is m/s:
parser.add_argument('-rvunits', default='ms')
# Allow user to change the maximum eccentricity for the fits; helps avoid issue that Batman can run into with high eccentricities
parser.add_argument('-ecclim', default=0.95)
# Define stellar density mean and stdev if you have it --- this will help with a constrained transit fit:
parser.add_argument('-sdensity_mean', default=None)
parser.add_argument('-sdensity_sigma', default=None)
# Define if the sampling for p and b in Espinoza (2018) wants to be used; define pl and pu (this assumes 
# sampling parameters in prior file are r1 and r2):
parser.add_argument('--efficient_bp', dest='efficient_bp', action='store_true')
parser.add_argument('-pl', default=None)
parser.add_argument('-pu', default=None)
# Number of live points:
parser.add_argument('-nlive', default=1000)
# Number of samples to draw from posterior to compute models:
parser.add_argument('-nsims', default=5000)
# Dealing with supersampling for long exposure times for LC. n_supersamp is the number of 
# supersampled points, exptime_supersamp the exposure time and instrument_supersamp the instrument
# for which you want to apply supersampling. If you need several instruments to have supersampling,
# you can give these input as comma separated values, e.g., '-instrument_supersamp TESS,K2 -n_supersamp 20,30 -exptime_supersamp 0.020434,0.020434' 
# will give values of n_supersamp of 20 and 30 to TESS and K2 lightcurves, respectively, and both of them with texp of 0.020434 days.
parser.add_argument('-n_supersamp', default=None)
parser.add_argument('-exptime_supersamp', default=None) 
parser.add_argument('-instrument_supersamp', default=None)
# Define if HODLRSolver wants to be used for george. Only applied to photometric GPs:
parser.add_argument('--george_hodlr', dest='george_hodlr', action='store_true')
# Define if Dynamic Nested Sampling is to be used:
parser.add_argument('--dynamic', dest='dynamic', action='store_true')
# Define if dynesty will be used:
parser.add_argument('--use_dynesty', dest='use_dynesty', action='store_true')
# Define some arguments for dynesty runs (see https://dynesty.readthedocs.io/en/latest/api.html). First, bounded method for dynesty:
parser.add_argument('-dynesty_bound', default='multi')
# Method used to sample uniformly within the likelihood constraint, conditioned on the provided bounds:
parser.add_argument('-dynesty_sample', default='rwalk')
# Number of threads to use within dynesty (giving a number here assumes one wants to perform multithreading):
parser.add_argument('-dynesty_nthreads', default='none')
