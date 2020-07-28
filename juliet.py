from mpl_toolkits.axes_grid.inset_locator import inset_axes
# Import batman, for lightcurve models:
import batman
# Import radvel, for RV models:
import radvel
# Import george for detrending:
import george
# Import celerite for detrending:
import celerite
# Plotting functions:
import seaborn as sns
import argparse
import matplotlib
import matplotlib.pyplot as plt

# Import dynesty for dynamic nested sampling:
try:
    import dynesty
except:
    print('Dynesty installation not detected.')
# Import multinest for (importance) nested sampling:
try:
    import pymultinest
except:
    print('(py)MultiNest installation (or libmultinest.dylib) not detected. juliet will only work with dynesty')
import sys

from scipy.interpolate import interp1d
import numpy as np
import utils
import os
import time

# Prepare the celerite term:
import celerite
from celerite import terms

# This class was written by Daniel Foreman-Mackey for his paper: 
# https://github.com/dfm/celerite/blob/master/paper/figures/rotation/rotation.ipynb
class RotationTerm(terms.Term):
    parameter_names = ("log_amp", "log_timescale", "log_period", "log_factor")
    def get_real_coefficients(self, params):
        log_amp, log_timescale, log_period, log_factor = params
        f = np.exp(log_factor)
        return (
            np.exp(log_amp) * (1.0 + f) / (2.0 + f), 
            np.exp(-log_timescale),
        )   

    def get_complex_coefficients(self, params):
        log_amp, log_timescale, log_period, log_factor = params
        f = np.exp(log_factor)
        return (
            np.exp(log_amp) / (2.0 + f), 
            0.0,
            np.exp(-log_timescale),
            2*np.pi*np.exp(-log_period),
        )   

# Definition of user-defined arguments. Useful when juliet is used in command-line mode:
parser = argparse.ArgumentParser()
# This reads the lightcurve file. First column is time, second column is flux, third flux_err, fourth telescope name:
parser.add_argument('-lcfile', default=None)
# This reads the RV data. First column is time, second column is rv, third rv_err, fourth telescope name:
parser.add_argument('-rvfile', default=None)
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
# This defines the minimum chunk of RV data that activates multi-panel plots:
parser.add_argument('-nrvchunk', default = 100)
# Decide if binned RVs will be plotted at the end:
parser.add_argument('--plotbinnedrvs', dest='plotbinnedrvs', action='store_true') 
# Allow user to change the maximum eccentricity for the fits; helps avoid issue that Batman can run into with high eccentricities
parser.add_argument('-ecclim', default=0.95,type=float)
# Define stellar density mean and stdev if you have it --- this will help with a constrained transit fit:
parser.add_argument('-sdensity_mean', default=None)
parser.add_argument('-sdensity_sigma', default=None)
# Define if the sampling for p and b in Espinoza (2018) wants to be used; define pl and pu (this assumes 
# sampling parameters in prior file are r1 and r2):
#parser.add_argument('--efficient_bp', dest='efficient_bp', action='store_true')
parser.add_argument('-pl', default=0.)
parser.add_argument('-pu', default=1.)
# Flux limits for the phased photometry plots:
parser.add_argument('-phasefluxplotlim', default=None)
# Number of live points:
parser.add_argument('-nlive', default=1000)
# Number of samples to draw from posterior to compute models:
parser.add_argument('-nsims', default=5000)
# Number of oversample points in rvs:
parser.add_argument('-noversample', default=1000)
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

args = parser.parse_args()
# Check george hodlr flag:
george_hodlr = args.george_hodlr

# Check the dynesty and dynamic flag:
use_dynesty = args.use_dynesty
dynamic = args.dynamic

# Check dynesty options:
dynesty_bound = args.dynesty_bound
dynesty_sample = args.dynesty_sample
dynesty_nthreads = args.dynesty_nthreads

from dynesty.utils import resample_equal

print('\n\t \t ---------------------------------------------\n')

print('\t \t                   juliet v.2.0            ') 
print('\t \t                                     ')
print('\t \t      Authors: N. Espinoza, D. Kossakowski')
print('\t \t      Contact: espinoza at mpia.de\n')

print('\t \t ---------------------------------------------\n')
# If not ran already, run dynesty, save posterior samples and evidences to pickle file:
# For this, first check if dynamic or normal NS is going to be used:
if use_dynesty:
    if dynamic:
        prefix = 'dynamic_dynesty_'
        print('\t Running DYNAMIC NESTED SAMPLING (dynesty)')
    else:
        prefix = 'dynesty_'
        print('\t Running NESTED SAMPLING (dynesty)')
else:
    prefix = 'multinest_'
    print('\t Running NESTED SAMPLING (multinest)')

# Output folder:
out_folder = args.ofolder+'/'
if not os.path.exists(out_folder):
    os.mkdir(out_folder)

# Read lightcurves and rvs --- save them to out_folder if not already there:
lcfilename = args.lcfile
rvfilename = args.rvfile
if (not os.path.exists(out_folder+'lc.dat')) and (lcfilename is not None):
    os.system('cp '+lcfilename+' '+out_folder+'lc.dat')
if (not os.path.exists(out_folder+'rvs.dat')) and (rvfilename is not None):
    os.system('cp '+rvfilename+' '+out_folder+'rvs.dat')

# Read the eparam files for lightcurves and rvs --- save them to out_folder if not already there:
lceparamfile = args.lceparamfile
rveparamfile = args.rveparamfile
if (not os.path.exists(out_folder+'lc_eparams.dat')) and (lceparamfile is not None):
    os.system('cp '+lceparamfile+' '+out_folder+'lc_eparams.dat')
if (not os.path.exists(out_folder+'rv_eparams.dat')) and (rveparamfile is not None):
    os.system('cp '+rveparamfile+' '+out_folder+'rv_eparams.dat')

sd_mean,sd_sigma = args.sdensity_mean,args.sdensity_sigma
stellar_density = False
if sd_mean is not None:
    sd_mean,sd_sigma = np.double(sd_mean),np.double(sd_sigma)
    stellar_density = True

if lcfilename is not None:
    t_lc,f_lc,ferr_lc,instruments_lc,instrument_indexes_lc,\
    ninstruments_lc,inames_lc, lm_boolean, lm_arguments = utils.readlc(lcfilename)
    print('\t Photometric instruments:',inames_lc)

    # First of all, generate a dictionary for each instrument. This will save the 
    # lightcurve objects:
    lc_dictionary = {}
    for i in range(ninstruments_lc):
        lc_dictionary[inames_lc[i]] = {}
        # Save if a given instrument will receive resampling:
        lc_dictionary[inames_lc[i]]['resampling'] = False
        # Save if a given instrument has GP fitting ON:
        lc_dictionary[inames_lc[i]]['GPDetrend'] = False
        # Save if transit fitting will be done for a given dataset/instrument (this is so users can fit photometry with, e.g., GPs):
        lc_dictionary[inames_lc[i]]['TransitFit'] = False

    # Convert times from TDB to UTC to match RVs. First, see if there is more than one instrument 
    # time definition. If not, assume all times are TDB and thus convert all times to UTC:
    lctimedefs = args.lctimedef.split(',')
    if len(lctimedefs) == 1:
        t_lc = utils.convert_time(lctimedefs[0].split('-')[-1].lower()+'->utc',t_lc)
    else:
        for lctimedef in lctimedefs:
            instrument,timedef = lctimedef.split('-')
            t_lc[instrument_indexes_lc[instrument]] = utils.convert_time(timedef.split()[0].lower()+'->utc',t_lc[instrument_indexes_lc[instrument]])
    # Extract limb-darkening law. If just one is given, assume same LD law for all instruments. If not, assume a 
    # different law for each instrument:
    ld_laws = args.ldlaw.split(',')
    if len(ld_laws) == 1:
        for i in range(ninstruments_lc):
            lc_dictionary[inames_lc[i]]['ldlaw'] = (ld_laws[0].split('-')[-1]).split()[0].lower()
    else:
        for ld_law in ld_laws:
            instrument,ld = ld_law.split('-')
            lc_dictionary[instrument.split()[0]]['ldlaw'] = ld.split()[0].lower()
                             
    # Extract phased-flux plot limits:
    phasefluxplotlim = args.phasefluxplotlim
    if phasefluxplotlim is not None:
        ylow_pflux,yhigh_pflux = args.phasefluxplotlim.split(',')
        ylow_pflux = np.double(ylow_pflux)
        yhigh_pflux = np.double(yhigh_pflux)
        phasefluxlims = True
    else:
        phasefluxlims = False
      
    # Extract supersampling parameters for transit model. If not given for each instrument, assume all must be 
    # resampled:
    if args.instrument_supersamp is not None:
        instrument_ss = args.instrument_supersamp.split(',')
        n_ss = np.array(args.n_supersamp.split(',')).astype('int')
        exptime_ss = np.array(args.exptime_supersamp.split(',')).astype('double')
        for i in range(len(instrument_ss)):
            lc_dictionary[instrument_ss[i]]['resampling'] = True
            lc_dictionary[instrument_ss[i]]['nresampling'] = n_ss[i]
            lc_dictionary[instrument_ss[i]]['exptimeresampling'] = exptime_ss[i]
    else:
        if args.n_supersamp is not None:
            n_ss = int(args.n_supersamp)
        else: 
            n_ss = args.n_supersamp
        if args.exptime_supersamp is not None:
            exptime_ss = float(args.exptime_supersamp)
            for i in range(ninstruments_lc):
                lc_dictionary[inames_lc[i]]['resampling'] = True
                lc_dictionary[inames_lc[i]]['nresampling'] = n_ss
                lc_dictionary[inames_lc[i]]['exptimeresampling'] = exptime_ss
                
        else: 
            exptime_ss = args.exptime_supersamp
    ###################

else:
    inames_lc = []
if rvfilename is not None:
    t_rv,rv_rv,rverr_rv,instruments_rv,instrument_indexes_rv,\
    ninstruments_rv,inames_rv,lm_rv_boolean, lm_rv_arguments = utils.readlc(rvfilename)
    rvresiduals = np.zeros(len(t_rv))
    rvresiduals_err = np.zeros(len(t_rv))
    idx_ordered_rv = np.argsort(len(t_rv))
    
    # First of all, generate a dictionary for RVs. This will save the 
    # RV objects:
    rv_dictionary = {} 
    rv_dictionary['GPDetrend'] = False
    for i in range(ninstruments_rv):
        rv_dictionary[inames_rv[i]] = {} 
    #    lc_dictionary[inames_lc[i]]['resampling'] = False
    #    lc_dictionary[inames_lc[i]]['GPDetrend'] = False
else:
    inames_rv = []

# Read priors, number of planets that transit and number of planets in RVs:
priorfile = args.priorfile
priors,n_transit,n_rv,numbering_transit,numbering_rv,n_params = utils.readpriors(priorfile)

# Check if RV line will be fitted:
if 'rv_slope' in priors.keys():
    if 'rv_quad' in priors.keys():
        fitrvquad = True
        fitrvline = False
    else:
        fitrvline = True
        fitrvquad = False
else:
    fitrvline = False
    fitrvquad = False

# Check if stellar density will be fitted instead of the a/R_*:
if 'rho' in priors.keys():
    fitrho = True
    print('\t Fitting of stellar density detected.')
else:
    fitrho = False


# Check for which instruments transits will be fitted:
if lcfilename is not None:
    for i in range(ninstruments_lc):
        for pri in priors.keys():
            if pri[0:2] == 'q1':
                if inames_lc[i] in pri.split('_'):
                    lc_dictionary[inames_lc[i]]['TransitFit'] = True
                    print('\t Transit fit detected for instrument ',inames_lc[i])

# Check eccentricity parametrization for each planet in the juliet numbering scheme.
# 0 = ecc, omega  1: ecosomega,esinomega  2: sqrt(e)cosomega, sqrt(e)sinomega
ecc_parametrization = {} # np.zeros(np.max([n_transit,n_rv]))
ecc_parametrization['rv'] = {}
ecc_parametrization['transit'] = {}
# At the same time, check if efficient exploration of the (b,p) plane was forced for each planet 
# (i.e., check if planets where constrained by r1,r2 or by b,p directly). Assume it was not for 
# each planet:
efficient_bp = n_transit*[False]
# First for the transiting planets:
for n in range(n_transit):
    i = numbering_transit[n]
    if 'ecosomega_p'+str(i) in priors.keys():
        ecc_parametrization['transit'][i] = 1
        print('\t ecosomega,esinomega parametrization for transiting planet ',i)
    elif 'secosomega_p'+str(i) in priors.keys():
        ecc_parametrization['transit'][i] = 2
        print('\t sqrt(e)cosomega,sqrt(e)sinomega parametrization for transiting planet ',i)
    else:
        ecc_parametrization['transit'][i] = 0
        print('\t e,omega parametrization for transiting planet ',i)
    if 'r1_p'+str(i) in priors.keys():
        efficient_bp[n] = True
        print('\t Efficient sampling of the (b,p) plane detected for planet p'+str(i))
        pl,pu = np.double(args.pl),np.double(args.pu)
        Ar = (pu - pl)/(2. + pl + pu)
for n in range(n_rv):
    i = numbering_rv[n]
    if 'ecosomega_p'+str(i) in priors.keys():
        ecc_parametrization['rv'][i] = 1
        print('\t ecosomega,esinomega parametrization for RV planet ',i)
    elif 'secosomega_p'+str(i) in priors.keys():
        ecc_parametrization['rv'][i] = 2
        print('\t sqrt(e)cosomega,sqrt(e)sinomega parametrization for RV planet ',i)
    else:
        ecc_parametrization['rv'][i] = 0
        print('\t e,omega parametrization for RV planet ',i)

# Save prior file to output folder if not already there:
if not os.path.exists(out_folder+'priors.dat'):
    os.system('cp '+priorfile+' '+out_folder+'priors.dat')

# RV units:
rvunits = args.rvunits
# nrvchunk:
nrvchunk = np.double(args.nrvchunk)
# If binned rvs will be plotted:
plotbinnedrvs = args.plotbinnedrvs

if lcfilename is not None:
    # Float the times (batman doesn't like non-float 64):
    t_lc = t_lc.astype('float64')

# Extract external parameter file for LC GP:
if lceparamfile is not None:
    GPDict = utils.readeparams(lceparamfile)
    for instrument in GPDict.keys():
        print('\t Fitting photometric GP to ',instrument,' instrument.')
        # Detect the GP the user wants for this instrument using the priors dictionary. 
        # For this, consider the possibility that the user might want to combine the GP parameters 
        # between instruments, so iterate.
        for pnames in priors.keys():
            vec = pnames.split('_')
            if (vec[0] == 'GP') and ('alpha0' in vec[1]) and (instrument in vec):
                # Detected multi-dimensional squared-exponential GP:
                lc_dictionary[instrument]['GPType'] = 'SEKernel'
                break
            if (vec[0] == 'GP') and ('malpha0' in vec[1]) and (instrument in vec):
                # Detected multi-dimensional Matern 3/2 GP:
                lc_dictionary[instrument]['GPType'] = 'M32Kernel'
                break
            if (vec[0] == 'GP') and ('Gamma' in vec[1]) and (instrument in vec):
                # Detected exp-sine-squared kernel:
                lc_dictionary[instrument]['GPType'] = 'ExpSineSquaredSEKernel'
                break
            if (vec[0] == 'GP') and ('B' in vec[1]) and (instrument in vec):
                # Detected celerite quasi-periodic kernel:
                lc_dictionary[instrument]['GPType'] = 'CeleriteQPKernel'
                break

            if (vec[0] == 'GP') and ('timescale' in vec[1]) and (instrument in vec):
                # If already defined, then this is a multiplication of Matern and Exp kernels:
                if 'GPType' in lc_dictionary[instrument]:
                    lc_dictionary[instrument]['GPType'] = 'CeleriteMaternExpKernel'
                    break
                # Detected celerite Exp kernel:
                lc_dictionary[instrument]['GPType'] = 'CeleriteExpKernel'

            if (vec[0] == 'GP') and ('rho' in vec[1]) and (instrument in vec):
                # If already defined, then this is a multiplication of Matern and Exp kernels:
                if 'GPType' in lc_dictionary[instrument]:
                    lc_dictionary[instrument]['GPType'] = 'CeleriteMaternExpKernel'
                    break
                # Detected celerite matern:
                lc_dictionary[instrument]['GPType'] = 'CeleriteMatern'

            if (vec[0] == 'GP') and ('Plife' in vec[1]) and (instrument in vec):
                # Detected celerite SHO:
                lc_dictionary[instrument]['GPType'] = 'CeleriteSHOKernel'
                break

        print('\t Detected ',lc_dictionary[instrument]['GPType'],'for the GP')
        # For each instrument for which there are external parameters, activate GP:
        lc_dictionary[instrument]['GPDetrend'] = True 
        # Save variables, standarize them if GP is SEKernel or M32Kernel:
        lc_dictionary[instrument]['X'] = GPDict[instrument]['variables']
        lc_dictionary[instrument]['nX'] = lc_dictionary[instrument]['X'].shape[1]
        print('\t (',lc_dictionary[instrument]['nX'],'external parameters)')
        if (lc_dictionary[instrument]['GPType'] == 'SEKernel') \
        or (lc_dictionary[instrument]['GPType'] == 'M32Kernel'):
            for i in range(lc_dictionary[instrument]['nX']):
                lc_dictionary[instrument]['X'][:,i] = (lc_dictionary[instrument]['X'][:,i] - \
                                                      np.mean(lc_dictionary[instrument]['X'][:,i]))/\
                                                      np.sqrt(np.var(lc_dictionary[instrument]['X'][:,i]))
        if lc_dictionary[instrument]['GPType'] == 'CeleriteQPKernel':
            rot_kernel = terms.TermSum(RotationTerm(log_amp=np.log(10.),\
                                                    log_timescale=np.log(10.0),\
                                                    log_period=np.log(3.0),\
                                                    log_factor=np.log(1.0)))

            # Now that we know the type of GP, we extract the "instrument" corresponding to each GP 
            # parameter. For example, for the ExpSineSquaredKernel, it might happen the user wants 
            # to have a common GP_Prot parameter shared along many instruments, e.g., GP_Prot_TESS_K2_RV,
            # which means the user wants a common Prot for TESS and K2 photometry, and also for the RVs. However, 
            # the same GP might have a different Gamma, i.e., there might be a GP_Gamma_TESS, GP_Gamma_K2 and GP_Gamma_RV.
            # The idea here is to, e.g., in the case of TESS photometry, gather lc_dictionary[instrument]['GP_Prot'] = 
            # 'TESS_K2_RV', and lc_dictionary[instrument]['GP_Gamma'] = 'TESS':
            for GPvariable in ['B','C','L','Prot']:
                for pnames in priors.keys():
                    vec = pnames.split('_')
                    if (vec[0] == 'GP') and (GPvariable in vec[1]) and (instrument in vec):
                        lc_dictionary[instrument]['GP_'+GPvariable] = '_'.join(vec[2:])
         
            # Jitter term:
            kernel_jitter = terms.JitterTerm(np.log(100*1e-6))

            # Wrap GP object to compute likelihood:
            kernel = rot_kernel + kernel_jitter
            lc_dictionary[instrument]['GPObject'] = celerite.GP(kernel, mean=0.0)
            # Note order of GP Vector: logB, logL, logProt, logC, logJitter
            lc_dictionary[instrument]['GPVector'] = np.zeros(5)
            lc_dictionary[instrument]['X'] = lc_dictionary[instrument]['X'][:,0]
            number_of_zeros = len(np.where(ferr_lc[instrument_indexes_lc[instrument]]==0.)[0])
            if number_of_zeros == len(instrument_indexes_lc[instrument]):
                lc_dictionary[instrument]['GPObject'].compute(lc_dictionary[instrument]['X'])
            else:
                lc_dictionary[instrument]['GPObject'].compute(lc_dictionary[instrument]['X'],yerr=ferr_lc[instrument_indexes_lc[instrument]])          

        if lc_dictionary[instrument]['GPType'] == 'CeleriteExpKernel':
            exp_kernel = terms.RealTerm(log_a=np.log(10.), log_c=np.log(10.))

            for GPvariable in ['sigma','timescale']:
                for pnames in priors.keys():
                    vec = pnames.split('_')
                    if (vec[0] == 'GP') and (GPvariable in vec[1]) and (instrument in vec):
                        lc_dictionary[instrument]['GP_'+GPvariable] = '_'.join(vec[2:])
            # Jitter term:
            kernel_jitter = terms.JitterTerm(np.log(100*1e-6))

            # Wrap GP object to compute likelihood:
            kernel = exp_kernel + kernel_jitter
            lc_dictionary[instrument]['GPObject'] = celerite.GP(kernel, mean=0.0)
            # Note order of GP Vector: logsigma, log(1/timescale), logJitter
            lc_dictionary[instrument]['GPVector'] = np.zeros(3)
            lc_dictionary[instrument]['X'] = lc_dictionary[instrument]['X'][:,0]
            number_of_zeros = len(np.where(ferr_lc[instrument_indexes_lc[instrument]]==0.)[0])
            if number_of_zeros == len(instrument_indexes_lc[instrument]):
                lc_dictionary[instrument]['GPObject'].compute(lc_dictionary[instrument]['X'])
            else:
                lc_dictionary[instrument]['GPObject'].compute(lc_dictionary[instrument]['X'],yerr=ferr_lc[instrument_indexes_lc[instrument]])

        if lc_dictionary[instrument]['GPType'] == 'CeleriteMatern':
            matern_kernel = terms.Matern32Term(log_sigma=np.log(10.), log_rho=np.log(10.))

            for GPvariable in ['sigma','rho']:
                for pnames in priors.keys():
                    vec = pnames.split('_')
                    if (vec[0] == 'GP') and (GPvariable in vec[1]) and (instrument in vec):
                        lc_dictionary[instrument]['GP_'+GPvariable] = '_'.join(vec[2:])
            # Jitter term:
            kernel_jitter = terms.JitterTerm(np.log(100*1e-6))

            # Wrap GP object to compute likelihood:
            kernel = matern_kernel + kernel_jitter
            lc_dictionary[instrument]['GPObject'] = celerite.GP(kernel, mean=0.0)
            # Note order of GP Vector: logsigma, log(rho), logJitter
            lc_dictionary[instrument]['GPVector'] = np.zeros(3)
            lc_dictionary[instrument]['X'] = lc_dictionary[instrument]['X'][:,0]
            number_of_zeros = len(np.where(ferr_lc[instrument_indexes_lc[instrument]]==0.)[0])
            if number_of_zeros == len(instrument_indexes_lc[instrument]):
                lc_dictionary[instrument]['GPObject'].compute(lc_dictionary[instrument]['X'])
            else:
                lc_dictionary[instrument]['GPObject'].compute(lc_dictionary[instrument]['X'],yerr=ferr_lc[instrument_indexes_lc[instrument]])

        if lc_dictionary[instrument]['GPType'] == 'CeleriteMaternExpKernel':
            matern_kernel = terms.Matern32Term(log_sigma=np.log(10.), log_rho=np.log(10.))
            exp_kernel = terms.RealTerm(log_a=np.log(10.), log_c=np.log(10.))
            for GPvariable in ['sigma','rho','timescale']:
                for pnames in priors.keys():
                    vec = pnames.split('_')
                    if (vec[0] == 'GP') and (GPvariable in vec[1]) and (instrument in vec):
                        lc_dictionary[instrument]['GP_'+GPvariable] = '_'.join(vec[2:])

            # Jitter term:
            kernel_jitter = terms.JitterTerm(np.log(100*1e-6))

            # Wrap GP object to compute likelihood:
            kernel = exp_kernel*matern_kernel + kernel_jitter
            lc_dictionary[instrument]['GPObject'] = celerite.GP(kernel, mean=0.0)
            # IMPORTANT: here first term of GP vector is log_a (our log GP_sigma), second log_c (our log 1/timescale),
            # third is log_sigma of matern (which we set to one), fourth is log_rho and fifth is the log jitter. However, 
            # we dont change the log_sigma of the matern, so this stays as zero forever and ever.
            lc_dictionary[instrument]['GPVector'] = np.zeros(5)
            lc_dictionary[instrument]['X'] = lc_dictionary[instrument]['X'][:,0]
            number_of_zeros = len(np.where(ferr_lc[instrument_indexes_lc[instrument]]==0.)[0])
            if number_of_zeros == len(instrument_indexes_lc[instrument]):
                lc_dictionary[instrument]['GPObject'].compute(lc_dictionary[instrument]['X'])
            else:
                lc_dictionary[instrument]['GPObject'].compute(lc_dictionary[instrument]['X'],yerr=ferr_lc[instrument_indexes_lc[instrument]])

        if lc_dictionary[instrument]['GPType'] == 'CeleriteSHOKernel':
            sho_kernel = terms.SHOTerm(log_S0=np.log(10.), log_Q=np.log(10.),log_omega0=np.log(10.))
            for GPvariable in ['C0','Prot','Plife']:
                for pnames in priors.keys():
                    vec = pnames.split('_')
                    if (vec[0] == 'GP') and (GPvariable in vec[1]) and (instrument in vec):
                        lc_dictionary[instrument]['GP_'+GPvariable] = '_'.join(vec[2:])
            # Jitter term:
            kernel_jitter = terms.JitterTerm(np.log(100*1e-6))

            # Wrap GP object to compute likelihood:
            kernel = sho_kernel + kernel_jitter
            lc_dictionary[instrument]['GPObject'] = celerite.GP(kernel, mean=0.0)
            # Note order of GP Vector: log_S0, log_Q, log_omega0 and log_sigma. Note, however, that we dont fit for those 
            # parameters directly.
            lc_dictionary[instrument]['GPVector'] = np.zeros(4)
            lc_dictionary[instrument]['X'] = lc_dictionary[instrument]['X'][:,0]
            number_of_zeros = len(np.where(ferr_lc[instrument_indexes_lc[instrument]]==0.)[0])
            if number_of_zeros == len(instrument_indexes_lc[instrument]):
                lc_dictionary[instrument]['GPObject'].compute(lc_dictionary[instrument]['X'])
            else:
                lc_dictionary[instrument]['GPObject'].compute(lc_dictionary[instrument]['X'],yerr=ferr_lc[instrument_indexes_lc[instrument]])            

        if lc_dictionary[instrument]['GPType'] == 'ExpSineSquaredSEKernel':
            for GPvariable in ['sigma','alpha','Gamma','Prot']:
                for pnames in priors.keys():
                    vec = pnames.split('_')
                    if (vec[0] == 'GP') and (GPvariable in vec[1]) and (instrument in vec):
                        lc_dictionary[instrument]['GP_'+GPvariable] = '_'.join(vec[2:])

            # Generate GP Base Kernel (Constant * ExpSquared * ExpSine2):
            K1 = 1.*george.kernels.ExpSquaredKernel(metric = 1.0)
            K2 = george.kernels.ExpSine2Kernel(gamma=1.0,log_period=1.0)
            lc_dictionary[instrument]['GPKernelBase'] = K1*K2
            # Generate Jitter term:
            lc_dictionary[instrument]['GPKernelJitter'] = george.modeling.ConstantModel(np.log((200.*1e-6)**2.))

            # Generate full kernel (i.e., GP plus jitter), generating full GP object:
            if george_hodlr:
                lc_dictionary[instrument]['GPObject'] = george.GP(lc_dictionary[instrument]['GPKernelBase'], mean=0.0,fit_mean=False,\
                                                        white_noise=lc_dictionary[instrument]['GPKernelJitter'],\
                                                        fit_white_noise=True,solver=george.HODLRSolver)
            else:
                lc_dictionary[instrument]['GPObject'] = george.GP(lc_dictionary[instrument]['GPKernelBase'], mean=0.0,fit_mean=False,\
                                                        white_noise=lc_dictionary[instrument]['GPKernelJitter'],\
                                                        fit_white_noise=True)
            # Create the parameter vector --- note its dim: GP_sigma (+1) + GP_alpha (+1) + GP_Gamma (+1) + GP_Prot (+1) + Jitter term (+1): 5.
            # Given how we defined the vector, first parameter of vector is jitter, second log_(GP_sigma**2), third 1./(2*alpha), fourth Gamma and fifth logProt.
            lc_dictionary[instrument]['GPVector'] = np.zeros(5)
            # Finally, compute GP object. Note we add the lightcurve uncertainties here, which are added in quadrature to the 
            # diagonal terms in the covariance matrix by george internally:
            number_of_zeros = len(np.where(ferr_lc[instrument_indexes_lc[instrument]]==0.)[0])
            if number_of_zeros == len(instrument_indexes_lc[instrument]):
                lc_dictionary[instrument]['GPObject'].compute(lc_dictionary[instrument]['X'])
            else:
                lc_dictionary[instrument]['GPObject'].compute(lc_dictionary[instrument]['X'],yerr=ferr_lc[instrument_indexes_lc[instrument]])

        if lc_dictionary[instrument]['GPType'] == 'SEKernel' \
        or lc_dictionary[instrument]['GPType'] == 'M32Kernel':
            alpha_name = 'alpha' if lc_dictionary[instrument]['GPType'] == 'SEKernel' else 'malpha'
            GPvariables = ['sigma']
            for ialpha in range(lc_dictionary[instrument]['nX']):
                GPvariables = GPvariables + [f'{alpha_name}'+str(ialpha)]
            for GPvariable in GPvariables:
                for pnames in priors.keys():
                    vec = pnames.split('_')
                    if (vec[0] == 'GP') and (GPvariable in vec[1]) and (instrument in vec):
                        lc_dictionary[instrument]['GP_'+GPvariable] = '_'.join(vec[2:])

            # Generate GPExpSquared Base Kernel:
            lc_dictionary[instrument]['GPKernelBase'] = 1.*george.kernels.ExpSquaredKernel(np.ones(lc_dictionary[instrument]['nX']),\
                                                        ndim=lc_dictionary[instrument]['nX'],\
                                                        axes=range(lc_dictionary[instrument]['nX']))
            # Generate Jitter term:
            lc_dictionary[instrument]['GPKernelJitter'] = george.modeling.ConstantModel(np.log((200.*1e-6)**2.))

            # Generate full kernel (i.e., GPExpSquared plus jitter), generating full GP object:
            if george_hodlr:
                lc_dictionary[instrument]['GPObject'] = george.GP(lc_dictionary[instrument]['GPKernelBase'], mean=0.0,fit_mean=False,\
                                                        white_noise=lc_dictionary[instrument]['GPKernelJitter'],\
                                                        fit_white_noise=True,solver=george.HODLRSolver)
            else:
                lc_dictionary[instrument]['GPObject'] = george.GP(lc_dictionary[instrument]['GPKernelBase'], mean=0.0,fit_mean=False,\
                                                        white_noise=lc_dictionary[instrument]['GPKernelJitter'],\
                                                        fit_white_noise=True)
            # Create the parameter vector --- note it equals number of external parameters plus 2: amplitude of the GP 
            # component and jitter:
            lc_dictionary[instrument]['GPVector'] = np.zeros(lc_dictionary[instrument]['X'].shape[1] + 2)
            # Finally, compute GP object. Note we add the lightcurve uncertainties here, which are added in quadrature to the 
            # diagonal terms in the covariance matrix by george internally:
            number_of_zeros = len(np.where(ferr_lc[instrument_indexes_lc[instrument]]==0.)[0])
            if number_of_zeros == len(instrument_indexes_lc[instrument]):
                lc_dictionary[instrument]['GPObject'].compute(lc_dictionary[instrument]['X'])
            else:
                lc_dictionary[instrument]['GPObject'].compute(lc_dictionary[instrument]['X'],yerr=ferr_lc[instrument_indexes_lc[instrument]])

# Extract external parameter file for RV GP:
if rveparamfile is not None:
    GPDict = utils.readeparams(rveparamfile,RV=True)
    print('\t Fitting RV GP.')
    rv_dictionary['GPDetrend'] = True
    # Detect the GP the user wants for the RVs using the priors dictionary. 
    # For this, consider the possibility that the user might want to combine the GP parameters 
    # between instruments, so iterate.
    for pnames in priors.keys():
        vec = pnames.split('_')
        if (vec[0] == 'GP') and ('alpha0' in vec[1]) and ('rv' in vec[-1].lower()):
            # Detected multi-dimensional squared-exponential GP:
            rv_dictionary['GPType'] = 'SEKernel'
            break
        if (vec[0] == 'GP') and ('malpha0' in vec[1]) and ('rv' in vec[-1].lower()):
            # Detected multi-dimensional Matern 3/2 GP:
            rv_dictionary['GPType'] = 'M32Kernel'
            break
        if (vec[0] == 'GP') and ('Gamma' in vec[1]) and ('rv' in vec[-1].lower()):
            # Detected exp-sine-squared kernel:
            rv_dictionary['GPType'] = 'ExpSineSquaredSEKernel'
            break
        if (vec[0] == 'GP') and ('timescale' in vec[1]) and ('rv' in vec[-1].lower()):
            # If already defined, then this is a multiplication of Matern and Exp kernels:
            if 'GPType' in rv_dictionary:
                rv_dictionary['GPType'] = 'CeleriteMaternExpKernel'
                break
            # Detected celerite Exp kernel:
            rv_dictionary['GPType'] = 'CeleriteExpKernel'
        if (vec[0] == 'GP') and ('rho' in vec[1]) and ('rv' in vec[-1].lower()):
            # If already defined, then this is a multiplication of Matern and Exp kernels:
            if 'GPType' in rv_dictionary:
                rv_dictionary['GPType'] = 'CeleriteMaternExpKernel'
                break
            # Detected celerite matern:
            rv_dictionary['GPType'] = 'CeleriteMatern'

        if (vec[0] == 'GP') and ('B' in vec[1]) and ('rv' in vec[-1].lower()):
            # Detected celerite quasi-periodic kernel:
            rv_dictionary['GPType'] = 'CeleriteQPKernel'
            break
        if (vec[0] == 'GP') and ('Plife' in vec[1]) and ('rv' in vec[-1].lower()):
            # Detected celerite SHO:
            rv_dictionary['GPType'] = 'CeleriteSHOKernel'
            break

    print('\t Detected ',rv_dictionary['GPType'],'for the GP')
    # Save variables, standarize them if GP is SEKernel or M32Kernel:
    rv_dictionary['X'] = GPDict['variables']
    rv_dictionary['nX'] = rv_dictionary['X'].shape[1]
    print('\t (',rv_dictionary['nX'],'external parameters)')

    if rv_dictionary['GPType'] == 'SEKernel' \
    or rv_dictionary['GPType'] == 'M32Kernel':
        kernel_name = 'SEKernel' if rv_dictionary['GPType'] == 'SEKernel' else 'M32Kernel'
        for i in range(rv_dictionary['nX']):
            rv_dictionary['X'][:,i] = (rv_dictionary['X'][:,i] - \
                                       np.mean(rv_dictionary['X'][:,i]))/\
                                       np.sqrt(np.var(rv_dictionary['X'][:,i]))

        print(f'\t Not yet supported {kernel_name} for RVs (do you really need it?).')
        import sys
        sys.exit()

    if rv_dictionary['GPType'] == 'CeleriteQPKernel':
        # Now that we know the type of GP, we extract the "instrument" corresponding to each GP 
        # parameter. For example, for the ExpSineSquaredKernel, it might happen the user wants 
        # to have a common GP_Prot parameter shared along many instruments, e.g., GP_Prot_TESS_K2_RV,
        # which means the user wants a common Prot for TESS and K2 photometry, and also for the RVs. However, 
        # the same GP might have a different Gamma, i.e., there might be a GP_Gamma_TESS, GP_Gamma_K2 and GP_Gamma_RV.
        # The idea here is to, e.g., in the case of TESS photometry, gather lc_dictionary[instrument]['GP_Prot'] = 
        # 'TESS_K2_RV', and lc_dictionary[instrument]['GP_Gamma'] = 'TESS':
        for GPvariable in ['B','C','L','Prot']:
            for pnames in priors.keys():
                vec = pnames.split('_')
                if (vec[0] == 'GP') and (GPvariable in vec[1]) and ('rv' in vec[-1].lower()):
                    rv_dictionary['GP_'+GPvariable] = '_'.join(vec[2:])

        #for instrument in inames_rv:
        rot_kernel = terms.TermSum(RotationTerm(log_amp=np.log(10.),\
                                                log_timescale=np.log(10.0),\
                                                log_period=np.log(3.0),\
                                                log_factor=np.log(1.0)))
        # Jitter term; dont add it, jitters will be added directly on the log-like (see Espinoza+2018).
        #kernel_jitter = terms.JitterTerm(np.log(100*1e-6))

        # Wrap GP object to compute likelihood:
        kernel = rot_kernel #+ kernel_jitter
        rv_dictionary['GPObject'] = celerite.GP(kernel, mean=0.0)
        # Note order of GP Vector: logB, logL, logProt, logC, logJitter
        rv_dictionary['GPVector'] = np.zeros(4)
        rv_dictionary['X'] = rv_dictionary['X'][:,0] 
        rv_dictionary['GPObject'].compute(rv_dictionary['X'],yerr=rverr_rv)
    if rv_dictionary['GPType'] == 'CeleriteSHOKernel':
        # Now that we know the type of GP, we extract the "instrument" corresponding to each GP 
        # parameter. For example, for the ExpSineSquaredKernel, it might happen the user wants 
        # to have a common GP_Prot parameter shared along many instruments, e.g., GP_Prot_TESS_K2_RV,
        # which means the user wants a common Prot for TESS and K2 photometry, and also for the RVs. However, 
        # the same GP might have a different Gamma, i.e., there might be a GP_Gamma_TESS, GP_Gamma_K2 and GP_Gamma_RV.
        # The idea here is to, e.g., in the case of TESS photometry, gather lc_dictionary[instrument]['GP_Prot'] = 
        # 'TESS_K2_RV', and lc_dictionary[instrument]['GP_Gamma'] = 'TESS':
        for GPvariable in ['C0','Prot','Plife']:
            for pnames in priors.keys():
                vec = pnames.split('_')
                if (vec[0] == 'GP') and (GPvariable in vec[1]) and ('rv' in vec[-1].lower()):
                    rv_dictionary['GP_'+GPvariable] = '_'.join(vec[2:])

        #for instrument in inames_rv:
        sho_kernel = terms.SHOTerm(log_S0=np.log(10.), log_Q=np.log(10.),log_omega0=np.log(10.))
        # Jitter term; dont add it, jitters will be added directly on the log-like (see Espinoza+2018).
        #kernel_jitter = terms.JitterTerm(np.log(100*1e-6))

        # Wrap GP object to compute likelihood:
        kernel = sho_kernel #+ kernel_jitter
        rv_dictionary['GPObject'] = celerite.GP(kernel, mean=0.0)
        # Note order of GP Vector: logS0, logQ, logomega0
        rv_dictionary['GPVector'] = np.zeros(3)
        rv_dictionary['X'] = rv_dictionary['X'][:,0]
        rv_dictionary['GPObject'].compute(rv_dictionary['X'],yerr=rverr_rv)
    if rv_dictionary['GPType'] == 'ExpSineSquaredSEKernel':
        for GPvariable in ['sigma','alpha','Gamma','Prot']:
            for pnames in priors.keys():
                vec = pnames.split('_')
                if (vec[0] == 'GP') and (GPvariable in vec[1]) and ('rv' in vec[-1].lower()):
                    rv_dictionary['GP_'+GPvariable] = '_'.join(vec[2:])

        #for instrument in inames_rv:
        # Generate GP Base Kernel (Constant * ExpSquared * ExpSine2):
        K1 = 1.*george.kernels.ExpSquaredKernel(metric = 1.0)
        K2 = george.kernels.ExpSine2Kernel(gamma=1.0,log_period=1.0)

        # Generate kernel part:
        rv_dictionary['GPKernelBase'] = K1*K2
        # Generate Jitter term:
        #rv_dictionary['GPKernelJitter'] = george.modeling.ConstantModel(np.log((200.*1e-6)**2.))

        # Generate full kernel (i.e., GP plus jitter), generating full GP object:
        rv_dictionary['GPObject'] = george.GP(rv_dictionary['GPKernelBase'], mean=0.0,fit_mean=False,\
                                                          fit_white_noise=False)#,solver=george.HODLRSolver, seed=42)
        # Create the parameter vector --- note its dim: GP_sigma (+1) + GP_alpha (+1) + GP_Gamma (+1) + GP_Prot (+1): 4.
        # Given how we defined the vector, first parameter of vector log_(GP_sigma**2), 2 1./(2*alpha), 3 Gamma and 4 logProt.
        rv_dictionary['GPVector'] = np.zeros(4)
        # Finally, compute GP object. 
        rv_dictionary['GPObject'].compute(rv_dictionary['X'],yerr=rverr_rv)

    if rv_dictionary['GPType'] == 'CeleriteExpKernel':
        exp_kernel = terms.RealTerm(log_a=np.log(10.), log_c=np.log(10.))

        for GPvariable in ['sigma','timescale']:
            for pnames in priors.keys():
                vec = pnames.split('_')
                if (vec[0] == 'GP') and (GPvariable in vec[1]) and ('rv' in vec[-1].lower()):
                   rv_dictionary['GP_'+GPvariable] = '_'.join(vec[2:])

        # Wrap GP object to compute likelihood:
        kernel = exp_kernel
        rv_dictionary['GPObject'] = celerite.GP(kernel, mean=0.0)
        # Note order of GP Vector: logsigma, log(1/timescale)
        rv_dictionary['GPVector'] = np.zeros(2)
        rv_dictionary['X'] = rv_dictionary['X'][:,0]
        rv_dictionary['GPObject'].compute(rv_dictionary['X'],yerr=rverr_rv)

    if rv_dictionary['GPType'] == 'CeleriteMatern':
        matern_kernel = terms.Matern32Term(log_sigma=np.log(10.), log_rho=np.log(10.))

        for GPvariable in ['sigma','rho']:
           for pnames in priors.keys():
                vec = pnames.split('_')
                if (vec[0] == 'GP') and (GPvariable in vec[1]) and ('rv' in vec[-1].lower()):
                    rv_dictionary['GP_'+GPvariable] = '_'.join(vec[2:])
        # Wrap GP object to compute likelihood:
        kernel = matern_kernel# + kernel_jitter
        rv_dictionary['GPObject'] = celerite.GP(kernel, mean=0.0)
        # Note order of GP Vector: logsigma, log(1/timescale)
        rv_dictionary['GPVector'] = np.zeros(2)
        rv_dictionary['X'] = rv_dictionary['X'][:,0]
        rv_dictionary['GPObject'].compute(rv_dictionary['X'],yerr=rverr_rv)

    if rv_dictionary['GPType'] == 'CeleriteMaternExpKernel':
        matern_kernel = terms.Matern32Term(log_sigma=np.log(10.), log_rho=np.log(10.))
        exp_kernel = terms.RealTerm(log_a=np.log(10.), log_c=np.log(10.))
        for GPvariable in ['sigma','rho','timescale']:
            for pnames in priors.keys():
                vec = pnames.split('_')
                if (vec[0] == 'GP') and (GPvariable in vec[1]) and ('rv' in vec[-1].lower()):
                    rv_dictionary['GP_'+GPvariable] = '_'.join(vec[2:])

        # Wrap GP object to compute likelihood:
        kernel = exp_kernel*matern_kernel# + kernel_jitter
        rv_dictionary['GPObject'] = celerite.GP(kernel, mean=0.0)
        # IMPORTANT: here first term of GP vector is log_a (our log GP_sigma), second log_c (our log 1/timescale),
        # third is log_sigma of matern (which we set to sigma = one, so log_sigma = 0), fourth is log_rho. However, 
        # we dont change the log_sigma of the matern, so this stays as zero forever and ever.
        rv_dictionary['GPVector'] = np.zeros(4)
        rv_dictionary['X'] = rv_dictionary['X'][:,0]
        rv_dictionary['GPObject'].compute(rv_dictionary['X'],yerr=rverr_rv)

# Other inputs like, e.g., nlive points:
n_live_points = int(args.nlive)
# Number of simulations:
n_sims = int(args.nsims)
# Number of oversamples in rvs:
noversample = int(args.noversample)

# Define transit-related functions:
def reverse_ld_coeffs(ld_law, q1, q2):
    if ld_law == 'quadratic':
        coeff1 = 2.*np.sqrt(q1)*q2
        coeff2 = np.sqrt(q1)*(1.-2.*q2)
    elif ld_law=='squareroot':
        coeff1 = np.sqrt(q1)*(1.-2.*q2)
        coeff2 = 2.*np.sqrt(q1)*q2
    elif ld_law=='logarithmic':
        coeff1 = 1.-np.sqrt(q1)*q2
        coeff2 = 1.-np.sqrt(q1)
    elif ld_law == 'linear':
        return q1,q2
    return coeff1,coeff2

def init_batman(t,law, n_ss=None, exptime_ss=None):
    """  
    This function initializes the batman code.
    """
    params = batman.TransitParams()
    params.t0 = 0.
    params.per = 1.
    params.rp = 0.1
    params.a = 15.
    params.inc = 87.
    params.ecc = 0.
    params.w = 90.
    if law == 'linear':
        params.u = [0.5]
    else:
        params.u = [0.1,0.3]
    params.limb_dark = law
    if n_ss is None or exptime_ss is None:
        m = batman.TransitModel(params, t)
    else: 
        m = batman.TransitModel(params, t, supersample_factor=n_ss, exp_time=exptime_ss)
    return params,m

def get_transit_model(t,t0,P,p,a,inc,q1,q2,ld_law,n_ss,exptime_ss):
    params,m = init_batman(t,law=ld_law,n_ss=n_ss,exptime_ss=exptime_ss)
    coeff1,coeff2 = reverse_ld_coeffs(ld_law, q1, q2)
    params.t0 = t0
    params.per = P
    params.rp = p
    params.a = a
    params.inc = inc
    if ld_law == 'linear':
        params.u = [coeff1]
    else:
        params.u = [coeff1,coeff2]
    return m.light_curve(params)

def init_radvel(nplanets=1):
    return radvel.model.Parameters(nplanets,basis='per tc e w k')

def transform_prior(val,pinfo):
    if pinfo['type'] == 'uniform':
        return utils.transform_uniform(val,pinfo['value'][0],pinfo['value'][1])
    if pinfo['type'] == 'normal':
        return utils.transform_normal(val,pinfo['value'][0],pinfo['value'][1])
    if pinfo['type'] == 'truncatednormal':
        return utils.transform_truncated_normal(val,pinfo['value'][0],pinfo['value'][1],a=pinfo['value'][2],b=pinfo['value'][3])
    if pinfo['type'] == 'jeffreys':
        return utils.transform_loguniform(val,pinfo['value'][0],pinfo['value'][1])
    if pinfo['type'] == 'beta':
        return utils.transform_beta(val,pinfo['value'][0],pinfo['value'][1])
    if pinfo['type'] == 'exponential':
        return utils.transform_exponential(val)

if lcfilename is not None:
    # Initialize batman for each different lightcurve:
    for instrument in lc_dictionary.keys():
        if lc_dictionary[instrument]['resampling']:
            lc_dictionary[instrument]['params'],lc_dictionary[instrument]['m'] = init_batman(t_lc[instrument_indexes_lc[instrument]], \
                                                           law=lc_dictionary[instrument]['ldlaw'], \
                                                           n_ss=lc_dictionary[instrument]['nresampling'],\
                                                           exptime_ss=lc_dictionary[instrument]['exptimeresampling'])  
        else:
            lc_dictionary[instrument]['params'],lc_dictionary[instrument]['m'] = init_batman(t_lc[instrument_indexes_lc[instrument]], \
                                                           law=lc_dictionary[instrument]['ldlaw'])

if rvfilename is not None:
    # Initialize radvel:
    radvel_params = init_radvel(nplanets=n_rv)

# Define gaussian log-likelihood:
log2pi = np.log(2.*np.pi)
def gaussian_log_likelihood(residuals,errors,jitter):
    taus = 1./(errors**2 + jitter**2)
    return -0.5*(len(residuals)*log2pi+np.sum(np.log(1./taus)+taus*(residuals**2)))

# Now define  priors and log-likelihood. Note here the idea is that any fixed parameters don't 
# receive prior numbering. 
transformed_priors = np.zeros(n_params)
def prior(cube, ndim=None, nparams=None):
    pcounter = 0
    for pname in priors.keys(): 
         if priors[pname]['type'] != 'fixed':
             if use_dynesty:
                 transformed_priors[pcounter] = transform_prior(cube[pcounter],priors[pname])
             else:
                 cube[pcounter] = transform_prior(cube[pcounter],priors[pname])
             pcounter += 1
    if use_dynesty:
        return transformed_priors
        

if lcfilename is not None:
    lcones = np.ones(len(t_lc))

# Gravitational constant (for stellar density):
G = 6.67408e-11 # mks

# Maximum eccentricity limit:
ecclim = np.double(args.ecclim)

# Save corresponding name of LD input instrument to the respective instrument (this is to link LD coefficients between 
# different "instruments"):
if lcfilename is not None:
    ld_iname = {}
    for pname in priors.keys():
        if pname[0:2] == 'q1':
            vec = pname.split('_')
            if len(vec)>2:
                for i in range(len(inames_lc)):
                    if inames_lc[i] in vec:
                        ld_iname[inames_lc[i]] = '_'.join(vec[1:])
            else:
                ld_iname[vec[1]] = vec[1]
    #priors['q1_'+instrument]

def loglike(cube, ndim=None, nparams=None):
    # Evaluate the log-likelihood. For this, first extract all inputs:
    pcounter = 0
    for pname in priors.keys():
        if priors[pname]['type'] != 'fixed':
            priors[pname]['cvalue'] = cube[pcounter]
            pcounter += 1
    # Photometric terms first:
    if lcfilename is not None:
        # Before everything continues, make sure periods are chronologically ordered (this is to avoid multiple modes due to 
        # periods "jumping" between planet numbering):
        for n in range(n_transit):
            i = numbering_transit[n]
            if n == 0:
                cP = priors['P_p'+str(i)]['cvalue']
            else:
                if cP < priors['P_p'+str(i)]['cvalue']:
                    cP = priors['P_p'+str(i)]['cvalue']
                else:
                    return -1e101
            if cP < 0.:
                return -1e101
        # Generate lightcurve models for each instrument:
        lcmodel = np.copy(lcones)
        for instrument in inames_lc:
            # For each transit model iterate through the 
            # number of planets, multiplying their transit models:
            if lc_dictionary[instrument]['TransitFit']:
                for n in range(n_transit):
                    i = numbering_transit[n]
                    if lc_dictionary[instrument]['ldlaw'] != 'linear':
                        coeff1,coeff2 = reverse_ld_coeffs(lc_dictionary[instrument]['ldlaw'],priors['q1_'+ld_iname[instrument]]['cvalue'],\
                                        priors['q2_'+ld_iname[instrument]]['cvalue'])
                        lc_dictionary[instrument]['params'].u = [coeff1,coeff2]
                    else:
                        lc_dictionary[instrument]['params'].u = [priors['q1_'+ld_iname[instrument]]['cvalue']]

                    if efficient_bp[n]:
                        if not fitrho:
                            a,r1,r2,t0,P = priors['a_p'+str(i)]['cvalue'],priors['r1_p'+str(i)]['cvalue'],\
                                           priors['r2_p'+str(i)]['cvalue'], priors['t0_p'+str(i)]['cvalue'], \
                                           priors['P_p'+str(i)]['cvalue']
                        else:
                            rho,r1,r2,t0,P = priors['rho']['cvalue'],priors['r1_p'+str(i)]['cvalue'],\
                                             priors['r2_p'+str(i)]['cvalue'], priors['t0_p'+str(i)]['cvalue'], \
                                             priors['P_p'+str(i)]['cvalue']
                            a = ((rho*G*((P*24.*3600.)**2))/(3.*np.pi))**(1./3.)
                        if r1 > Ar:
                            b,p = (1+pl)*(1. + (r1-1.)/(1.-Ar)),\
                                  (1-r2)*pl + r2*pu
                        else:
                            b,p = (1. + pl) + np.sqrt(r1/Ar)*r2*(pu-pl),\
                                  pu + (pl-pu)*np.sqrt(r1/Ar)*(1.-r2)
                    else:
                        if not fitrho:
                            a,b,p,t0,P = priors['a_p'+str(i)]['cvalue'],priors['b_p'+str(i)]['cvalue'],\
                                         priors['p_p'+str(i)]['cvalue'], priors['t0_p'+str(i)]['cvalue'], \
                                         priors['P_p'+str(i)]['cvalue']
                        else:
                            rho,b,p,t0,P = priors['rho']['cvalue'],priors['b_p'+str(i)]['cvalue'],\
                                         priors['p_p'+str(i)]['cvalue'], priors['t0_p'+str(i)]['cvalue'], \
                                         priors['P_p'+str(i)]['cvalue']
                            a = ((rho*G*((P*24.*3600.)**2))/(3.*np.pi))**(1./3.)
                    if ecc_parametrization['transit'][i] == 0:
                        ecc,omega = priors['ecc_p'+str(i)]['cvalue'],priors['omega_p'+str(i)]['cvalue']
                    elif ecc_parametrization['transit'][i] == 1:
                        ecc = np.sqrt(priors['ecosomega_p'+str(i)]['cvalue']**2+priors['esinomega_p'+str(i)]['cvalue']**2)
                        omega = np.arctan2(priors['esinomega_p'+str(i)]['cvalue'],priors['ecosomega_p'+str(i)]['cvalue'])*180./np.pi
                    else:
                        ecc = priors['secosomega_p'+str(i)]['cvalue']**2+priors['sesinomega_p'+str(i)]['cvalue']**2
                        omega = np.arctan2(priors['sesinomega_p'+str(i)]['cvalue'],priors['secosomega_p'+str(i)]['cvalue'])*180./np.pi

                    if ecc>ecclim:
                        return -1e101
                    else:
                        ecc_factor = (1. + ecc*np.sin(omega * np.pi/180.))/(1. - ecc**2)
                        inc_inv_factor = (b/a)*ecc_factor
                        if not (b>1.+p or inc_inv_factor >=1.):
                            inc = np.arccos(inc_inv_factor)*180./np.pi
                            lc_dictionary[instrument]['params'].t0 = t0 
                            lc_dictionary[instrument]['params'].per = P
                            lc_dictionary[instrument]['params'].rp = p
                            lc_dictionary[instrument]['params'].a = a
                            lc_dictionary[instrument]['params'].inc = inc
                            lc_dictionary[instrument]['params'].ecc = ecc
                            lc_dictionary[instrument]['params'].w = omega
                            lcmodel[instrument_indexes_lc[instrument]] = lcmodel[instrument_indexes_lc[instrument]]*lc_dictionary[instrument]['m'].light_curve(lc_dictionary[instrument]['params'])
                            #iidx = np.where(lcmodel!=1.)[0]
                        else:
                            return -1e101 
    # Compute combined log-likelihood for lightcurve data:
    log_likelihood = 0.0
    for instrument in inames_lc:
        inst_model = (lcmodel[instrument_indexes_lc[instrument]]*priors['mdilution_'+instrument]['cvalue'] \
                     + (1. - priors['mdilution_'+instrument]['cvalue']))*\
                     (1./(1. + priors['mdilution_'+instrument]['cvalue']*priors['mflux_'+instrument]['cvalue'])) 

        # If inputs for linear model, add it to the instrumental model:
        if lm_boolean[instrument]:
            for idx_X in range(lm_arguments[instrument].shape[1]):
                inst_model = inst_model + lm_arguments[instrument][:,idx_X]*priors['theta'+str(idx_X)+'_'+instrument]['cvalue']
        residuals = f_lc[instrument_indexes_lc[instrument]] - inst_model

        # If not GP Detrend (which means no external parameters given for the instrument), 
        # return gaussian log-likelihood:
        if not lc_dictionary[instrument]['GPDetrend']:
            log_likelihood += gaussian_log_likelihood(residuals,\
                                                      ferr_lc[instrument_indexes_lc[instrument]],\
                                                      priors['sigma_w_'+instrument]['cvalue']*1e-6)
        else:
            if lc_dictionary[instrument]['GPType'] == 'SEKernel' \
            or lc_dictionary[instrument]['GPType'] == 'M32Kernel':
                alpha_name = 'alpha' if lc_dictionary[instrument]['GPType'] == 'SEKernel' else 'malpha'
                # Save the log(variance) of the jitter term on the current GP vector:
                lc_dictionary[instrument]['GPVector'][0] = np.log((priors['sigma_w_'+instrument]['cvalue']*1e-6)**2.)
                # Save pooled variance of the GP process:
                lc_dictionary[instrument]['GPVector'][1] = np.log((priors['GP_sigma_'+lc_dictionary[instrument]['GP_sigma']]['cvalue']*1e-6)**2.)
                # Now save (log of) coefficients of each GP term:
                for i in range(lc_dictionary[instrument]['nX']):
                    lc_dictionary[instrument]['GPVector'][2+i] = np.log(
                            1./priors[f'GP_{alpha_name}'+str(i)+'_'+lc_dictionary[instrument]['GP_{alpha_name}'+str(i)]]['cvalue']
                    )
            if lc_dictionary[instrument]['GPType'] == 'ExpSineSquaredSEKernel':
                # Save the log(variance) of the jitter term on the current GP vector:
                lc_dictionary[instrument]['GPVector'][0] = np.log((priors['sigma_w_'+instrument]['cvalue']*1e-6)**2.)
                # Save pooled log(variance) of the GP process:
                lc_dictionary[instrument]['GPVector'][1] = np.log((priors['GP_sigma_'+lc_dictionary[instrument]['GP_sigma']]['cvalue']*1e-6)**2.)
                # Save log-alpha:
                lc_dictionary[instrument]['GPVector'][2] = np.log(1./priors['GP_alpha_'+lc_dictionary[instrument]['GP_alpha']]['cvalue'])
                # Save the Gamma:
                lc_dictionary[instrument]['GPVector'][3] = priors['GP_Gamma_'+lc_dictionary[instrument]['GP_Gamma']]['cvalue']
                # And save log(Prot):
                lc_dictionary[instrument]['GPVector'][4] = np.log(priors['GP_Prot_'+lc_dictionary[instrument]['GP_Prot']]['cvalue'])
            if lc_dictionary[instrument]['GPType'] == 'CeleriteQPKernel':
                # Note order of GP Vector: logB, logL, logProt, logC, logJitter                  
                # Save the log(B) term of the current GP vector:
                lc_dictionary[instrument]['GPVector'][0] = np.log(priors['GP_B_'+lc_dictionary[instrument]['GP_B']]['cvalue'])
                # Save the log(L) term of the current GP vector:
                lc_dictionary[instrument]['GPVector'][1] = np.log(priors['GP_L_'+lc_dictionary[instrument]['GP_L']]['cvalue'])
                # Save the log(Prot) term of the current GP vector:
                lc_dictionary[instrument]['GPVector'][2] = np.log(priors['GP_Prot_'+lc_dictionary[instrument]['GP_Prot']]['cvalue'])
                # Save the log(C) term of the current GP vector:
                lc_dictionary[instrument]['GPVector'][3] = np.log(priors['GP_C_'+lc_dictionary[instrument]['GP_C']]['cvalue'])
                # Save the log(jitter) term of the current GP vector:
                lc_dictionary[instrument]['GPVector'][4] = np.log(priors['sigma_w_'+instrument]['cvalue']*1e-6)
            if lc_dictionary[instrument]['GPType'] == 'CeleriteExpKernel':
                # Save the log(sigma_GP) term of the current GP vector:
                lc_dictionary[instrument]['GPVector'][0] = np.log((priors['GP_sigma_'+lc_dictionary[instrument]['GP_sigma']]['cvalue'])*1e-6)
                # Save the log(1/timescale) term of the current GP vector:
                lc_dictionary[instrument]['GPVector'][1] = np.log(1./priors['GP_timescale_'+lc_dictionary[instrument]['GP_timescale']]['cvalue'])
                # Save the log(jitte) term of the current GP vector:
                lc_dictionary[instrument]['GPVector'][2] = np.log(priors['sigma_w_'+instrument]['cvalue']*1e-6)
            if lc_dictionary[instrument]['GPType'] == 'CeleriteMatern':
                # Save the log(sigma_GP) term of the current GP vector:
                lc_dictionary[instrument]['GPVector'][0] = np.log((priors['GP_sigma_'+lc_dictionary[instrument]['GP_sigma']]['cvalue'])*1e-6)
                # Save the log(1/timescale) term of the current GP vector:
                lc_dictionary[instrument]['GPVector'][1] = np.log(priors['GP_rho_'+lc_dictionary[instrument]['GP_rho']]['cvalue'])
                # Save the log(jitte) term of the current GP vector:
                lc_dictionary[instrument]['GPVector'][2] = np.log(priors['sigma_w_'+instrument]['cvalue']*1e-6)
            if lc_dictionary[instrument]['GPType'] == 'CeleriteMaternExpKernel':
                # NOTE: We leave index 2 without value ON PURPOSE: the idea is that here, that is always 0 (because this defines the log(sigma) of the 
                # matern kernel in the multiplication, which we set to 1).
                # Save the log(sigma_GP) term of the current GP vector:
                lc_dictionary[instrument]['GPVector'][0] = np.log((priors['GP_sigma_'+lc_dictionary[instrument]['GP_sigma']]['cvalue'])*1e-6)
                # Save the log(1/timescale) term of the current GP vector:
                lc_dictionary[instrument]['GPVector'][1] = np.log(1./priors['GP_timescale_'+lc_dictionary[instrument]['GP_timescale']]['cvalue'])
                # Save the log(1/timescale) term of the current GP vector:
                lc_dictionary[instrument]['GPVector'][3] = np.log(priors['GP_rho_'+lc_dictionary[instrument]['GP_rho']]['cvalue'])
                # Save the log(jitte) term of the current GP vector:
                lc_dictionary[instrument]['GPVector'][4] = np.log(priors['sigma_w_'+instrument]['cvalue']*1e-6)                
            if lc_dictionary[instrument]['GPType'] == 'CeleriteSHOKernel':
                # Transform C0, Prot and Plife to S0, Q and omega0. Note we assume C0 comes in ppm:
                C0,Prot,Plife = priors['GP_C0_'+lc_dictionary[instrument]['GP_C0']]['cvalue']*1e-6,priors['GP_Prot_'+lc_dictionary[instrument]['GP_Prot']]['cvalue'],\
                                priors['GP_Plife_'+lc_dictionary[instrument]['GP_Plife']]['cvalue']
                S0 = (C0*(Prot**2))/(2. * (np.pi**2) * Plife)
                omega0 = 2. * np.pi/Prot
                Q = np.pi*Plife/Prot
                if Q > 0.5:
                    return -1e101
                lc_dictionary[instrument]['GPVector'][0] = np.log(S0)
                # Save the log(1/timescale) term of the current GP vector:
                lc_dictionary[instrument]['GPVector'][1] = np.log(Q)
                # Save the log(1/timescale) term of the current GP vector:
                lc_dictionary[instrument]['GPVector'][2] = np.log(omega0)
                # Save the log(jitte) term of the current GP vector:
                lc_dictionary[instrument]['GPVector'][3] = np.log(priors['sigma_w_'+instrument]['cvalue']*1e-6)

            lc_dictionary[instrument]['GPObject'].set_parameter_vector(lc_dictionary[instrument]['GPVector'])
            log_likelihood += lc_dictionary[instrument]['GPObject'].log_likelihood(residuals)

    # Add the likelihood term for the stellar density:
    if (stellar_density) and (not fitrho):
        model = ((3.*np.pi)/(G*(P*(24.*3600.0))**2))*(a)**3
        log_likelihood += - 0.5*(np.log(2*np.pi) + 2.*np.log(sd_sigma) + ((model-sd_mean)/sd_sigma)**2)

    # Now RVs:
    if rvfilename is not None:
        # Before everything continues, make sure periods are chronologically ordered (this is to avoid multiple modes due to 
        # periods "jumping" between planet numbering):
        for n in range(n_rv):
            i = numbering_rv[n]
            if n == 0:
                cP = priors['P_p'+str(i)]['cvalue']
            else:
                if cP < priors['P_p'+str(i)]['cvalue']:
                    cP = priors['P_p'+str(i)]['cvalue']
                else:
                    return -1e101
            if cP < 0.:
                return -1e101

        # Now prepare RV model. Note here it is extremely important to see the distinction between n and numbering_rv[n]. n is 
        # the numbering of the planet among all the RV planets, and is what is used to tell radvel which planet is which. However, 
        # numbering_rv[n] is the planet number within the juliet framework, which extracts the parameters from the prior file.
        for n in range(n_rv):
            i = numbering_rv[n]
            K,t0,P = priors['K_p'+str(i)]['cvalue'],\
                     priors['t0_p'+str(i)]['cvalue'],\
                     priors['P_p'+str(i)]['cvalue']

            if ecc_parametrization['rv'][i] == 0:
                ecc,omega = priors['ecc_p'+str(i)]['cvalue'],priors['omega_p'+str(i)]['cvalue']*np.pi/180.
            elif ecc_parametrization['rv'][i] == 1:
                ecc = np.sqrt(priors['ecosomega_p'+str(i)]['cvalue']**2+priors['esinomega_p'+str(i)]['cvalue']**2)
                omega = np.arctan2(priors['esinomega_p'+str(i)]['cvalue'],priors['ecosomega_p'+str(i)]['cvalue'])
            else:
                ecc = priors['secosomega_p'+str(i)]['cvalue']**2+priors['sesinomega_p'+str(i)]['cvalue']**2
                omega = np.arctan2(priors['sesinomega_p'+str(i)]['cvalue'],priors['secosomega_p'+str(i)]['cvalue'])

            # Reject samples giving unphysical eccentricities:
            if ecc > 1.:
                return -1e101
            radvel_params['per'+str(n+1)] = radvel.Parameter(value=P)
            radvel_params['tc'+str(n+1)] = radvel.Parameter(value=t0)
            radvel_params['w'+str(n+1)] = radvel.Parameter(value=omega)
            radvel_params['e'+str(n+1)] = radvel.Parameter(value=ecc)
            radvel_params['k'+str(n+1)] = radvel.Parameter(value=K)
        # Compute RV model:
        rvmodel = radvel.model.RVModel(radvel_params).__call__(t_rv)

    # If added linear trend, compute its term:
    if fitrvline:
        # Line is of the form RV = aline*t + bline:
        aline = priors['rv_slope']['cvalue']
        bline = -priors['rv_tzero']['cvalue']*aline
        # Add line to the RV model:
        rvmodel += aline*t_rv + bline
    if fitrvquad:
        # Quad is of the form RV = q*t**2 + aline*t + bline:
        qquad = priors['rv_quad']['cvalue']
        aline = priors['rv_slope']['cvalue']
        bline = -priors['rv_tzero']['cvalue']*aline - (priors['rv_tzero']['cvalue']**2)*qquad
        # Add line to the RV model:
        rvmodel += qquad*(t_rv**2) + aline*t_rv + bline

    # Compute log-likelihood for each RV instrument separately in case of white-noise:
    for instrument in inames_rv:
        # If no external parameters, return added gaussian log-likelihood:
        if rveparamfile is None:
            residuals = rv_rv[instrument_indexes_rv[instrument]] - \
                        (rvmodel[instrument_indexes_rv[instrument]] + priors['mu_'+instrument]['cvalue'])
            log_likelihood += gaussian_log_likelihood(residuals,\
                              rverr_rv[instrument_indexes_rv[instrument]],\
                              priors['sigma_w_rv_'+instrument]['cvalue'])
        else:
           rvresiduals[instrument_indexes_rv[instrument]] = rv_rv[instrument_indexes_rv[instrument]] - \
                                                            (rvmodel[instrument_indexes_rv[instrument]] + priors['mu_'+instrument]['cvalue'])
           rvresiduals_err[instrument_indexes_rv[instrument]] = np.sqrt(rverr_rv[instrument_indexes_rv[instrument]]**2 + \
                                                                        priors['sigma_w_rv_'+instrument]['cvalue']**2)
    if rveparamfile is not None:
        if rv_dictionary['GPType'] == 'ExpSineSquaredSEKernel':
            # Save pooled log(variance) of the GP process:
            rv_dictionary['GPVector'][0] = np.log((priors['GP_sigma_'+rv_dictionary['GP_sigma']]['cvalue'])**2.)
            # Save log-alpha:
            rv_dictionary['GPVector'][1] = np.log(1./priors['GP_alpha_'+rv_dictionary['GP_alpha']]['cvalue'])
            # Save the Gamma:
            rv_dictionary['GPVector'][2] = priors['GP_Gamma_'+rv_dictionary['GP_Gamma']]['cvalue']
            # And save log(Prot):
            rv_dictionary['GPVector'][3] = np.log(priors['GP_Prot_'+rv_dictionary['GP_Prot']]['cvalue'])
        if rv_dictionary['GPType'] == 'CeleriteExpKernel':
            # Note order of GP Vector: logB, logL, logProt, logC, logJitter                  
            # Save the log(sigma) term of the current GP vector:
            rv_dictionary['GPVector'][0] = np.log(priors['GP_sigma_'+rv_dictionary['GP_sigma']]['cvalue'])
            # Save the log(timescale) term of the current GP vector:
            rv_dictionary['GPVector'][1] = np.log(priors['GP_timescale_'+rv_dictionary['GP_timescale']]['cvalue'])
        if rv_dictionary['GPType'] == 'CeleriteMatern':
            # Note order of GP Vector: logB, logL, logProt, logC, logJitter                  
            # Save the log(sigma) term of the current GP vector:
            rv_dictionary['GPVector'][0] = np.log(priors['GP_sigma_'+rv_dictionary['GP_sigma']]['cvalue'])
            # Save the log(rho) term of the current GP vector:
            rv_dictionary['GPVector'][1] = np.log(priors['GP_rho_'+rv_dictionary['GP_rho']]['cvalue'])
        if rv_dictionary['GPType'] == 'CeleriteMaternExpKernel':
            # NOTE: We leave index 2 without value ON PURPOSE: the idea is that here, that is always 0 (because this defines the log(sigma) of the 
            # matern kernel in the multiplication, which we set to 1).
            # Save the log(sigma_GP) term of the current GP vector:
            rv_dictionary['GPVector'][0] = np.log(priors['GP_sigma_'+rv_dictionary['GP_sigma']]['cvalue'])
            # Save the log(timescale) term of the current GP vector:
            rv_dictionary['GPVector'][1] = np.log(priors['GP_timescale_'+rv_dictionary['GP_timescale']]['cvalue'])
            # Save the log(rho) term of the current GP vector:
            rv_dictionary['GPVector'][3] = np.log(priors['GP_rho_'+rv_dictionary['GP_rho']]['cvalue'])
        if rv_dictionary['GPType'] == 'CeleriteQPKernel':
            # Note order of GP Vector: logB, logL, logProt, logC, logJitter                  
            # Save the log(B) term of the current GP vector:
            rv_dictionary['GPVector'][0] = np.log(priors['GP_B_'+rv_dictionary['GP_B']]['cvalue'])
            # Save the log(L) term of the current GP vector:
            rv_dictionary['GPVector'][1] = np.log(priors['GP_L_'+rv_dictionary['GP_L']]['cvalue'])
            # Save the log(L) term of the current GP vector:
            rv_dictionary['GPVector'][2] = np.log(priors['GP_Prot_'+rv_dictionary['GP_Prot']]['cvalue'])
            # Save the log(L) term of the current GP vector:
            rv_dictionary['GPVector'][3] = np.log(priors['GP_C_'+rv_dictionary['GP_C']]['cvalue'])
        if rv_dictionary['GPType'] == 'CeleriteSHOKernel':
            # Transform C0, Prot and Plife to S0, Q and omega0. Note we assume C0 comes in same units as RVs:
            C0,Prot,Plife = priors['GP_C0_'+rv_dictionary['GP_C0']]['cvalue'],priors['GP_Prot_'+rv_dictionary['GP_Prot']]['cvalue'],\
                            priors['GP_Plife_'+rv_dictionary['GP_Plife']]['cvalue']
            S0 = (C0*(Prot**2))/(2. * (np.pi**2) * Plife)
            omega0 = 2. * np.pi/Prot
            Q = np.pi*Plife/Prot
            if Q > 0.5:
                return -1e101
            rv_dictionary['GPVector'][0] = np.log(S0)
            # Save the log(1/timescale) term of the current GP vector:
            rv_dictionary['GPVector'][1] = np.log(Q)
            # Save the log(1/timescale) term of the current GP vector:
            rv_dictionary['GPVector'][2] = np.log(omega0)
        rv_dictionary['GPObject'].set_parameter_vector(rv_dictionary['GPVector'])
        rv_dictionary['GPObject'].compute(rv_dictionary['X'],yerr=rvresiduals_err)
        log_likelihood += rv_dictionary['GPObject'].log_likelihood(rvresiduals)

    # And now return joint log-likelihood:
    return log_likelihood

out_file = out_folder+'out_multinest_'
print('\t Fitting ',n_transit,' transiting planets and ',n_rv,' radial-velocity systems.')
print('\t ',n_params,' free parameters.')
if fitrvline:
    print('\t RV Line Fit: True')
if fitrvquad:
    print('\t RV Quadratic Fit: True')

if n_transit == 0 and (lcfilename is not None):
    print('\n\t >> WARNING: NOT FITTING ANY TRANSITING PLANET. PHOTOMETRY WILL BE ASSUMED AS FLAT. <<')
    print('\t (If this was NOT desired, you forgot to define a prior for (p,b) or (r1,r2) for the transiting planets.)\n')

if n_rv == 0 and (rvfilename is not None):
    print('\n\t >> WARNING: NOT FITTING ANY RV PLANET. RV DATA WILL BE ASSUMED AS FLAT. <<')
    print('\t (If this was NOT desired, you forgot to define a prior for K for the RV planets.)\n')

import pickle
# If not ran already, run dynesty or MultiNest, and save posterior samples and evidences to pickle file:
if (not os.path.exists(out_folder+'posteriors.pkl')) and (not os.path.exists(out_folder+'_dynesty_NS_posteriors.pkl')) and \
   (not os.path.exists(out_folder+'_dynesty_DNS_posteriors.pkl')):
    out = {}
    # Run dynesty or MultiNest:
    if not use_dynesty:
        pymultinest.run(loglike, prior, n_params, \
                        n_live_points = n_live_points,\
                        max_modes=100,\
                        outputfiles_basename=out_file, resume = False, verbose = True)
        # Run and get output:
        output = pymultinest.Analyzer(outputfiles_basename=out_file, n_params = n_params)
        # Get out parameters: this matrix has (samples,n_params+1):
        posterior_samples = output.get_equal_weighted_posterior()[:,:-1]
        # Get INS lnZ:
        out['lnZ'] = output.get_stats()['global evidence']
        out['lnZerr'] = output.get_stats()['global evidence error']
    else:
        if dynamic:
            if dynesty_nthreads == 'none':
                sampler = dynesty.DynamicNestedSampler(loglike, prior, n_params, nlive=n_live_points, bound = dynesty_bound, sample = dynesty_sample)
                # Run and get output:
                sampler.run_nested()
                results = sampler.results
            else:
                from multiprocessing import Pool
                import contextlib
                nthreads = int(dynesty_nthreads)
                with contextlib.closing(Pool(processes=nthreads-1)) as executor:
                    sampler = dynesty.DynamicNestedSampler(loglike, prior, n_params, nlive=n_live_points, \
                                                           bound = dynesty_bound, sample = dynesty_sample, \
                                                           pool=executor, queue_size=nthreads)
                    sampler.run_nested()
                    results = sampler.results
                    
        else:
            if dynesty_nthreads == 'none':
                sampler = dynesty.NestedSampler(loglike, prior, n_params, nlive=n_live_points, bound = dynesty_bound, sample = dynesty_sample)
                # Run and get output:
                sampler.run_nested()
                results = sampler.results
            else:
                from multiprocessing import Pool
                import contextlib
                nthreads = int(dynesty_nthreads)
                with contextlib.closing(Pool(processes=nthreads-1)) as executor:
                    sampler = dynesty.NestedSampler(loglike, prior, n_params, nlive=n_live_points,\
                                                    bound = dynesty_bound, sample = dynesty_sample,\
                                                    pool=executor, queue_size=nthreads)
                    sampler.run_nested()
                    results = sampler.results
        out['dynesty_output'] = results
        # Get weighted posterior:
        weights = np.exp(results['logwt'] - results['logz'][-1])
        posterior_samples = resample_equal(results.samples, weights)
        # Get lnZ:
        out['lnZ'] = results.logz[-1]
        out['lnZerr'] = results.logzerr[-1]
    # Prepare output file:
    out['posterior_samples'] = {}
    out['posterior_samples']['unnamed'] = posterior_samples
    # Extract parameters:
    pcounter = 0 
    for pname in priors.keys():
        if priors[pname]['type'] != 'fixed':
            priors[pname]['cvalue'] = np.median(posterior_samples[:,pcounter])
            out['posterior_samples'][pname] = posterior_samples[:,pcounter]
            pcounter += 1
    out['posterior_samples']['loglike'] = np.zeros(posterior_samples.shape[0])
    for i in range(posterior_samples.shape[0]):
        out['posterior_samples']['loglike'][i] = loglike(posterior_samples[i,:])
    if True in efficient_bp:
        out['pu'] = pu
        out['pl'] = pl
    if use_dynesty:
        if dynamic:
            pickle.dump(out,open(out_folder+'_dynesty_DNS_posteriors.pkl','wb'))
        else:
            pickle.dump(out,open(out_folder+'_dynesty_NS_posteriors.pkl','wb'))
    else:
        pickle.dump(out,open(out_folder+'posteriors.pkl','wb'))
else:
    print('Detected output files --- extracting...')
    priors,n_transit,n_rv,numbering_transit,numbering_rv,n_params = utils.readpriors(out_folder+'priors.dat')
    if use_dynesty:
        if dynamic:
            try:
                out = pickle.load(open(out_folder+'_dynesty_DNS_posteriors.pkl','rb'))
            except:
                out = pickle.load(open(out_folder+'_dynesty_DNS_posteriors.pkl','rb'), encoding='latin1')
        else:
            try:
                out = pickle.load(open(out_folder+'_dynesty_NS_posteriors.pkl','rb'))
            except:
                out = pickle.load(open(out_folder+'_dynesty_NS_posteriors.pkl','rb'), encoding='latin1')
    else:
        try:
            out = pickle.load(open(out_folder+'posteriors.pkl','rb'))
        except:
            out = pickle.load(open(out_folder+'posteriors.pkl','rb'), encoding='latin1')
    # Extract parameters:
    for pname in priors.keys():
        if priors[pname]['type'] != 'fixed':
            priors[pname]['cvalue'] = np.median(out['posterior_samples'][pname])
    posterior_samples = out['posterior_samples']['unnamed'] 
    if 'pu' in out.keys():
        pu = out['pu']
        pl = out['pl']
        Ar = (pu - pl)/(2. + pl + pu)

print('Done!')
# Write the posterior parameters to a file:
if not os.path.exists(out_folder+'posteriors.dat'):
    outpp = open(out_folder+'posteriors.dat','w')
    utils.writepp(outpp,out)

# Define number of samples we'll get to plot the models + uncertainties (default is maximum between all and 
# 5000).
nsims = np.min([n_sims,out['posterior_samples']['unnamed'].shape[0]])
print('\t Drawing',nsims,'out of',out['posterior_samples']['unnamed'].shape[0],'samples from the posterior...')
if nsims == out['posterior_samples']['unnamed'].shape[0]:
    idx_sims = np.arange(out['posterior_samples']['unnamed'].shape[0])
else:
    idx_sims = np.random.choice(np.arange(out['posterior_samples']['unnamed'].shape[0]),nsims,replace=False)

# Ok, here comes the plotting functions (*takes deep breath*).
# Colors for RV instruments:
rv_colors = ['#F6511D','#0571b0','#ca0020','#090446','#780116','#574AE2']
rv_markers = ['o','o','o','o','o','o']

# First, RV plots:
if rvfilename is not None:
    # First, define zero-point for RV plotting time:
    zero_t_rv = int(np.min(t_rv))
    ###############################################################
    ###############################################################
    ################## FIRST PLOT: RV V/S TIME ####################
    ###############################################################
    ###############################################################
    # Here we not only generate the RV vs time plot, but also generate 
    # the GP model in case a GP was used; we later use this to substract it 
    # from the data to show the phased RVs

    # First, we check if the entire RV dataset can be divided into one big chunk of nrvchunk days 
    # (set by default at 100 days). If this is not the case, we activate the RV multi-panel plot:
    if np.max(t_rv)-np.min(t_rv) > nrvchunk:
        rvmultipanel = True
    else:
        rvmultipanel = False

    # The first plot is RV v/s time. First, we define some preambles for this plot:
    ### ERASE THIS:
    #idxERASE = np.where(((t_rv-2450839)>(2900+2900))&((t_rv-2450839)<(2900+2900+2900)))[0]
    if rvmultipanel is False:
        fig, axs = plt.subplots(2, 1, gridspec_kw = {'height_ratios':[3,1]}, figsize=(10,4))
    else:
        # If in multi-panel plot, first agglomerate data in nrvchunks, and save all the corresponding
        # data for each panel to the chunks dictionary:
        idxrv = np.argsort(t_rv)#t_rv[idxERASE])
        npanels = 0
        chunks  = {} 
        chunks[npanels] = {}
        chunks[npanels]['tmin'] = t_rv[idxrv[0]]-5#[idxERASE][idxrv[0]] - 5
        for i in idxrv:
            if t_rv[i]-chunks[npanels]['tmin'] > nrvchunk:
                chunks[npanels]['tmax'] = chunks[npanels]['tmin'] + nrvchunk + 5
                npanels += 1
                chunks[npanels] = {}
                chunks[npanels]['tmin'] = t_rv[i]-5#t_rv[idxERASE][i] - 5
        if 'tmax' not in chunks[npanels].keys():
            chunks[npanels]['tmax'] = chunks[npanels]['tmin'] + nrvchunk + 5
        # Now that all data is saved, create multi-panel plot:
        if npanels>6:
            rr,cc = 20,20
        if npanels>1:
            rr,cc = 10,10
        else:
            rr,cc = 10,6
        fig, axs = plt.subplots(2*(npanels+1), 1, gridspec_kw = {'height_ratios':[3,1]*(npanels+1)}, figsize=(rr,cc))
            
        
    sns.set_context("talk")
    sns.set_style("ticks")
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'
    matplotlib.rcParams['font.size'] = '5'
    matplotlib.rcParams['axes.linewidth'] = 1.2
    matplotlib.rcParams['xtick.direction'] = 'out'
    matplotlib.rcParams['ytick.direction'] = 'out'
    matplotlib.rcParams['lines.markeredgewidth'] = 1
    # Substract the best-fit line, if any, to the RV data:
    if fitrvline:
        # Line is of the form RV = aline*t + bline:
        aline = priors['rv_slope']['cvalue']
        bline = -priors['rv_tzero']['cvalue']*aline
        # Add line to the RV model:
        rvline = aline*t_rv + bline
        rv_rv = rv_rv - rvline
    if fitrvquad:
        # Quad is of the form RV = q*t**2 + aline*t + bline:
        qquad = priors['rv_quad']['cvalue']
        aline = priors['rv_slope']['cvalue']
        bline = -priors['rv_tzero']['cvalue']*aline - (priors['rv_tzero']['cvalue']**2)*qquad
        # Add line to the RV model:
        rvquad = qquad*(t_rv**2) + aline*t_rv + bline
        rv_rv = rv_rv - rvquad
    # Now, iterate between the two plots; one for the RV v/s time, and one for the residuals v/s time:
    print('Plotting RV vs time...')
    for i in range(2):
        if rvmultipanel is False:
            ax = axs[i]
        if i == 0:
          # Create dictionary that will save the systemic-corrected RVs (useful for the other plots!):
          sys_corrected = {}
          # First row is plot of time v/s rv plot. To plot this, first plot the data:
          color_counter =  0
          all_rv = np.zeros(len(t_rv))
          all_rv_err = np.zeros(len(t_rv))
          for instrument in inames_rv: 
              all_rv[[instrument_indexes_rv[instrument]]] = rv_rv[instrument_indexes_rv[instrument]] - priors['mu_'+instrument]['cvalue']
              all_rv_err[[instrument_indexes_rv[instrument]]] = np.sqrt(rverr_rv[instrument_indexes_rv[instrument]]**2 + priors['sigma_w_rv_'+instrument]['cvalue']**2)
              corrected_rv = rv_rv[instrument_indexes_rv[instrument]] - priors['mu_'+instrument]['cvalue']
              corrected_rv_err = np.sqrt(rverr_rv[instrument_indexes_rv[instrument]]**2 + priors['sigma_w_rv_'+instrument]['cvalue']**2)
              if rvmultipanel is False:
                  ax.errorbar(t_rv[instrument_indexes_rv[instrument]] - zero_t_rv,corrected_rv,\
                              yerr=corrected_rv_err,fmt=rv_markers[color_counter],ms=7,\
                              label=instrument.upper(),elinewidth=1,color=rv_colors[color_counter],alpha=0.75)
              else:
                  for npanel in chunks.keys():
                      ax = axs[2*npanel] 
                      ax.errorbar(t_rv[instrument_indexes_rv[instrument]] - zero_t_rv,corrected_rv,\
                              yerr=corrected_rv_err,fmt=rv_markers[color_counter],ms=7,\
                              label=instrument.upper(),elinewidth=1,color=rv_colors[color_counter],alpha=0.75) 
              color_counter += 1
              # Save systemic corrected RVs:
              sys_corrected[instrument] = {}
              sys_corrected[instrument]['values'] = corrected_rv
              sys_corrected[instrument]['errors'] = corrected_rv_err
          # Now RV model on top. For this, oversample the times:
          if rvmultipanel is False:
              t_rv_model = np.linspace(np.min(t_rv)-10,np.max(t_rv)+10,5*noversample)
          else:
              t_rv_model = np.array([])
              for npanels in chunks.keys():
                  t_rv_model = np.append(t_rv_model,np.linspace(chunks[npanels]['tmin'],chunks[npanels]['tmax'],noversample))
          
          # We might want to change this when implementing multi-dimensional GPs:        
          t_rv_gp_model = t_rv_model
          # Now compute many models, for which we'll get the quantiles later for the 
          # joint RV model. Do the same for the samples according to the times in t_rv_model 
          # (oversampled model) and for the times in t_rv_model (model with same samples as data 
          # --- useful to compute residuals):
          all_rv_models = np.zeros([nsims,len(t_rv_model)])
          all_rv_models_real = np.zeros([nsims,len(t_rv)])
          
          # If GP is on, generate a vector that will save the GP component, to be substracted later for the 
          # phased RVs:
          #if rveparamfile is not None:
          all_gp_models_real = np.zeros([nsims,len(t_rv)])
          all_gp_models = np.zeros([nsims,len(t_rv_model)])
              
          # Here comes one of the slow parts: for the first nsims posterior samples, compute a model:
          print('Sampling models...')
          counter = -1
          for j in idx_sims:
              counter = counter + 1
              # Sample the jth sample of parameter values:
              for pname in priors.keys():
                  if priors[pname]['type'] != 'fixed':
                      priors[pname]['cvalue'] = out['posterior_samples'][pname][j]

              # With those samples, compute full RV model and the planet-by-planet model:
              for n in range(n_rv):
                  iplanet = numbering_rv[n]
                  K,t0,P = priors['K_p'+str(iplanet)]['cvalue'],\
                           priors['t0_p'+str(iplanet)]['cvalue'],\
                           priors['P_p'+str(iplanet)]['cvalue']

                  if ecc_parametrization['rv'][iplanet] == 0:
                      ecc,omega = priors['ecc_p'+str(iplanet)]['cvalue'],priors['omega_p'+str(iplanet)]['cvalue']*np.pi/180.
                  elif ecc_parametrization['rv'][iplanet] == 1:
                      ecc = np.sqrt(priors['ecosomega_p'+str(iplanet)]['cvalue']**2+priors['esinomega_p'+str(iplanet)]['cvalue']**2)
                      omega = np.arctan2(priors['esinomega_p'+str(iplanet)]['cvalue'],priors['ecosomega_p'+str(iplanet)]['cvalue'])
                  else:
                      ecc = priors['secosomega_p'+str(iplanet)]['cvalue']**2+priors['sesinomega_p'+str(iplanet)]['cvalue']**2
                      omega = np.arctan2(priors['sesinomega_p'+str(iplanet)]['cvalue'],priors['secosomega_p'+str(iplanet)]['cvalue'])

                  radvel_params['per'+str(n+1)] = radvel.Parameter(value=P)
                  radvel_params['tc'+str(n+1)] = radvel.Parameter(value=t0)
                  radvel_params['w'+str(n+1)] = radvel.Parameter(value=omega)
                  radvel_params['e'+str(n+1)] = radvel.Parameter(value=ecc)
                  radvel_params['k'+str(n+1)] = radvel.Parameter(value=K)

              # Compute full RV model:
              if rveparamfile is None:
                  all_rv_models[counter,:] = radvel.model.RVModel(radvel_params).__call__(t_rv_model)
                  all_rv_models_real[counter,:] = radvel.model.RVModel(radvel_params).__call__(t_rv)
              else:
                  #if rv_dictionary['GPType'] == 'ExpSineSquaredSEKernel':
                  #    # Save the log(variance) of the jitter term on the current GP vector:
                  #    #rv_dictionary[instrument]['GPVector'][0] = np.log((priors['sigma_w_rv_'+instrument]['cvalue'])**2.)
                  #    # Save pooled log(variance) of the GP process:
                  #    GPVector[0] = np.log((priors['GP_sigma_'+rv_dictionary['GP_sigma']]['cvalue'])**2.)
                  #    # Save log-alpha:
                  #    GPVector[1] = np.log(1./priors['GP_alpha_'+rv_dictionary['GP_alpha']]['cvalue'])
                  #    # Save the Gamma:
                  #    GPVector[2] = priors['GP_Gamma_'+rv_dictionary['GP_Gamma']]['cvalue']
                  #    # And save log(Prot):
                  #    GPVector[3] = np.log(priors['GP_Prot_'+rv_dictionary['GP_Prot']]['cvalue'])
                  #if rv_dictionary['GPType'] == 'CeleriteQPKernel':
                  #    # Note order of GP Vector: logB, logL, logProt, logC, logJitter                  
                  #    # Save the log(B) term of the current GP vector:
                  #    GPVector[0] = np.log(priors['GP_B_'+rv_dictionary['GP_B']]['cvalue'])
                  #    # Save the log(L) term of the current GP vector:
                  #    GPVector[1] = np.log(priors['GP_L_'+rv_dictionary['GP_L']]['cvalue'])
                  #    # Save the log(L) term of the current GP vector:
                  #    GPVector[2] = np.log(priors['GP_Prot_'+rv_dictionary['GP_Prot']]['cvalue'])
                  #    # Save the log(L) term of the current GP vector:
                  #    GPVector[3] = np.log(priors['GP_C_'+rv_dictionary['GP_C']]['cvalue'])
                  rvmodel = radvel.model.RVModel(radvel_params).__call__(t_rv)
                  for instrument in inames_rv:
                      rvresiduals[instrument_indexes_rv[instrument]] = rv_rv[instrument_indexes_rv[instrument]] - \
                                                                       (rvmodel[instrument_indexes_rv[instrument]] + priors['mu_'+instrument]['cvalue'])
                      rvresiduals_err[instrument_indexes_rv[instrument]] = np.sqrt(rverr_rv[instrument_indexes_rv[instrument]]**2 + \
                                                                           priors['sigma_w_rv_'+instrument]['cvalue']**2)

                  if rv_dictionary['GPType'] == 'ExpSineSquaredSEKernel':
                      # Save pooled log(variance) of the GP process:
                      rv_dictionary['GPVector'][0] = np.log((priors['GP_sigma_'+rv_dictionary['GP_sigma']]['cvalue'])**2.)
                      # Save log-alpha:
                      rv_dictionary['GPVector'][1] = np.log(1./priors['GP_alpha_'+rv_dictionary['GP_alpha']]['cvalue'])
                      # Save the Gamma:
                      rv_dictionary['GPVector'][2] = priors['GP_Gamma_'+rv_dictionary['GP_Gamma']]['cvalue']
                      # And save log(Prot):
                      rv_dictionary['GPVector'][3] = np.log(priors['GP_Prot_'+rv_dictionary['GP_Prot']]['cvalue'])
                  if rv_dictionary['GPType'] == 'CeleriteExpKernel':
                      # Note order of GP Vector: logB, logL, logProt, logC, logJitter                  
                      # Save the log(sigma) term of the current GP vector:
                      rv_dictionary['GPVector'][0] = np.log(priors['GP_sigma_'+rv_dictionary['GP_sigma']]['cvalue'])
                      # Save the log(timescale) term of the current GP vector:
                      rv_dictionary['GPVector'][1] = np.log(priors['GP_timescale_'+rv_dictionary['GP_timescale']]['cvalue'])
                  if rv_dictionary['GPType'] == 'CeleriteMatern':
                      # Note order of GP Vector: logB, logL, logProt, logC, logJitter                  
                      # Save the log(sigma) term of the current GP vector:
                      rv_dictionary['GPVector'][0] = np.log(priors['GP_sigma_'+rv_dictionary['GP_sigma']]['cvalue'])
                      # Save the log(rho) term of the current GP vector:
                      rv_dictionary['GPVector'][1] = np.log(priors['GP_rho_'+rv_dictionary['GP_rho']]['cvalue'])
                  if rv_dictionary['GPType'] == 'CeleriteMaternExpKernel':
                      # NOTE: We leave index 2 without value ON PURPOSE: the idea is that here, that is always 0 (because this defines the log(sigma) of the 
                      # matern kernel in the multiplication, which we set to 1).
                      # Save the log(sigma_GP) term of the current GP vector:
                      rv_dictionary['GPVector'][0] = np.log(priors['GP_sigma_'+rv_dictionary['GP_sigma']]['cvalue'])
                      # Save the log(timescale) term of the current GP vector:
                      rv_dictionary['GPVector'][1] = np.log(priors['GP_timescale_'+rv_dictionary['GP_timescale']]['cvalue'])
                      # Save the log(rho) term of the current GP vector:
                      rv_dictionary['GPVector'][3] = np.log(priors['GP_rho_'+rv_dictionary['GP_rho']]['cvalue'])
                  if rv_dictionary['GPType'] == 'CeleriteQPKernel':
                      # Note order of GP Vector: logB, logL, logProt, logC, logJitter                  
                      # Save the log(B) term of the current GP vector:
                      rv_dictionary['GPVector'][0] = np.log(priors['GP_B_'+rv_dictionary['GP_B']]['cvalue'])
                      # Save the log(L) term of the current GP vector:
                      rv_dictionary['GPVector'][1] = np.log(priors['GP_L_'+rv_dictionary['GP_L']]['cvalue'])
                      # Save the log(Prot) term of the current GP vector:
                      rv_dictionary['GPVector'][2] = np.log(priors['GP_Prot_'+rv_dictionary['GP_Prot']]['cvalue'])
                      # Save the log(C) term of the current GP vector:
                      rv_dictionary['GPVector'][3] = np.log(priors['GP_C_'+rv_dictionary['GP_C']]['cvalue'])
                  if rv_dictionary['GPType'] == 'CeleriteSHOKernel':
                      # Transform C0, Prot and Plife to S0, Q and omega0. Note we assume C0 comes in same units as RVs:
                      C0,Prot,Plife = priors['GP_C0_'+rv_dictionary['GP_C0']]['cvalue'],priors['GP_Prot_'+rv_dictionary['GP_Prot']]['cvalue'],\
                                      priors['GP_Plife_'+rv_dictionary['GP_Plife']]['cvalue']
                      S0 = (C0*(Prot**2))/(2. * (np.pi**2) * Plife)
                      omega0 = 2. * np.pi/Prot
                      Q = np.pi*Plife/Prot
                      rv_dictionary['GPVector'][0] = np.log(S0)
                      # Save the log(1/timescale) term of the current GP vector:
                      rv_dictionary['GPVector'][1] = np.log(Q)
                      # Save the log(1/timescale) term of the current GP vector:
                      rv_dictionary['GPVector'][2] = np.log(omega0)

                  rv_dictionary['GPObject'].set_parameter_vector(rv_dictionary['GPVector'])
                  rv_dictionary['GPObject'].compute(rv_dictionary['X'],yerr=rvresiduals_err)

                  # Generate Keplerians:
                  #if counter%100==0: print(counter)
                  model = radvel.model.RVModel(radvel_params).__call__(t_rv_model)
                  model_real = radvel.model.RVModel(radvel_params).__call__(t_rv)
                  # Predict real values: 
                  gpmodel_real,gpvar = rv_dictionary['GPObject'].predict(rvresiduals, rv_dictionary['X'], return_var=True)
                  # If the GP model returns NaN values -- skip it
                  if np.isnan(np.mean(gpmodel_real)):
                    print('Detected a sample that computes a NaN GP model')
                    continue
                  all_gp_models_real[counter,:] = gpmodel_real
                  all_rv_models_real[counter,:] = model_real + gpmodel_real
                  gpmodel,gpvar = rv_dictionary['GPObject'].predict(rvresiduals, t_rv_gp_model, return_var=True)
                  all_gp_models[counter,:] = gpmodel
                  all_rv_models[counter,:] = gpmodel + model
          # Now, finally, compute the median and the quantiles (1,2 and 3-sigma) for each time sample of the 
          # oversampled model (this is the "o" before the names). 
          omedian_model = np.zeros(len(t_rv_model))
          omedian_model_GP = np.zeros(len(t_rv_model))
          omodel_up1, omodel_down1 = np.zeros(len(t_rv_model)),np.zeros(len(t_rv_model))
          omodel_up2, omodel_down2 = np.zeros(len(t_rv_model)),np.zeros(len(t_rv_model))
          omodel_up3, omodel_down3 = np.zeros(len(t_rv_model)),np.zeros(len(t_rv_model))

          for i_tsample in range(len(t_rv_model)):
              # Compute quantiles for the full model:
              val,valup1,valdown1 = utils.get_quantiles(all_rv_models[:,i_tsample])
              val,valup2,valdown2 = utils.get_quantiles(all_rv_models[:,i_tsample],alpha=0.95)
              val,valup3,valdown3 = utils.get_quantiles(all_rv_models[:,i_tsample],alpha=0.99)
              omedian_model[i_tsample] = val
              valGP,valup1GP,valdown1GP = utils.get_quantiles(all_gp_models[:,i_tsample])
              omedian_model_GP[i_tsample] = valGP
              omodel_up1[i_tsample],omodel_down1[i_tsample] = valup1,valdown1
              omodel_up2[i_tsample],omodel_down2[i_tsample] = valup2,valdown2
              omodel_up3[i_tsample],omodel_down3[i_tsample] = valup3,valdown3
          
          # Save oversampled full RV models and GP models:
          fout = open(out_folder+'time_rv_model.dat','w')
          fout.write('# Time \t GP Component \t Full model (GP + Keplerian) \t model1down \t model1up \t model2down \t model2up \t model3down \t model3up\n')
          for ii in range(len(omedian_model_GP)):
              fout.write('{0:.10f} \t {1:.10f} \t {2:.10f} \t {3:.10f} \t {4:.10f} \t {5:.10f} \t {6:.10f} \t {7:.10f} \t {8:.10f}\n'.format(\
                t_rv_model[ii],omedian_model_GP[ii],omedian_model[ii],\
                omodel_down1[ii], omodel_up1[ii], omodel_down2[ii], omodel_up2[ii], omodel_down3[ii], omodel_up3[ii]))
          fout.close()

          if rvmultipanel is False:
              # Plot model and uncertainty in model given by posterior sampling:
              ax.fill_between(t_rv_model - zero_t_rv,omodel_down1,omodel_up1,color='cornflowerblue',alpha=0.25)
              ax.fill_between(t_rv_model - zero_t_rv,omodel_down2,omodel_up2,color='cornflowerblue',alpha=0.25)
              ax.fill_between(t_rv_model - zero_t_rv,omodel_down3,omodel_up3,color='cornflowerblue',alpha=0.25)
              ax.plot(t_rv_model - zero_t_rv,omedian_model,'-',linewidth=2,color='black') 
              ax.set_xlim([np.min(t_rv)-1-zero_t_rv,np.max(t_rv)+1-zero_t_rv])
              if rvunits == 'ms':
                  ax.set_ylabel('Radial velocity (m/s)')
              else:
                  ax.set_ylabel('Radial velocity (km/s)')
              ax.get_xaxis().set_major_formatter(plt.NullFormatter()) 
              ax.legend(ncol=3)
          else:
              idxsort = np.argsort(t_rv_model - zero_t_rv)
              for npanel in chunks.keys():
                  ax = axs[2*npanel] 
                  ax.fill_between(t_rv_model[idxsort] - zero_t_rv,omodel_down1[idxsort],omodel_up1[idxsort],color='cornflowerblue',alpha=0.25)
                  ax.fill_between(t_rv_model[idxsort] - zero_t_rv,omodel_down2[idxsort],omodel_up2[idxsort],color='cornflowerblue',alpha=0.25)
                  ax.fill_between(t_rv_model[idxsort] - zero_t_rv,omodel_down3[idxsort],omodel_up3[idxsort],color='cornflowerblue',alpha=0.25)
                  ax.plot(t_rv_model[idxsort] - zero_t_rv,omedian_model[idxsort],'-',linewidth=2,color='black')
                  ax.set_xlim([np.min(chunks[npanel]['tmin'])-zero_t_rv,chunks[npanel]['tmax']-zero_t_rv])
                  if rvunits == 'ms':
                      ax.set_ylabel('RV (m/s)')
                  else:
                      ax.set_ylabel('RV (km/s)')
                  #### ERASE:
                  ax.set_ylim([-10,10])
                  ax.get_xaxis().set_major_formatter(plt.NullFormatter())
                  if npanel == 0:
                      lgd = ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,ncol=len(inames_rv), mode="expand", borderaxespad=0.)
                      #ax.legend(ncol=3)
        if i == 1:
          # Second row is residuals. First, compute the median real model to get the residuals:
          all_rv_models_real = np.median(all_rv_models_real,axis=0)
          all_gp_models_real = np.median(all_gp_models_real,axis=0)
          # Plot a zero line to guide the eye:
          # Compute, plot and save the residuals:
          color_counter = 0
          # Prepare file to save systemic-and-quadratic-trend corrected RVs:
          fout = open(out_folder+'rv_residuals.dat','w')
          fout.write('# Time Residual Error Instrument \n')
          fout2 = open(out_folder+'time_rv_data.dat','w')
          fout2.write('# Time \t RV - Systemic RV - Quadratic Trend \t RV Error \t GP Component \t Full Model (GP + Keplerians) \t Instrument\n')
          for instrument in inames_rv:
              if rvmultipanel is False:
                  ax.plot([-1e10,1e10],[0.,0.],'--',linewidth=2,color='black')
                  ax.errorbar(t_rv[instrument_indexes_rv[instrument]] - zero_t_rv, \
                               sys_corrected[instrument]['values'] - all_rv_models_real[instrument_indexes_rv[instrument]], \
                               sys_corrected[instrument]['errors'],fmt=rv_markers[color_counter],ms=7,\
                               label=instrument,elinewidth=1,color=rv_colors[color_counter],alpha=0.75)
              else:
                  for npanel in chunks.keys():
                      ax = axs[2*npanel+1]
                      ax.plot([-1e10,1e10],[0.,0.],'--',linewidth=1,color='black')
                      ax.errorbar(t_rv[instrument_indexes_rv[instrument]] - zero_t_rv, \
                               sys_corrected[instrument]['values'] - all_rv_models_real[instrument_indexes_rv[instrument]], \
                               sys_corrected[instrument]['errors'],fmt=rv_markers[color_counter],ms=7,\
                               label=instrument,elinewidth=1,color=rv_colors[color_counter],alpha=0.75)
              for ii in range(len(t_rv[instrument_indexes_rv[instrument]])):
                  vals = '{0:.10f} {1:.10f} {2:.10f}'.format(t_rv[instrument_indexes_rv[instrument]][ii],sys_corrected[instrument]['values'][ii] - all_rv_models_real[instrument_indexes_rv[instrument]][ii],\
                                                             sys_corrected[instrument]['errors'][ii])
                  vals2 = '{0:.10f} \t {1:.10f} \t {2:.10f} \t {3:.10f} \t {4:.10f}'.format(t_rv[instrument_indexes_rv[instrument]][ii],sys_corrected[instrument]['values'][ii],sys_corrected[instrument]['errors'][ii], \
                                                              all_gp_models_real[instrument_indexes_rv[instrument]][ii],all_rv_models_real[instrument_indexes_rv[instrument]][ii])
                  fout.write(vals+' '+instrument+' \n')
                  fout2.write(vals2+' '+instrument+' \n')
              color_counter += 1
          fout.close()
          fout2.close()
          if rvmultipanel is False:
              ax.set_ylabel('Residuals')
              ax.set_xlabel('Time (BJD - '+str(zero_t_rv)+')')
              ax.set_xlim([np.min(t_rv)-1-zero_t_rv,np.max(t_rv)+1-zero_t_rv])
          else:
              for npanel in chunks.keys():
                  ax = axs[2*npanel+1]
                  ax.set_xlim([np.min(chunks[npanel]['tmin'])-zero_t_rv,chunks[npanel]['tmax']-zero_t_rv])
                  ax.set_ylabel('O-C')
                  if npanel == np.max(chunks.keys()):
                      ax.set_xlabel('Time (BJD - '+str(zero_t_rv)+')')
                  # ERASE:
                  ax.set_ylim([-10,10])
    # Plot RV vs time:
    plt.tight_layout()
    if rvmultipanel:
        plt.savefig(out_folder+'rv_vs_time.pdf',bbox_extra_artists=(lgd,), bbox_inches='tight')
    else:
        #ax.legend(ncol=3)
        plt.savefig(out_folder+'rv_vs_time.pdf')

    ###############################################################
    ###############################################################
    ################## SECOND PLOT: PHASED RVs ####################
    ###############################################################
    ###############################################################

    # Now, plot RV for each planet; each column is a different planet, two rows: one for phased data, other for phased residuals. 
    # As before, preambles: gridspec_kw = {'height_ratios':[3,1]*(npanels+1)}
    if n_rv == 1:
        fig, axs = plt.subplots(2, n_rv, gridspec_kw = {'height_ratios':[3,1]}, figsize=(5,4))
    else:
        fig, axs = plt.subplots(2, n_rv, gridspec_kw = {'height_ratios':[3,1]}, figsize=(13,4))
    sns.set_context("talk")
    sns.set_style("ticks")
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'
    matplotlib.rcParams['font.size'] = '5'
    matplotlib.rcParams['axes.linewidth'] = 1.2
    matplotlib.rcParams['xtick.direction'] = 'out'
    matplotlib.rcParams['ytick.direction'] = 'out'
    matplotlib.rcParams['lines.markeredgewidth'] = 1
 
    print('RV per planet...')
    # Now iterate through the planets:
    for n in range(n_rv):
        iplanet = numbering_rv[n]
        if n_rv == 1:
            ax = axs[0]
            ax_res = axs[1]
        else:
            ax = axs[0][n]
            ax_res = axs[1][n]
        ax_res.plot([-0.5,0.5],[0.,0.],'--',linewidth=1,color='black',zorder=0)
        # First, generate a model that contains the components of all the other planets. For computing this, let us be consistent and not use 
        # simply the medians of the samples but lets use samples from the posterior, which we will then marginalize through median combining 
        # the models we draw from the posterior samples. If GP is on, also substract best-fit GP model:
        rvmodel_minus_iplanet = np.zeros([len(t_rv),len(idx_sims)])
        for nn in range(n_rv):
            counter = 0
            for ii in idx_sims:
                i = numbering_rv[nn]
                if i != iplanet:
                    for pname in priors.keys():
                        if priors[pname]['type'] != 'fixed':
                            priors[pname]['cvalue'] = out['posterior_samples'][pname][ii]
                    K,t0,P = priors['K_p'+str(i)]['cvalue'],\
                             priors['t0_p'+str(i)]['cvalue'],\
                             priors['P_p'+str(i)]['cvalue'] 
                    if ecc_parametrization['rv'][i] == 0:
                        ecc,omega = priors['ecc_p'+str(i)]['cvalue'],priors['omega_p'+str(i)]['cvalue']*np.pi/180.
                    elif ecc_parametrization['rv'][i] == 1:
                        ecc = np.sqrt(priors['ecosomega_p'+str(i)]['cvalue']**2+priors['esinomega_p'+str(i)]['cvalue']**2)
                        omega = np.arctan2(priors['esinomega_p'+str(i)]['cvalue'],priors['ecosomega_p'+str(i)]['cvalue'])
                    else:
                        ecc = priors['secosomega_p'+str(i)]['cvalue']**2+priors['sesinomega_p'+str(i)]['cvalue']**2
                        omega = np.arctan2(priors['sesinomega_p'+str(i)]['cvalue'],priors['secosomega_p'+str(i)]['cvalue'])

                    radvel_params['per'+str(nn+1)] = radvel.Parameter(value=P)
                    radvel_params['tc'+str(nn+1)] = radvel.Parameter(value=t0)
                    radvel_params['w'+str(nn+1)] = radvel.Parameter(value=omega)
                    radvel_params['e'+str(nn+1)] = radvel.Parameter(value=ecc)
                    radvel_params['k'+str(nn+1)] = radvel.Parameter(value=K)
                    rvmodel_minus_iplanet[:,counter] += radvel.model.RVModel(radvel_params).__call__(t_rv,planet_num=nn+1) 
                counter = counter + 1
        # Get median of the posterior modelled samples:
        rvmodel_minus_iplanet = np.median(rvmodel_minus_iplanet,axis=1)
        # Get (median) phases for the current planetary model:
        sim_phases = np.zeros([len(idx_sims),len(t_rv)])  
        counter= 0 
        for ii in idx_sims:
            for pname in priors.keys():
                if priors[pname]['type'] != 'fixed':   
                    priors[pname]['cvalue'] = out['posterior_samples'][pname][ii]
            P,t0 = priors['P_p'+str(iplanet)]['cvalue'],priors['t0_p'+str(iplanet)]['cvalue']
            sim_phases[counter,:] = utils.get_phases(t_rv,P,t0)
            counter = counter + 1
        # Get the actual phases:
        phases = np.median(sim_phases,axis=0)

        # Now plot phased RVs minus the component model without the current planet:
        planet_rvs = np.array([])
        color_counter = 0
        # If GP, remove it as well:
        if rveparamfile is not None:
            rvmodel_minus_iplanet += all_gp_models_real#np.median(all_gp_models_real,axis=0)
        # Now we generate possible binned RV data to be plotted. This also serves for defining y-axis limits in the residuals.
        # This combines all the RV data of all instruments. First, generate all possible bins:
        possible_phases_bin = np.arange(-0.4,0.50,0.1)
        # (The possible_phases_bin = np.array([-0.45,-0.35,-0.25,-0.15,0.0,0.15,0.25,0.35,0.45]))
        phases_bin = np.array([])
        rv_bin = np.array([])
        rv_bin_err = np.array([])
        res_rv_bin = np.array([])
        # Iterate through each possible bin; all points within 0.05 in phase-space from a given phase-bin are gathered in the index ppidx, 
        # then data for the planet is obtained by substracting the model without the current planet under analysis/plotting (data_planet), 
        # along with the inverse variance (w) and the residuals (data_residuals):
        for i in range(len(possible_phases_bin)):
            data_planet = np.array([])
            w = np.array([])
            data_residuals = np.array([])
            for instrument in inames_rv:
                ppidx = np.where(np.abs(phases[instrument_indexes_rv[instrument]]-possible_phases_bin[i])<0.05)[0]
                data_planet = np.append(data_planet,sys_corrected[instrument]['values'][ppidx]-rvmodel_minus_iplanet[instrument_indexes_rv[instrument]][ppidx])
                w = np.append(w,1./(sys_corrected[instrument]['errors'][ppidx]**2))  
                data_residuals = np.append(data_residuals,sys_corrected[instrument]['values'][ppidx] - all_rv_models_real[instrument_indexes_rv[instrument]][ppidx])
            # If more than one datapoint in the current bin, generate binned points:
            if len(w)>1:
                phases_bin = np.append(phases_bin,possible_phases_bin[i])
                rv_bin = np.append(rv_bin,np.sum(w*data_planet)/np.sum(w))
                res_rv_bin = np.append(res_rv_bin,np.sum(w*data_residuals)/np.sum(w))
                rv_bin_err = np.append(rv_bin_err,np.sqrt(1./np.sum(w**2)))
            elif len(w) == 1:
                rv_bin_err = np.append(rv_bin_err,np.sqrt(1./w**2.))
        if plotbinnedrvs:
            ax.errorbar(phases_bin,rv_bin,yerr=rv_bin_err,fmt='o',ms=8,elinewidth=3,markerfacecolor='white',markeredgecolor='black',ecolor='black')
            ax_res.errorbar(phases_bin,res_rv_bin,yerr=rv_bin_err,fmt='o',ms=8,elinewidth=3,markerfacecolor='white',markeredgecolor='black',ecolor='black')

        # Define output file to save planet RVs and model:
        fout = open(out_folder+'phased_rv_planet'+str(iplanet)+'.dat','w')
        fout.write('# Phases \t Time \t Phased RV \t Phased RV Error \t Model\n')
        for instrument in inames_rv:
            #all_rv_data_phases = np.append(all_rv_data_phases,phases[instrument_indexes_rv[instrument]])
            #all_rv_data_data = np.append(all_rv_data_data,\
            #                   sys_corrected[instrument]['values']-rvmodel_minus_iplanet[instrument_indexes_rv[instrument]])
            ax.errorbar(phases[instrument_indexes_rv[instrument]],\
                        sys_corrected[instrument]['values']-rvmodel_minus_iplanet[instrument_indexes_rv[instrument]],\
                        yerr=sys_corrected[instrument]['errors'],fmt=rv_markers[color_counter],ms=6,\
                        elinewidth=1,color=rv_colors[color_counter],alpha=0.3,zorder=1)  
            ax.errorbar(phases[instrument_indexes_rv[instrument]],\
                        sys_corrected[instrument]['values']-rvmodel_minus_iplanet[instrument_indexes_rv[instrument]],\
                        fmt=rv_markers[color_counter],ms=6,\
                        color=rv_colors[color_counter],alpha=0.8,zorder=1)  

            # Save RVs and model:
            for ii in range(len(phases[instrument_indexes_rv[instrument]])):
                fout.write('{0:.10f} {1:.10f} {2:.10f} {3:.10f} {4:.10f} {5}\n'.format(phases[instrument_indexes_rv[instrument]][ii],t_rv[instrument_indexes_rv[instrument]][ii],\
                                                                                   sys_corrected[instrument]['values'][ii]-rvmodel_minus_iplanet[instrument_indexes_rv[instrument]][ii],\
                                                                                   sys_corrected[instrument]['errors'][ii],all_rv_models_real[instrument_indexes_rv[instrument]][ii]-rvmodel_minus_iplanet[instrument_indexes_rv[instrument]][ii],\
                                                                                   instrument))

            # Plot binned data:
            # Plot residuals phased at this planet:
            ax_res.errorbar(phases[instrument_indexes_rv[instrument]], \
                            sys_corrected[instrument]['values'] - all_rv_models_real[instrument_indexes_rv[instrument]],\
                            sys_corrected[instrument]['errors'],fmt=rv_markers[color_counter],ms=6,elinewidth=1,\
                            color=rv_colors[color_counter],alpha=0.3,zorder=1)
            ax_res.errorbar(phases[instrument_indexes_rv[instrument]], \
                            sys_corrected[instrument]['values'] - all_rv_models_real[instrument_indexes_rv[instrument]],\
                            fmt=rv_markers[color_counter],ms=6,elinewidth=1,\
                            color=rv_colors[color_counter],alpha=0.8,zorder=1)
            # This following array is useful for computing limits of the plot:
            planet_rvs = np.append(planet_rvs,sys_corrected[instrument]['values']-rvmodel_minus_iplanet[instrument_indexes_rv[instrument]])
            color_counter += 1
        fout.close() 
        #idx_phases_sorted = np.argsort(all_rv_data_phases)
        #phases_bin,rv_bin,rv_bin_err = utils.bin_data(all_rv_data_phases[idx_phases_sorted],\
        #                                            all_rv_data_data[idx_phases_sorted],5)
        #ax.errorbar(phases_bin,rv_bin,yerr=rv_bin_err,fmt='o',mec='black',mfc='white',elinewidth=1,ecolor='black')
        # Now, as in the previous plot, sample models from the posterior parameters along the phases of interest. 
        # For this, first define a range of phases of interest:
        model_phases = np.linspace(-0.6,0.6,10000) 

        #t_model_phases = np.zeros(len(model_phases),len(idx_sims))
        #counter = -1

        # With this get the respective times for the model phases:
        #t_model_phases = model_phases*P + t0

        # Now generate the models:
        all_rv_models = np.zeros([nsims,len(model_phases)])
        counter = -1
        for j in idx_sims:
            counter = counter + 1
            # Sample the jth sample of parameter values:
            for pname in priors.keys():
                if priors[pname]['type'] != 'fixed':
                    priors[pname]['cvalue'] = out['posterior_samples'][pname][j]

            # With those samples, compute full RV model and the planet-by-planet model:
            K,t0,P = priors['K_p'+str(iplanet)]['cvalue'],\
                priors['t0_p'+str(iplanet)]['cvalue'],\
                priors['P_p'+str(iplanet)]['cvalue']

            if ecc_parametrization['rv'][iplanet] == 0:
                ecc,omega = priors['ecc_p'+str(iplanet)]['cvalue'],priors['omega_p'+str(iplanet)]['cvalue']*np.pi/180.
            elif ecc_parametrization['rv'][iplanet] == 1:
                ecc = np.sqrt(priors['ecosomega_p'+str(iplanet)]['cvalue']**2+priors['esinomega_p'+str(iplanet)]['cvalue']**2)
                omega = np.arctan2(priors['esinomega_p'+str(iplanet)]['cvalue'],priors['ecosomega_p'+str(iplanet)]['cvalue'])
            else:
                ecc = priors['secosomega_p'+str(iplanet)]['cvalue']**2+priors['sesinomega_p'+str(iplanet)]['cvalue']**2
                omega = np.arctan2(priors['sesinomega_p'+str(iplanet)]['cvalue'],priors['secosomega_p'+str(iplanet)]['cvalue'])

            radvel_params['per'+str(n+1)] = radvel.Parameter(value=P)
            radvel_params['tc'+str(n+1)] = radvel.Parameter(value=t0)
            radvel_params['w'+str(n+1)] = radvel.Parameter(value=omega)
            radvel_params['e'+str(n+1)] = radvel.Parameter(value=ecc)
            radvel_params['k'+str(n+1)] = radvel.Parameter(value=K)

            # Compute full RV model:
            t_model_phases = model_phases*P + t0
            all_rv_models[counter,:] = radvel.model.RVModel(radvel_params).__call__(t_model_phases,planet_num=n+1)
        # As before, once again compute median model and the respective error bands:
        omedian_model = np.zeros(len(t_model_phases))
        omodel_up1, omodel_down1 = np.zeros(len(t_model_phases)),np.zeros(len(t_model_phases))
        omodel_up2, omodel_down2 = np.zeros(len(t_model_phases)),np.zeros(len(t_model_phases))
        omodel_up3, omodel_down3 = np.zeros(len(t_model_phases)),np.zeros(len(t_model_phases))

        for i_tsample in range(len(t_model_phases)):
            # Compute quantiles for the full model:
            val,valup1,valdown1 = utils.get_quantiles(all_rv_models[:,i_tsample])
            val,valup2,valdown2 = utils.get_quantiles(all_rv_models[:,i_tsample],alpha=0.95)
            val,valup3,valdown3 = utils.get_quantiles(all_rv_models[:,i_tsample],alpha=0.99)
            omedian_model[i_tsample] = val
            omodel_up1[i_tsample],omodel_down1[i_tsample] = valup1,valdown1
            omodel_up2[i_tsample],omodel_down2[i_tsample] = valup2,valdown2
            omodel_up3[i_tsample],omodel_down3[i_tsample] = valup3,valdown3

        # Now plot the phased model. Compute sorting indexes as well and plot sorted phases:
        ax.fill_between(model_phases,omodel_down1,omodel_up1,color='cornflowerblue',alpha=0.25)
        ax.fill_between(model_phases,omodel_down2,omodel_up2,color='cornflowerblue',alpha=0.25)
        ax.fill_between(model_phases,omodel_down3,omodel_up3,color='cornflowerblue',alpha=0.25)
        ax.plot(model_phases,omedian_model,'-',linewidth=2,color='black')
        ax.set_xlim([-0.5,0.5])       
        
         # Save oversampled full RV models and GP models phased version:
        fout = open(out_folder+'phased_rv_planet'+str(iplanet)+'_model.dat','w')
        fout.write('# Phases \t model \t model1down \t model1up \t model2down \t model2up \t model3down \t model3up\n')
        for ii in range(len(omedian_model)):
            fout.write('{0:.10f} \t {1:.10f} \t {2:.10f} \t {3:.10f} \t {4:.10f} \t {5:.10f} \t {6:.10f} \t {7:.10f}\n'.format(\
                model_phases[ii],omedian_model[ii],\
                omodel_down1[ii], omodel_up1[ii], omodel_down2[ii], omodel_up2[ii], omodel_down3[ii], omodel_up3[ii]))
        fout.close()
        
        if priors['P_p'+str(iplanet)]['type'] != 'fixed':
            P = np.median(out['posterior_samples']['P_p'+str(iplanet)])
        else:
            P = priors['P_p'+str(iplanet)]['cvalue']
        if priors['t0_p'+str(iplanet)]['type'] != 'fixed':
            t0 = np.median(out['posterior_samples']['t0_p'+str(iplanet)])
        else:
            t0 = priors['t0_p'+str(iplanet)]['cvalue']
        #P,t0 = np.median(out['posterior_samples']['P_p'+str(iplanet)]),\
        #:       np.median(out['posterior_samples']['t0_p'+str(iplanet)])
        #print(P,t0,iplanet)
        title_text = r'$P={0:.3f}$, $t_0 = {1:.5f}$'.format(P,t0)
        if n_rv>1:
            ax.set_title(title_text)
        if n == 0:
            if rvunits == 'ms':
                ax.set_ylabel('Radial velocity (m/s)')
                
            else:
                ax.set_ylabel('Radial velocity (km/s)')
            ax_res.set_ylabel('Residuals')
        yval_lim = np.max([np.abs(np.min(omodel_down3)),np.abs(np.max(omodel_up3)),3.*np.sqrt(np.var(planet_rvs))])
        ##########3 CHANGE RV PHASE PLOT YLIM:
        ax.set_ylim([-yval_lim,yval_lim])
        #ax.set_ylim([-5,5])
        ax.get_xaxis().set_major_formatter(plt.NullFormatter())
        # Plot residuals phased at this planet. 
        #UNCOMMENT:
        ax_res.set_ylim([-5*np.median(rv_bin_err),5*np.median(rv_bin_err)])
        #ax_res.set_ylim([-80,80])
        ########################## ERASE THIS
        #for iii in range(all_rv_models.shape[0]):
        #    ax.plot(model_phases,all_rv_models[iii,:],color='black',alpha=0.001,zorder=100)
        #####################################
        ax_res.set_xlim([-0.5,0.5])
        ax_res.set_xlabel('Phase')

        #              ax.errorbar(t_rv[instrument_indexes_rv[instrument]] - zero_t_rv, \
        #                       sys_corrected[instrument]['values'] - all_rv_models_real[instrument_indexes_rv[instrument]], \
        #                       sys_corrected[instrument]['errors'],fmt='.',label=instrument,elinewidth=1,color=rv_colors[color_counter],alpha=0.5)

    # Plot RV v/s phase for each planet:
    plt.tight_layout()
    plt.savefig(out_folder+'rvs_planets.pdf',bbox_inches='tight')


# Finally, transit plots:
if lcfilename is not None:

    ###############################################################
    ###############################################################
    #######     THIRD PLOT: PHOTOMETRY BY INSTRUMENT       ########  
    #######     AS A FUNCTION OF TIME. IF GP, PLOT         ########
    #######     GP + MODEL AND REMOVE GP (I.E. "DETREND")  ########
    ###############################################################
    ###############################################################


    # First, photometry (time vs flux) per instrument on different plots. Top plot will be relative flux 
    # vs time, lower plot residuals:
    norm_factors = {}
    linear_model = {}
    for instrument in inames_lc:
        print(r'\t Generating plot for instrument '+instrument)
        tbaseline = np.max(t_lc[instrument_indexes_lc[instrument]]) - np.min(t_lc[instrument_indexes_lc[instrument]])
        if tbaseline > 0.5:
            fig, axs = plt.subplots(2, 1,gridspec_kw = {'height_ratios':[3,1]}, figsize=(10,5))
        else:
            fig, axs = plt.subplots(2, 1,gridspec_kw = {'height_ratios':[3,1]}, figsize=(9,7))
        sns.set_context("talk")
        sns.set_style("ticks")
        matplotlib.rcParams['mathtext.fontset'] = 'stix'
        matplotlib.rcParams['font.family'] = 'STIXGeneral'
        matplotlib.rcParams['font.size'] = '5'
        matplotlib.rcParams['axes.linewidth'] = 1.2
        matplotlib.rcParams['xtick.direction'] = 'out'
        matplotlib.rcParams['ytick.direction'] = 'out'
        matplotlib.rcParams['lines.markeredgewidth'] = 1

        # Generate lightcurve models for each instrument, for each posterior sample:
        tinstrument = t_lc[instrument_indexes_lc[instrument]]
        all_lc_real_models = np.ones([nsims,len(tinstrument)])
        all_lc_GP_real_models = np.zeros([nsims,len(tinstrument)])
        GPmodel = np.ones(len(tinstrument))
        median_linear_model = np.zeros(len(tinstrument))
        # Generate model lightcurves for each sample in the current instrument:
        counter = -1
        # Here we will save the normalization factors. People from the future: this basically corrects a plotting bug that appears 
        # because the expectation value of y ~ (T*D + (1-D))*(1/(1+DM)) is y = T*E[D/(1+DM)] + E[(1-D)/(1+D*M)] = T*F1 + F2, so later when we want 
        # to normalize the data for plotting (so out-of-transit is one) we need to do normed(y) = (y - F2)/F1. Before, I was just 
        # calculating E[D] and E[M] and calculating y = T*E[D]/(1+E[D]*E[M]) + (1-E[D])/(1+E[D]*E[M]), which is wrong.
        norm_factors[instrument] = {}
        norm_factors[instrument]['F1'] = np.array([])
        norm_factors[instrument]['F2'] = np.array([])
        # Now that we are adding possible linear models, also store the posterior samples of this linear model to substract it later:
        linear_model[instrument] = np.zeros([nsims,len(tinstrument)])

        for j in idx_sims:
            counter = counter + 1
            # Sample the jth sample of parameter values:
            for pname in priors.keys():
                if priors[pname]['type'] != 'fixed':
                    priors[pname]['cvalue'] = out['posterior_samples'][pname][j]
            # For each transit model iterate through the 
            # number of planets, multiplying their transit models:
            if lc_dictionary[instrument]['TransitFit']:
                for n in range(n_transit):
                    i = numbering_transit[n]
                    if lc_dictionary[instrument]['ldlaw'] != 'linear':
                        coeff1,coeff2 = reverse_ld_coeffs(lc_dictionary[instrument]['ldlaw'],priors['q1_'+ld_iname[instrument]]['cvalue'],\
                                        priors['q2_'+ld_iname[instrument]]['cvalue'])
                        lc_dictionary[instrument]['params'].u = [coeff1,coeff2]
                    else:
                        lc_dictionary[instrument]['params'].u = [priors['q1_'+ld_iname[instrument]]['cvalue']]

                    if efficient_bp[n]:
                        if not fitrho:
                            a,r1,r2,t0,P = priors['a_p'+str(i)]['cvalue'],priors['r1_p'+str(i)]['cvalue'],\
                                           priors['r2_p'+str(i)]['cvalue'], priors['t0_p'+str(i)]['cvalue'], \
                                           priors['P_p'+str(i)]['cvalue']
                        else:
                            rho,r1,r2,t0,P = priors['rho']['cvalue'],priors['r1_p'+str(i)]['cvalue'],\
                                             priors['r2_p'+str(i)]['cvalue'], priors['t0_p'+str(i)]['cvalue'], \
                                             priors['P_p'+str(i)]['cvalue']
                            a = ((rho*G*((P*24.*3600.)**2))/(3.*np.pi))**(1./3.)
                        if r1 > Ar:
                            b,p = (1+pl)*(1. + (r1-1.)/(1.-Ar)),\
                                  (1-r2)*pl + r2*pu
                        else:
                            b,p = (1. + pl) + np.sqrt(r1/Ar)*r2*(pu-pl),\
                                  pu + (pl-pu)*np.sqrt(r1/Ar)*(1.-r2)
                    else:
                        if not fitrho:
                            a,b,p,t0,P = priors['a_p'+str(i)]['cvalue'],priors['b_p'+str(i)]['cvalue'],\
                                         priors['p_p'+str(i)]['cvalue'], priors['t0_p'+str(i)]['cvalue'], \
                                         priors['P_p'+str(i)]['cvalue']
                        else:
                            rho,b,p,t0,P = priors['rho']['cvalue'],priors['b_p'+str(i)]['cvalue'],\
                                         priors['p_p'+str(i)]['cvalue'], priors['t0_p'+str(i)]['cvalue'], \
                                         priors['P_p'+str(i)]['cvalue']
                            a = ((rho*G*((P*24.*3600.)**2))/(3.*np.pi))**(1./3.)
                    if ecc_parametrization['transit'][i] == 0:
                        ecc,omega = priors['ecc_p'+str(i)]['cvalue'],priors['omega_p'+str(i)]['cvalue']
                    elif ecc_parametrization['transit'][i] == 1:
                        ecc = np.sqrt(priors['ecosomega_p'+str(i)]['cvalue']**2+priors['esinomega_p'+str(i)]['cvalue']**2)
                        omega = np.arctan2(priors['esinomega_p'+str(i)]['cvalue'],priors['ecosomega_p'+str(i)]['cvalue'])*180./np.pi
                    else:
                        ecc = priors['secosomega_p'+str(i)]['cvalue']**2+priors['sesinomega_p'+str(i)]['cvalue']**2
                        omega = np.arctan2(priors['sesinomega_p'+str(i)]['cvalue'],priors['secosomega_p'+str(i)]['cvalue'])*180./np.pi

                    ecc_factor = (1. + ecc*np.sin(omega * np.pi/180.))/(1. - ecc**2)
                    inc_inv_factor = (b/a)*ecc_factor
                    inc = np.arccos(inc_inv_factor)*180./np.pi
                    lc_dictionary[instrument]['params'].t0 = t0 
                    lc_dictionary[instrument]['params'].per = P
                    lc_dictionary[instrument]['params'].rp = p
                    lc_dictionary[instrument]['params'].a = a
                    lc_dictionary[instrument]['params'].inc = inc
                    lc_dictionary[instrument]['params'].ecc = ecc
                    lc_dictionary[instrument]['params'].w = omega
                    all_lc_real_models[counter,:] = all_lc_real_models[counter,:]*\
                                              lc_dictionary[instrument]['m'].light_curve(lc_dictionary[instrument]['params'])


            all_lc_real_models[counter,:] = (all_lc_real_models[counter,:]*priors['mdilution_'+instrument]['cvalue'] + \
                                      (1. - priors['mdilution_'+instrument]['cvalue']))*\
                                      (1./(1. + priors['mdilution_'+instrument]['cvalue']*priors['mflux_'+instrument]['cvalue']))

            # If inputs for linear model, add it to the instrumental model:
            if lm_boolean[instrument]:
                for idx_X in range(lm_arguments[instrument].shape[1]):
                    linear_model[instrument][counter,:] = linear_model[instrument][counter,:] + lm_arguments[instrument][:,idx_X]*priors['theta'+str(idx_X)+'_'+instrument]['cvalue']
                    all_lc_real_models[counter,:] = all_lc_real_models[counter,:] + lm_arguments[instrument][:,idx_X]*priors['theta'+str(idx_X)+'_'+instrument]['cvalue']
                    
            F1,F2 = priors['mdilution_'+instrument]['cvalue']/(1. + (priors['mdilution_'+instrument]['cvalue'])*(priors['mflux_'+instrument]['cvalue'])),\
                    (1. - priors['mdilution_'+instrument]['cvalue'])/(1. + priors['mdilution_'+instrument]['cvalue']*priors['mflux_'+instrument]['cvalue'])
            norm_factors[instrument]['F1'] = np.append(norm_factors[instrument]['F1'],F1)
            norm_factors[instrument]['F2'] = np.append(norm_factors[instrument]['F2'],F2)
            if lc_dictionary[instrument]['GPDetrend']:
                # Set current values to GP Vector:
                if lc_dictionary[instrument]['GPType'] == 'SEKernel' \
                or lc_dictionary[instrument]['GPType'] == 'M32Kernel':
                    alpha_name = 'alpha' if lc_dictionary[instrument]['GPType'] == 'SEKernel' else 'malpha'
                    # Save the log(variance) of the jitter term on the current GP vector:
                    lc_dictionary[instrument]['GPVector'][0] = np.log((priors['sigma_w_'+instrument]['cvalue']*1e-6)**2.)
                    # Save pooled variance of the GP process:
                    lc_dictionary[instrument]['GPVector'][1] = np.log((priors['GP_sigma_'+lc_dictionary[instrument]['GP_sigma']]['cvalue']*1e-6)**2.)
                    # Now save (log of) coefficients of each GP term:
                    for i in range(lc_dictionary[instrument]['nX']):
                        lc_dictionary[instrument]['GPVector'][2+i] = np.log(
                                0.5/priors[f'GP_{alpha_name}'+str(i)+'_'+lc_dictionary[instrument][f'GP_{alpha_name}'+str(i)]]['cvalue']
                        )
                if lc_dictionary[instrument]['GPType'] == 'ExpSineSquaredSEKernel':
                    # Save the log(variance) of the jitter term on the current GP vector:
                    lc_dictionary[instrument]['GPVector'][0] = np.log((priors['sigma_w_'+instrument]['cvalue']*1e-6)**2.)
                    # Save pooled log(variance) of the GP process:
                    lc_dictionary[instrument]['GPVector'][1] = np.log((priors['GP_sigma_'+lc_dictionary[instrument]['GP_sigma']]['cvalue']*1e-6)**2.)
                    # Save log-alpha (note one passes log(lengthscale**2) to george, which assumes a SqExp part of the form 
                    # sigmaGP**2 * exp(r**2/(2.*M)), with M = lengthscale**2. In our notation, alpha = 1/(2*lengthscale**2):
                    lc_dictionary[instrument]['GPVector'][2] = np.log(0.5/priors['GP_alpha_'+lc_dictionary[instrument]['GP_alpha']]['cvalue'])
                    # Save the Gamma:
                    lc_dictionary[instrument]['GPVector'][3] = priors['GP_Gamma_'+lc_dictionary[instrument]['GP_Gamma']]['cvalue']
                    # And save log(Prot):
                    lc_dictionary[instrument]['GPVector'][4] = np.log(priors['GP_Prot_'+lc_dictionary[instrument]['GP_Prot']]['cvalue'])
                if lc_dictionary[instrument]['GPType'] == 'CeleriteQPKernel':
                    # Note order of GP Vector: logB, logL, logProt, logC, logJitter                  
                    # Save the log(B) term of the current GP vector:
                    lc_dictionary[instrument]['GPVector'][0] = np.log(priors['GP_B_'+lc_dictionary[instrument]['GP_B']]['cvalue'])
                    # Save the log(L) term of the current GP vector:
                    lc_dictionary[instrument]['GPVector'][1] = np.log(priors['GP_L_'+lc_dictionary[instrument]['GP_L']]['cvalue'])
                    # Save the log(L) term of the current GP vector:
                    lc_dictionary[instrument]['GPVector'][2] = np.log(priors['GP_Prot_'+lc_dictionary[instrument]['GP_Prot']]['cvalue'])
                    # Save the log(L) term of the current GP vector:
                    lc_dictionary[instrument]['GPVector'][3] = np.log(priors['GP_C_'+lc_dictionary[instrument]['GP_C']]['cvalue'])
                    # Save the log(L) term of the current GP vector:
                    lc_dictionary[instrument]['GPVector'][4] = np.log(priors['sigma_w_'+instrument]['cvalue']*1e-6)
                if lc_dictionary[instrument]['GPType'] == 'CeleriteExpKernel':
                    # Save the log(sigma_GP) term of the current GP vector:
                    lc_dictionary[instrument]['GPVector'][0] = np.log((priors['GP_sigma_'+lc_dictionary[instrument]['GP_sigma']]['cvalue']**2)*1e-6)
                    # Save the log(1/timescale) term of the current GP vector:
                    lc_dictionary[instrument]['GPVector'][1] = np.log(1./priors['GP_timescale_'+lc_dictionary[instrument]['GP_timescale']]['cvalue'])
                    # Save the log(jitte) term of the current GP vector:
                    lc_dictionary[instrument]['GPVector'][2] = np.log(priors['sigma_w_'+instrument]['cvalue']*1e-6)
                if lc_dictionary[instrument]['GPType'] == 'CeleriteMatern':
                    # Save the log(sigma_GP) term of the current GP vector:
                    lc_dictionary[instrument]['GPVector'][0] = np.log((priors['GP_sigma_'+lc_dictionary[instrument]['GP_sigma']]['cvalue'])*1e-6)
                    # Save the log(1/timescale) term of the current GP vector:
                    lc_dictionary[instrument]['GPVector'][1] = np.log(priors['GP_rho_'+lc_dictionary[instrument]['GP_rho']]['cvalue'])
                    # Save the log(jitte) term of the current GP vector:
                    lc_dictionary[instrument]['GPVector'][2] = np.log(priors['sigma_w_'+instrument]['cvalue']*1e-6)
                if lc_dictionary[instrument]['GPType'] == 'CeleriteMaternExpKernel':
                    # Save the log(sigma_GP) term of the current GP vector:
                    lc_dictionary[instrument]['GPVector'][0] = np.log((priors['GP_sigma_'+lc_dictionary[instrument]['GP_sigma']]['cvalue'])*1e-6)
                    # Save the log(1/timescale) term of the current GP vector:
                    lc_dictionary[instrument]['GPVector'][1] = np.log(1./priors['GP_timescale_'+lc_dictionary[instrument]['GP_timescale']]['cvalue'])
                    # Save the log(1/timescale) term of the current GP vector:
                    lc_dictionary[instrument]['GPVector'][3] = np.log(priors['GP_rho_'+lc_dictionary[instrument]['GP_rho']]['cvalue'])
                    # Save the log(jitte) term of the current GP vector:
                    lc_dictionary[instrument]['GPVector'][4] = np.log(priors['sigma_w_'+instrument]['cvalue']*1e-6)
                if lc_dictionary[instrument]['GPType'] == 'CeleriteSHOKernel':
                    # Transform C0, Prot and Plife to S0, Q and omega0. Note we assume C0 comes in ppm:
                    C0,Prot,Plife = priors['GP_C0_'+lc_dictionary[instrument]['GP_C0']]['cvalue']*1e-6,priors['GP_Prot_'+lc_dictionary[instrument]['GP_Prot']]['cvalue'],\
                                    priors['GP_Plife_'+lc_dictionary[instrument]['GP_Plife']]['cvalue']
                    S0 = (C0*(Prot**2))/(2. * (np.pi**2) * Plife)
                    omega0 = 2. * np.pi/Prot
                    Q = np.pi*Plife/Prot
                    lc_dictionary[instrument]['GPVector'][0] = np.log(S0)
                    # Save the log(1/timescale) term of the current GP vector:
                    lc_dictionary[instrument]['GPVector'][1] = np.log(Q)
                    # Save the log(1/timescale) term of the current GP vector:
                    lc_dictionary[instrument]['GPVector'][2] = np.log(omega0)
                    # Save the log(jitte) term of the current GP vector:
                    lc_dictionary[instrument]['GPVector'][3] = np.log(priors['sigma_w_'+instrument]['cvalue']*1e-6)


                lc_dictionary[instrument]['GPObject'].set_parameter_vector(lc_dictionary[instrument]['GPVector'])
                # Generate residuals with this model:
                residuals = f_lc[instrument_indexes_lc[instrument]] - all_lc_real_models[counter,:]
                # Predict sampled points:
                pred_mean = lc_dictionary[instrument]['GPObject'].predict(residuals, \
                            lc_dictionary[instrument]['X'], return_var=False,return_cov=False)
                # If the GP model returns NaN values -- skip it
                if np.isnan(np.mean(pred_mean)):
                    print('Detected a sample that computes a NaN GP model')
                    continue
                all_lc_GP_real_models[counter,:] = pred_mean
                all_lc_real_models[counter,:] = all_lc_real_models[counter,:] + pred_mean

        # As before, once again compute median model and the respective error bands:
        omedian_model = np.zeros(len(tinstrument))
        omodel_up1, omodel_down1 = np.zeros(len(tinstrument)),np.zeros(len(tinstrument))
        omodel_up2, omodel_down2 = np.zeros(len(tinstrument)),np.zeros(len(tinstrument))
        omodel_up3, omodel_down3 = np.zeros(len(tinstrument)),np.zeros(len(tinstrument))

        for i_tsample in range(len(tinstrument)):
            # Compute quantiles for the full model:
            val,valup1,valdown1 = utils.get_quantiles(all_lc_real_models[:,i_tsample])
            val,valup2,valdown2 = utils.get_quantiles(all_lc_real_models[:,i_tsample],alpha=0.95)
            val,valup3,valdown3 = utils.get_quantiles(all_lc_real_models[:,i_tsample],alpha=0.99)
            omedian_model[i_tsample] = val
            omodel_up1[i_tsample],omodel_down1[i_tsample] = valup1,valdown1
            omodel_up2[i_tsample],omodel_down2[i_tsample] = valup2,valdown2
            omodel_up3[i_tsample],omodel_down3[i_tsample] = valup3,valdown3

        # Do the same for the GP (if no GP, this will be an array of zeros):
        for i_tsample in range(len(tinstrument)):
            GPmodel[i_tsample] = np.median(all_lc_GP_real_models[:,i_tsample])

        # And same for linear models (if no linear model, this will be an array of zeros as well):
        for i_tsample in range(len(tinstrument)):
            median_linear_model[i_tsample] = np.median(linear_model[instrument][:,i_tsample])

        ax = axs[0]
        # Calculate time baseline of observations. Useful to define bounds of the plotted data:
        tbaseline = np.max(tinstrument)-np.min(tinstrument)
        
        # Plot data. Alphas defined whether the time-baseline is 
        # short (<0.5, most likely ground-based data) or large 
        # (most likely space-based data):
        if tbaseline < 0.5:
            alpha_notbinned = 0.5
            alpha_binned = 0.8
        else:
            alpha_notbinned = 0.2
            alpha_binned = 0.5

        tzero = 2457000
        #if tbaseline > 0.5 and (not lc_dictionary[instrument]['resampling']):
        #    ax.plot(tinstrument-tzero,f_lc[instrument_indexes_lc[instrument]],'.k',markersize=5,alpha=alpha_notbinned)
        #    phases_bin,f_bin,f_bin_err = utils.bin_data(tinstrument-tzero,f_lc[instrument_indexes_lc[instrument]],15)
        #    ax.errorbar(phases_bin,f_bin,yerr=f_bin_err,fmt='.k',markersize=5,elinewidth=1,alpha=alpha_binned)
        #else:
        try:
            ferr_instrument = np.sqrt((ferr_lc[instrument_indexes_lc[instrument]])**2 + \
                          (np.median(out['posterior_samples']['sigma_w_'+instrument])*1e-6)**2)
        except KeyError: # If the prior of the sigma_w_instrument was fixed, then there is no posterior dist for it
            ferr_instrument = ferr_lc[instrument_indexes_lc[instrument]]

        ax.errorbar(tinstrument-tzero,f_lc[instrument_indexes_lc[instrument]],\
                    yerr=ferr_instrument,\
                    fmt='.k',markersize=1,alpha=alpha_notbinned,elinewidth=1)

        fout = open(out_folder+'time_lc_'+instrument+'.dat','w')
        fout.write('# Time \t Data \t Error \t Model \t Linear + GP Component\n')

        F1med,F2med = np.median(norm_factors[instrument]['F1']),np.median(norm_factors[instrument]['F2'])
        for i in range(len(tinstrument)):
            fout.write('{0:.10f} {1:.10f} {2:.10f} {3:.10f} {4:.10f}\n'.format(tinstrument[i],f_lc[instrument_indexes_lc[instrument]][i],\
                                                                      ferr_instrument[i],omedian_model[i],F1med + F2med + median_linear_model[i] + GPmodel[i]))
        fout.close()
        # Now plot the phased model. Compute sorting indexes as well and plot sorted phases:
        ax.fill_between(tinstrument-tzero,omodel_down1,omodel_up1,color='cornflowerblue',alpha=0.25)
        ax.fill_between(tinstrument-tzero,omodel_down2,omodel_up2,color='cornflowerblue',alpha=0.25)
        ax.fill_between(tinstrument-tzero,omodel_down3,omodel_up3,color='cornflowerblue',alpha=0.25)
        ax.plot(tinstrument-tzero,omedian_model,'-',linewidth=1,color='black')
        ax.set_ylabel('Relative flux')

        ########### CHANGE FOR CHANGING Y AXIS PHOT PLOT:
        ax.set_ylim([np.min(omedian_model) - np.max(ferr_instrument)*7.5,np.max(omedian_model)+np.max(ferr_instrument)*7.5])
        #ax.set_ylim([0.996,1.003])
        ###########
        # Define the x-axis limits based on time baseline of observations. Basically if it is larger than 
        # half a day, it is most likely space-based and we thus base our plot around the phased transit event. 
        # If not, we base our plot around the expected ingress and egress:
        ax.set_xlim([np.min(tinstrument-tzero),np.max(tinstrument-tzero)])
        ax.get_xaxis().set_major_formatter(plt.NullFormatter())

        # Plot residuals:
        ax2 = axs[1]
        # Plot zero line to guide the eye:
        ax2.plot([-1e10,1e10],[0.,0.],'--',linewidth=2,color='black')
        # Plot residuals:
        #if tbaseline < 0.5:
        ax2.errorbar(tinstrument-tzero,(f_lc[instrument_indexes_lc[instrument]]-omedian_model)*1e6,\
                    yerr=ferr_instrument*1e6,\
                    fmt='.k',markersize=2,elinewidth=1,alpha=alpha_notbinned)
        #else:
        #    #ax.plot(tinstrument-tzero,(f_lc[instrument_indexes_lc[instrument]]-omedian_model)*1e6,'.k',markersize=5,alpha=alpha_notbinned)
        #    #phases_bin,f_bin,f_bin_err = utils.bin_data(tinstrument,(f_lc[instrument_indexes_lc[instrument]]-omedian_model)*1e6,15)
        #    #ax.errorbar(phases_bin,f_bin,yerr=f_bin_err,fmt='.k',markersize=5,elinewidth=1,alpha=alpha_binned)
        ax2.ticklabel_format(useOffset=False, style='plain')
        ax2.set_ylabel('Residuals (ppm)')
        ax2.set_xlabel('Time (BJD - '+str(tzero)+')')
        ################ CHANGE FOR CHANGING Y AXIS RES PHOT PLOT:
        ax2.set_ylim([-np.max(ferr_instrument)*7.5*1e6,np.max(ferr_instrument)*7.5*1e6])
        #ax2.set_ylim([-2500,2500])
        ################
        ax2.set_xlim([np.min(tinstrument-tzero),np.max(tinstrument-tzero)])
        plt.tight_layout()
        plt.savefig(out_folder+'phot_vs_time_instrument_'+instrument+'.pdf')
        # If GPDetrend, remove GP component, set the detrend to false so next plot shows the GP-corrected/detrended photometry:
        if lc_dictionary[instrument]['GPDetrend']:
            f_lc[instrument_indexes_lc[instrument]] = f_lc[instrument_indexes_lc[instrument]] - GPmodel
            lc_dictionary[instrument]['GPDetrend'] = False
        # Also remove linear trend if any:
        if lm_boolean[instrument]:
            f_lc[instrument_indexes_lc[instrument]] = f_lc[instrument_indexes_lc[instrument]] - median_linear_model

    ###############################################################
    ###############################################################
    ########## FOURTH PLOT: PHOTOMETRY BY PLANET, #################
    ####### BY INSTRUMENT; SAVE PHOTOMETRY BY PLANET    ###########
    ###############################################################
    ###############################################################

    # Before we begin, let us correct the observed transit lightcurve 
    # from the dilutions and the mean fluxes of each instrument so we 
    # ensure we have a mean of 1 out of transit in the plots. For this, 
    # re-scale the fluxes --- and also re-scale the errors. Save all this 
    # in a dictionary:
    finstrument = {}
    for instrument in inames_lc:
        finstrument[instrument] = {}
        if 'sigma_w_'+instrument in out['posterior_samples'].keys():
            sigma_w = np.median(out['posterior_samples']['sigma_w_'+instrument])
        else:
            sigma_w = priors['sigma_w_'+instrument]['cvalue']
        F1med,F2med = np.median(norm_factors[instrument]['F1']),np.median(norm_factors[instrument]['F2'])
        finstrument[instrument]['flux'] = (f_lc[instrument_indexes_lc[instrument]] - F2med)/F1med
        finstrument[instrument]['flux_error'] = np.sqrt(ferr_lc[instrument_indexes_lc[instrument]]**2 + (sigma_w*1e-6)**2)/F1med

    # Now, let's begin with the phased transit plots of each planet for each instrument on different plots:
    for nplanet in range(n_transit):
      iplanet = numbering_transit[nplanet]
      print(r'\t Generating phased plot for planet '+ str(iplanet))

      for instrument in inames_lc:
        if lc_dictionary[instrument]['TransitFit']:
            print(r'\t Generating phased plot for planet '+ str(iplanet) + ' and instrument ' + instrument)

            fig, axs = plt.subplots(2, 1,gridspec_kw = {'height_ratios':[3,1]}, figsize=(9,7))
            sns.set_context("talk")
            sns.set_style("ticks")
            matplotlib.rcParams['mathtext.fontset'] = 'stix'
            matplotlib.rcParams['font.family'] = 'STIXGeneral'
            matplotlib.rcParams['font.size'] = '5'
            matplotlib.rcParams['axes.linewidth'] = 1.2
            matplotlib.rcParams['xtick.direction'] = 'out'
            matplotlib.rcParams['ytick.direction'] = 'out'
            matplotlib.rcParams['lines.markeredgewidth'] = 1

            # First, get phases for the current planetary model. For this get median period and t0 (if there were not fixed):
            if priors['P_p'+str(iplanet)]['type'] != 'fixed':
                P = np.median(out['posterior_samples']['P_p'+str(iplanet)])
            else:
                P = priors['P_p'+str(iplanet)]['cvalue']
            if priors['t0_p'+str(iplanet)]['type'] != 'fixed':
                t0 = np.median(out['posterior_samples']['t0_p'+str(iplanet)])
            else:
                t0 = priors['t0_p'+str(iplanet)]['cvalue']
            # Get the actual phases:
            phases = utils.get_phases(t_lc[instrument_indexes_lc[instrument]],P,t0)

            # Now, as in the previous plot, sample models from the posterior parameters along the phases of interest. 
            # For this, first define a range of phases of interest:
            model_phases = np.linspace(-0.25,0.25,10000)
            # With this get the respective times for the model phases:
            t_model_phases = model_phases*P + t0

            # Initialize model:
            if lc_dictionary[instrument]['resampling']:
                params_model, m_model = init_batman(t_model_phases, law=lc_dictionary[instrument]['ldlaw'], \
                                        n_ss=lc_dictionary[instrument]['nresampling'], \
                                        exptime_ss=lc_dictionary[instrument]['exptimeresampling'])
            else:
                params_model, m_model = init_batman(t_model_phases, law=lc_dictionary[instrument]['ldlaw'])

            # Now generate the (oversampled and "real", with the real data sampling) models:
            all_lc_models = np.zeros([nsims,len(t_model_phases)])
            all_lc_real_models = np.zeros([nsims,len(phases)])
            # Define vector that will save the planetary model *without* the current planet (to remove it from the data):
            all_lc_real_models_no_planet = np.ones([nsims,len(phases)])
            lcmodel = np.ones(len(t_model_phases))
            lcmodel_real = np.ones(len(phases))
            counter = -1
            for j in idx_sims:
                counter = counter + 1
                # Sample the jth sample of parameter values:
                for pname in priors.keys():
                    if priors[pname]['type'] != 'fixed':
                        priors[pname]['cvalue'] = out['posterior_samples'][pname][j]

                # First, generate the model for the planet under consideration:
                if lc_dictionary[instrument]['ldlaw'] != 'linear':
                    coeff1,coeff2 = reverse_ld_coeffs(lc_dictionary[instrument]['ldlaw'],priors['q1_'+ld_iname[instrument]]['cvalue'],\
                                    priors['q2_'+ld_iname[instrument]]['cvalue'])
                    params_model.u = [coeff1,coeff2]
                else:
                    params_model.u = [priors['q1_'+ld_iname[instrument]]['cvalue']]

                if efficient_bp[nplanet]:
                    if not fitrho:
                        a,r1,r2,t0,P = priors['a_p'+str(iplanet)]['cvalue'],priors['r1_p'+str(iplanet)]['cvalue'],\
                                       priors['r2_p'+str(iplanet)]['cvalue'], priors['t0_p'+str(iplanet)]['cvalue'], \
                                       priors['P_p'+str(iplanet)]['cvalue']
                    else:
                        rho,r1,r2,t0,P = priors['rho']['cvalue'],priors['r1_p'+str(iplanet)]['cvalue'],\
                                       priors['r2_p'+str(iplanet)]['cvalue'], priors['t0_p'+str(iplanet)]['cvalue'], \
                                       priors['P_p'+str(iplanet)]['cvalue']
                        a = ((rho*G*((P*24.*3600.)**2))/(3.*np.pi))**(1./3.)
                    if r1 > Ar:
                        b,p = (1+pl)*(1. + (r1-1.)/(1.-Ar)),\
                              (1-r2)*pl + r2*pu
                    else:
                        b,p = (1. + pl) + np.sqrt(r1/Ar)*r2*(pu-pl),\
                              pu + (pl-pu)*np.sqrt(r1/Ar)*(1.-r2)
                else:
                    if not fitrho:
                        a,b,p,t0,P = priors['a_p'+str(iplanet)]['cvalue'],priors['b_p'+str(iplanet)]['cvalue'],\
                                     priors['p_p'+str(iplanet)]['cvalue'], priors['t0_p'+str(iplanet)]['cvalue'], \
                                     priors['P_p'+str(iplanet)]['cvalue']
                    else:
                        rho,b,p,t0,P = priors['rho']['cvalue'],priors['b_p'+str(iplanet)]['cvalue'],\
                                       priors['p_p'+str(iplanet)]['cvalue'], priors['t0_p'+str(iplanet)]['cvalue'], \
                                       priors['P_p'+str(iplanet)]['cvalue']
                        a = ((rho*G*((P*24.*3600.)**2))/(3.*np.pi))**(1./3.)

                if ecc_parametrization['transit'][iplanet] == 0:
                    ecc,omega = priors['ecc_p'+str(iplanet)]['cvalue'],priors['omega_p'+str(iplanet)]['cvalue']
                elif ecc_parametrization['transit'][iplanet] == 1:
                    ecc = np.sqrt(priors['ecosomega_p'+str(iplanet)]['cvalue']**2+priors['esinomega_p'+str(iplanet)]['cvalue']**2)
                    omega = np.arctan2(priors['esinomega_p'+str(iplanet)]['cvalue'],priors['ecosomega_p'+str(iplanet)]['cvalue'])*(180/np.pi)
                else:
                    ecc = priors['secosomega_p'+str(iplanet)]['cvalue']**2+priors['sesinomega_p'+str(iplanet)]['cvalue']**2
                    omega = np.arctan2(priors['sesinomega_p'+str(iplanet)]['cvalue'],priors['secosomega_p'+str(iplanet)]['cvalue'])*(180/np.pi)

                ecc_factor = (1. + ecc*np.sin(omega * np.pi/180.))/(1. - ecc**2)
                inc_inv_factor = (b/a)*ecc_factor
                if not (b>1.+p or inc_inv_factor >=1.):
                    inc = np.arccos(inc_inv_factor)*180./np.pi
                    params_model.t0 = t0
                    params_model.per = P
                    params_model.rp = p
                    params_model.a = a
                    params_model.inc = inc
                    params_model.ecc = ecc
                    params_model.w = omega

                    all_lc_models[counter,:] = m_model.light_curve(params_model)
                    all_lc_real_models[counter,:] = lc_dictionary[instrument]['m'].light_curve(params_model)
                        
                # Now, generate the model for all the planets *minus* the planet in consideration: 
                for n in range(n_transit):
                  i = numbering_transit[n]
                  if i != iplanet:
                    if lc_dictionary[instrument]['ldlaw'] != 'linear':
                        coeff1,coeff2 = reverse_ld_coeffs(lc_dictionary[instrument]['ldlaw'],priors['q1_'+ld_iname[instrument]]['cvalue'],\
                                        priors['q2_'+ld_iname[instrument]]['cvalue'])
                        lc_dictionary[instrument]['params'].u = [coeff1,coeff2]
                    else:
                        lc_dictionary[instrument]['params'].u = [priors['q1_'+ld_iname[instrument]]['cvalue']]

                    if efficient_bp[n]:
                        if not fitrho:
                            a,r1,r2,t0,P = priors['a_p'+str(i)]['cvalue'],priors['r1_p'+str(i)]['cvalue'],\
                                           priors['r2_p'+str(i)]['cvalue'], priors['t0_p'+str(i)]['cvalue'], \
                                           priors['P_p'+str(i)]['cvalue']
                        else:
                            rho,r1,r2,t0,P = priors['rho']['cvalue'],priors['r1_p'+str(i)]['cvalue'],\
                                             priors['r2_p'+str(i)]['cvalue'], priors['t0_p'+str(i)]['cvalue'], \
                                             priors['P_p'+str(i)]['cvalue']
                            a = ((rho*G*((P*24.*3600.)**2))/(3.*np.pi))**(1./3.)
                        if r1 > Ar:
                            b,p = (1+pl)*(1. + (r1-1.)/(1.-Ar)),\
                                  (1-r2)*pl + r2*pu
                        else:
                            b,p = (1. + pl) + np.sqrt(r1/Ar)*r2*(pu-pl),\
                                  pu + (pl-pu)*np.sqrt(r1/Ar)*(1.-r2)
                    else:
                        if not fitrho:
                            a,b,p,t0,P = priors['a_p'+str(i)]['cvalue'],priors['b_p'+str(i)]['cvalue'],\
                                         priors['p_p'+str(i)]['cvalue'], priors['t0_p'+str(i)]['cvalue'], \
                                         priors['P_p'+str(i)]['cvalue']
                        else:
                            rho,b,p,t0,P = priors['rho']['cvalue'],priors['b_p'+str(i)]['cvalue'],\
                                         priors['p_p'+str(i)]['cvalue'], priors['t0_p'+str(i)]['cvalue'], \
                                         priors['P_p'+str(i)]['cvalue']
                            a = ((rho*G*((P*24.*3600.)**2))/(3.*np.pi))**(1./3.)
                    if ecc_parametrization['transit'][i] == 0:
                        ecc,omega = priors['ecc_p'+str(i)]['cvalue'],priors['omega_p'+str(i)]['cvalue']
                    elif ecc_parametrization['transit'][i] == 1:
                        ecc = np.sqrt(priors['ecosomega_p'+str(i)]['cvalue']**2+priors['esinomega_p'+str(i)]['cvalue']**2)
                        omega = np.arctan2(priors['esinomega_p'+str(i)]['cvalue'],priors['ecosomega_p'+str(i)]['cvalue'])*180./np.pi
                    else:
                        ecc = priors['secosomega_p'+str(i)]['cvalue']**2+priors['sesinomega_p'+str(i)]['cvalue']**2
                        omega = np.arctan2(priors['sesinomega_p'+str(i)]['cvalue'],priors['secosomega_p'+str(i)]['cvalue'])*180./np.pi

                    ecc_factor = (1. + ecc*np.sin(omega * np.pi/180.))/(1. - ecc**2)
                    inc_inv_factor = (b/a)*ecc_factor
                    inc = np.arccos(inc_inv_factor)*180./np.pi
                    lc_dictionary[instrument]['params'].t0 = t0
                    lc_dictionary[instrument]['params'].per = P
                    lc_dictionary[instrument]['params'].rp = p
                    lc_dictionary[instrument]['params'].a = a
                    lc_dictionary[instrument]['params'].inc = inc
                    lc_dictionary[instrument]['params'].ecc = ecc
                    lc_dictionary[instrument]['params'].w = omega
                    all_lc_real_models_no_planet[counter,:] = all_lc_real_models_no_planet[counter,:]*\
                                                        lc_dictionary[instrument]['m'].light_curve(lc_dictionary[instrument]['params'])
            # As before, once again compute median model and the respective error bands:
            omedian_model = np.zeros(len(t_model_phases))
            omodel_up1, omodel_down1 = np.zeros(len(t_model_phases)),np.zeros(len(t_model_phases))
            omodel_up2, omodel_down2 = np.zeros(len(t_model_phases)),np.zeros(len(t_model_phases))
            omodel_up3, omodel_down3 = np.zeros(len(t_model_phases)),np.zeros(len(t_model_phases))

            # Uncomment this line (and comment the ax = axs[0] below) to see the model samples in the transit
            # plots:
            #ax = axs[0]
            #for ii in range(all_lc_models.shape[0]):
            #    ax.plot(model_phases,all_lc_models[ii,:],color='grey',alpha=0.1)
            for i_tsample in range(len(t_model_phases)):
                # Compute quantiles for the full model:
                val,valup1,valdown1 = utils.get_quantiles(all_lc_models[:,i_tsample])
                val,valup2,valdown2 = utils.get_quantiles(all_lc_models[:,i_tsample],alpha=0.95)
                val,valup3,valdown3 = utils.get_quantiles(all_lc_models[:,i_tsample],alpha=0.99)
                omedian_model[i_tsample] = val
                omodel_up1[i_tsample],omodel_down1[i_tsample] = valup1,valdown1
                omodel_up2[i_tsample],omodel_down2[i_tsample] = valup2,valdown2
                omodel_up3[i_tsample],omodel_down3[i_tsample] = valup3,valdown3

            lcmodel = np.zeros(len(phases))
            lcmodel_noplanet = np.zeros(len(phases))
            # Do the same for the "real" sampling of the data, and the model without the planet in consideration:
            for i_tsample in range(len(phases)):
                #val,valup1,valdown1 = utils.get_quantiles(all_lc_real_models[:,i_tsample])
                lcmodel[i_tsample] = np.median(all_lc_real_models[:,i_tsample])
                lcmodel_noplanet[i_tsample] = np.median(all_lc_real_models_no_planet[:,i_tsample])

            ax = axs[0]
            # Calculate time baseline of observations. Useful to define bounds of the plotted data:
            tbaseline = np.max(t_lc[instrument_indexes_lc[instrument]])-\
                        np.min(t_lc[instrument_indexes_lc[instrument]])

            # Plot data. Alphas defined whether the time-baseline is 
            # short (<0.5, most likely ground-based data) or large 
            # (most likely space-based data):
            if tbaseline < 0.5 or lc_dictionary[instrument]['resampling']:
                alpha_notbinned = 0.5
                alpha_binned = 0.8
            else:
                alpha_notbinned = 0.2
                alpha_binned = 0.5
            

            if tbaseline > 0.5 and (not lc_dictionary[instrument]['resampling']):
                ax.plot(phases,finstrument[instrument]['flux']/lcmodel_noplanet,'.k',markersize=5,alpha=alpha_notbinned)
                idx_sort = np.argsort(phases)
                phases_bin,f_bin,f_bin_err = utils.bin_data(phases[idx_sort],\
                                             finstrument[instrument]['flux'][idx_sort]/lcmodel_noplanet[idx_sort],15)
                ax.errorbar(phases_bin,f_bin,yerr=f_bin_err,fmt='.k',markersize=5,elinewidth=1,alpha=alpha_binned)
            else:
                ax.errorbar(phases,finstrument[instrument]['flux']/lcmodel_noplanet,\
                            yerr = finstrument[instrument]['flux_error']/lcmodel_noplanet,\
                            fmt='.k',markersize=5,alpha=alpha_notbinned,elinewidth=1)

            fout = open(out_folder+'phased_lc_planet'+str(iplanet)+'_'+instrument+'.dat','w')
            fout.write('# Phases \t Time \t Phased LC \t Phased LC Error \t Model\n')
            for i in range(len(phases)):
                fout.write('{0:.10f} {1:.10f} {2:.10f} {3:.10f} {4:.10f}\n'.format(phases[i],t_lc[i],(finstrument[instrument]['flux']/lcmodel_noplanet)[i],\
                                                                          (finstrument[instrument]['flux_error']/lcmodel_noplanet)[i],lcmodel[i]))
            fout.close() 
            # Now, define the phase at which the lightcurve model goes to 1, so we find the ingress and egress 
            # time in phase space:
            idx = np.where(lcmodel == 1)[0]
            idx_min_phase = np.where(np.abs(phases[idx]) == np.min(np.abs(phases[idx])))[0]
            if len(idx_min_phase) > 1:
                idx_min_phase = idx_min_phase[0]
            min_phase = np.abs(phases[idx][idx_min_phase])

            # Define also mean errorbars to define yaxes:
            sigma_median = np.median(finstrument[instrument]['flux_error']/lcmodel_noplanet)

            # Now plot the phased model. Compute sorting indexes as well and plot sorted phases:
            ax.fill_between(model_phases,omodel_down1,omodel_up1,color='cornflowerblue',alpha=0.25)
            ax.fill_between(model_phases,omodel_down2,omodel_up2,color='cornflowerblue',alpha=0.25)
            #ax.fill_between(model_phases,omodel_down3,omodel_up3,color='cornflowerblue',alpha=0.25)
            ax.plot(model_phases,omedian_model,'-',linewidth=2,color='black')
            ax.set_ylabel('Relative flux')

            if not efficient_bp[nplanet]:
                depth = np.median(out['posterior_samples']['p_p'+str(iplanet)])**2#priors['p_p'+str(iplanet)]['cvalue']
            else:
                depth = np.array([])
                for i in range(len(out['posterior_samples']['r1_p'+str(iplanet)])):
                    r1,r2 = out['posterior_samples']['r1_p'+str(iplanet)][i],out['posterior_samples']['r2_p'+str(iplanet)][i]
                    if r1 > Ar:
                        b,p = (1+pl)*(1. + (r1-1.)/(1.-Ar)),\
                              (1-r2)*pl + r2*pu
                    else:
                        b,p = (1. + pl) + np.sqrt(r1/Ar)*r2*(pu-pl),\
                              pu + (pl-pu)*np.sqrt(r1/Ar)*(1.-r2)       
                    depth = np.append(depth,p**2)
                depth = np.median(depth)    

            ax.get_xaxis().set_major_formatter(plt.NullFormatter())
            ax.set_xlim([-2*min_phase,2*min_phase])
            #### CHANGE Y AXIS PHASE PHOT PLOT:
            if phasefluxlims:
                ax.set_ylim([ylow_pflux,yhigh_pflux])
            else:
                ax.set_ylim([1. - depth - sigma_median*10,1. + sigma_median*5])
            #ax.set_ylim([0.997,1.002])
            #if lc_dictionary[instrument]['GPDetrend']:
            #    ax.set_ylim([1- depth - depth*0.5,1.001 + depth*0.5+0.001])
            #else:
            #    if depth*1e6 > 1000.:
            #        ax.set_ylim([1- depth - depth*0.5 -1000*1e-6,1.001 + depth*0.2])
            #    else:
            #        ax.set_ylim([1 - 1000*1e-6,1.001 + depth*0.5])

            # Define the x-axis limits based on time baseline of observations. Basically if it is larger than 
            # half a day, it is most likely space-based and we thus base our plot around the phased transit event. 
            # If not, we base our plot around the expected ingress and egress:
            #if tbaseline>0.5:
            #    if depth*1e6 > 1000.:
            #        ax.set_xlim([-0.03,0.03])
            #    else:
            #        ax.set_xlim([-0.15,0.15])
            #else:
            #    ax.set_xlim([np.min(phases),np.max(phases)])
            #ax.get_xaxis().set_major_formatter(plt.NullFormatter())

            # Plot residuals:
            ax2 = axs[1]
            # Plot zero line to guide the eye:
            ax2.plot([-1e10,1e10],[0.,0.],'--',linewidth=2,color='black')
            # Plot residuals:
            if tbaseline < 0.5 or lc_dictionary[instrument]['resampling']:
                ax2.errorbar(phases,(finstrument[instrument]['flux']/lcmodel_noplanet-lcmodel)*1e6,\
                            yerr=(finstrument[instrument]['flux_error']/lcmodel_noplanet)*1e6,\
                            fmt='.k',markersize=5,elinewidth=1,alpha=alpha_notbinned)
            else:
                ax2.plot(phases,(finstrument[instrument]['flux']/lcmodel_noplanet-lcmodel)*1e6,'.k',markersize=5,alpha=alpha_notbinned)
                idx_argsort = np.argsort(phases)
                phases_bin,f_bin,f_bin_err = utils.bin_data(phases[idx_argsort],\
                                             (finstrument[instrument]['flux'][idx_argsort]/lcmodel_noplanet[idx_argsort]-\
                                             lcmodel[idx_argsort])*1e6,15)
                ax2.errorbar(phases_bin,f_bin,yerr=f_bin_err,fmt='.k',markersize=5,elinewidth=1,alpha=alpha_binned)
            ax2.set_ylabel('Residuals (ppm)')
            ax2.set_xlabel('Phase')
            ax2.set_xlim([-2*min_phase,2*min_phase])
            # CHANGE PHASE PLOT PHOT RESIDUALS:
            ax2.set_ylim([-sigma_median*5*1e6,sigma_median*5*1e6])
            #ax2.set_ylim([-2000,2000])
            #if tbaseline>0.5:
            #    if depth*1e6 > 1000.:
            #        ax.set_xlim([-0.03,0.03])
            #        ax.set_ylim([-2000,2000])
            #    else:
            #        ax.set_xlim([-0.15,0.15])
            #        ax.set_ylim([-1000,1000])
            #else:
            #    ax.set_xlim([np.min(phases),np.max(phases)])
            plt.tight_layout()
            plt.savefig(out_folder+'phot_planet'+str(iplanet)+'_instrument_'+instrument+'.pdf')
