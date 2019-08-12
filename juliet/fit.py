# Import batman, for lightcurve models:
import batman
# Import radvel, for RV models:
import radvel
# Import george for detrending:
import george
# Import celerite for detrending:
import celerite
# Import dynesty for dynamic nested sampling:
try:
    import dynesty
    from dynesty.utils import resample_equal
    force_pymultinest = False
except:
    print('Dynesty installation not detected. Forcing pymultinest.')
    force_pymultinest = True
# Import multinest for (importance) nested sampling:
try:
    import pymultinest
    force_dynesty = False
except:
    print('(py)MultiNest installation (or libmultinest.dylib) not detected. Forcing sampling with dynesty.')
    force_dynesty = True

# Import generic useful classes:
import os
import sys
import numpy as np

# Prepare the celerite term:
import celerite
from celerite import terms

from .utils import *
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

__all__ = ['fit','gaussian_process'] 

class fit(object):
    """
    Given a dictionary with priors and a dataset either given through arrays or through files containing the data, 
    this class performs a juliet fit and returns an object with the results. Example usage:

               >>> out = juliet.fit(priors,t_lc=times,y_lc=fluxes,yerr_lc=fluxes)

    :param priors: (dict)                   
        A python ``dict`` containing each of the parameters to be fit, along with their 
        respective prior distributions and hyperparameters. Each key of this dictionary 
        has to have a parameter name (e.g., ``r1_p1``, ``sigma_w_TESS``), and each of 
        those elements are, in turn, dictionaries as well containing two keys: a ``distribution``
        key which defines the prior distribution of the parameter and a ``hyperparameters`` key, 
        which contains the hyperparameters of that distribution. 

        Example setup of the ``priors`` dictionary:
            >> priors = {}
            >> priors['r1_p1'] = {}
            >> priors['r1_p1']['distribution'] = 'Uniform'
            >> priors['r1_p1']['hyperparameters'] = [0.,1.]


    :param t_lc: (optional, array of floats)
        Array containing the times corresponding to the lightcurve data.

    :param y_lc: (optional, array of floats)
        Array containing the relative fluxes of the lightcurve data at each time ``t_lc``.     

    :param yerr_lc: (optional, array of floats)
        Array containig the errors on the relative fluxes ``y_lc`` at each time ``t_lc``.

    :param instruments_lc: (optional, array of strings) 
        Array containing the names (``strings``) of the instrument corresponding to each of the 
        ``t_lc``, ``y_lc`` and ``yerr_lc`` datapoints. If not given, this is set to an array of length 
        ``len(t_lc)``, where each element is named ``Data``.
   
    :param GP_regressors_lc: (optional, multi-dimensional array of floats) 
        Array of shape ``(m,n)`` containing in each column the ``n`` GP regressors to be used for 
        ``m`` photometric measurements. Note that ``m <= len(t_lc)``. Also, note the order of each regressor 
        of each instrument has to match the corresponding order in the ``t_lc`` and ``instrument_lc`` arrays.

    :param GP_instruments_lc: (optional, array of strings) 
        Array of length ``m`` indicating the names of the instruments to which to apply the GP regression. 
        If not given, it is assumed that ``m = len(t_lc)`` and the GP will apply to all the photometric measurements. 

    :param linear_regressors_lc: (optional, multi-dimensional array of floats) 
        Array of shape ``(q,p)`` containing in each column the ``p`` linear regressors to be used for 
        ``q`` photometric measurements. Note that ``q <= len(t_lc)``. Also, note the order of each regressor 
        of each instrument has to match the corresponding order in the ``t_lc`` and ``instrument_lc`` arrays.

    :param linear_instruments_lc: (optional, array of strings)
        Array of length ``q`` indicating the names of the instruments to which to apply the linear regression.

    :param GP_regressors_rv: (optional, multi-dimensional array of floats)  
        Same as ``GP_regressors_lc`` but for the radial-velocity data. 

    :param GP_instruments_rv: (optional, array of strings)
        Same as ``GP_instruments_lc`` but for the radial-velocities.     

    :param linear_regressors_rv: (optional, multi-dimensional array of floats) 
        Same as ``linear_regressors_lc``, but for the radial-velocities.

    :param linear_instruments_rv: (optional, array of strings)
        Sames as ``linear_instruments_rv``, but for the radial-velocities.

    :param t_rv: (optional, array of floats)                    
        Same as ``t_lc``, but for the radial-velocities.
   
    :param y_rv: (optional, array of floats)
        Same as ``y_lc``, but for the radial-velocities.

    :param yerr_rv: (optional, array of floats)
        Same as ``yerr_lc``, but for the radial-velocities.

    :param george_hodlr: (optional, boolean)             
        If ``True``, use the HODLR solver for george Gaussian Process evaluation. Default is ``False``.
   
    :param use_dynesty: (optional, boolean)              
        If ``True``, use dynesty instead of `MultiNest` for posterior sampling and evidence evaluation. Default is 
        ``False``, unless `MultiNest` via ``pymultinest`` is not working on the system.
 
    :param dynamic: (optional, boolean)                 
        If ``True``, use dynamic Nested Sampling with dynesty. Default is ``False``.
 
    :param dynesty_bound: (optional, string)           
        Define the dynesty bound method to use (currently either ``single`` or ``multi``, to use either single ellipsoids or multiple 
        ellipsoids). Default is ``multi`` (for details, see the `dynesty API <https://dynesty.readthedocs.io/en/latest/api.html>`_).

    :param dynesty_sample: (optional, string)         
        Define the sampling method for dynesty to use. Default is ``rwalk``. Accorfing to the `dynesty API <https://dynesty.readthedocs.io/en/latest/api.html>`_, 
        this should be changed depending on the number of parameters being fitted. If smaller than about 20, ``rwalk`` is optimal. For larger dimensions, 
        ``slice`` or ``rslice`` should be used.

    :param dynesty_nthreads: (optional, int)        
        Define the number of threads to use within dynesty. Default is to use just 1.

    :param out_folder: (optional, string) 
        If a path is given, results will be saved to that path as a ``pickle`` file.

    :param lcfilename:  (optional, string)             
        If a path to a lightcurve file is given, ``t_lc``, ``y_lc``, ``yerr_lc`` and ``instruments_lc`` will be read from there. The basic file format is a pure 
        ascii file where times are in the first column, relative fluxes in the second, errors in the third and instrument names in the fourth. If more columns are given for 
        a given instrument, those will be identified as linear regressors for those instruments.

    :param rvfilename: (optional, string)               
        Same as ``lcfilename``, but for the radial-velocities.

    :param GPlceparamfile: (optional, string)          
        If a path to a file is given, the columns of that file will be used as GP regressors for the lightcurve fit. The file format is a pure ascii file 
        where regressors are given in different columns, and the last column holds the instrument name. The order of this file has to be consistent with 
        ``t_lc`` and/or the ``lcfilename`` file.

    :param GPrveparamfile: (optional, string)          
        Same as ``GPlceparamfile`` but for the radial-velocities.

    :param lctimedef: (optional, string)               
        Time definitions for each of the lightcurve instruments. Default is ``TDB`` for all instruments. If more than one instrument is given, this string 
        should have instruments and time-definitions separated by commas, e.g., ``TESS-TDB, LCOGT-UTC``, etc.

    :param rvtimedef: (optional, string)               
        Time definitions for each of the radial-velocity instruments. Default is ``UTC`` for all insstruments. If more than one instrument is given, 
        this string should have instruments and time-definitions separated by commas, e.g., ``FEROS-TDB, HARPS-UTC``, etc.

    :param ld_laws: (optional, string)                 
        Limb-darkening law to be used for each instrument. Default is ``quadratic`` for all instruments. If more than one instrument is given, 
        this string should have instruments and limb-darkening laws separated by commas, e.g., ``TESS-quadratic, LCOGT-linear``.

    :param priorfile: (optional, string)                
        If a path to a file is given, it will be assumed this is a prior file. The ``priors`` dictionary will be overwritten by the data in this 
        file. The file structure is a plain ascii file, with the name of the parameters in the first column, name of the prior distribution in the 
        second column and hyperparameters in the third column.

    :param pl: (optional, float)                      
        If the ``(r1,r2)`` parametrization for ``(b,p)`` is used, this defines the lower limit of the planet-to-star radius ratio to be sampled. 
        Default is ``0``.

    :param pu: (optional, float)                    
        Same as ``pl``, but for the upper limit. Default is ``1``.

    :param n_live_points: (optional, int)            
        Number of live-points to be sampled. Default is ``500``.

    :param ecclim: (optional, float)                   
        Upper limit on the maximum eccentricity to sample. Default is ``1``.

    :param instrument_supersamp: (optional, array of strings)     
        Define for which lightcurve instruments super-sampling will be applied (e.g., in the case of long-cadence integrations).

    :param n_supersamp: (optional, array of ints)              
        Define the number of datapoints to supersample. Order should be consistent with order in ``instrument_supersamp``.

    :param exptime_supersamp: (optional, array of floats)        
        Define the exposure-time of the observations for the supersampling. Order should be consistent with order in ``instrument_supersamp``.

    """
    def data_preparation(self,times,instruments,linear_regressors,linear_instruments):
        """
        This function generates four useful internal arrays for this class: inames which saves the instrument names,
        instrument_indexes, which saves the indexes corresponding to each instrument, lm_boolean which saves booleans for each 
        instrument to indicate if there are linear regressors and lm_arguments which are the linear-regressors for each instrument.
        """
        inames = []
        for i in range(len(times)):
            if instruments[i] not in inames:
                inames.append(instruments[i])
        ninstruments = len(inames)
        instrument_indexes = {}
        for instrument in inames:
            instrument_indexes[instrument] = np.where(instruments == instrument)[0]

        # Also generate lm_lc_boolean and lm_lc_arguments in case linear regressors were passed:
        lm_boolean = {}
        lm_arguments = {}
        if linear_regressors is not None:
            for instrument in inames:
                if instrument in linear_instruments:
                    lm_boolean[instrument] = True
                    lm_arguments[instrument] = linear_regressors[instrument_indexes[instrument],:]
                else:
                    lm_boolean[instrument] = False
        else:
            for instrument in inames:
                lm_boolean[instrument] = False
        return inames, instrument_indexes, lm_boolean, lm_arguments

    def save_regressors(self,fname,GP_regressors,GP_instruments):
        """
        This function saves the GP regressors to fname.
        """
        fout = open(fname,'w')
        for i in range(GP_regressors.shape[0]):
            for j in range(GP_regressors.shape[1]):
                fout.write('{0:.10f} '.format(GP_regressors[i,j]))
            if GP_instruments is not None:
                fout.write('{0:}\n'.format(GP_instruments[i]))
            else:
                fout.write('\n')
        fout.close()

    def save_data(self,fname,t,y,yerr,instruments,lm_boolean,lm_arguments):
        """
        This function saves t,y,yerr,instruments,lm_boolean and lm_arguments data to fname.
        """
        fout = open(fname,'w')
        lm_counters = {}
        for i in range(len(t)):
            fout.write('{0:.10f} {1:.10f} {2:.10f} {3:}'.format(t[i],y[i],yerr[i],instruments[i]))
            if lm_boolean[instruments[i]]:
                if instruments[i] not in lm_counters.keys():
                    lm_counters[instruments[i]] = 0 
                for j in range(lm_arguments[instruments[i]].shape[1]):
                    fout.write(' {0:.10f}'.format(lm_arguments[instruments[i]][lm_counters[instruments[i]]][j]))
                lm_counters[instruments[i]] += 1
            fout.write('\n')
        fout.close()

    def generate_datadict(inames, priors, dictype, resampling = False, GPDetrend = False, TransitFit = False, \
                          ld_laws = None, instrument_supersamp = None, n_supersamp = None, exptime_supersamp = None):

        """
        This generates the options dictionary for lightcurves, RVs, and everything else you want to fit. All parameters 
        were already defined for this class, with the exception of:

        :param dictype: (string)
            Defines the type of dictionary type. It can either be 'lc' (for the lightcurve dictionary) or 'rv' (for the 
            radial-velocity one). 
        """

        dictionary = {}
        ninstruments = len(inames)
        for i in range(ninstruments):
            dictionary[inames[i]] = {}
            # Save if a given instrument will receive resampling (initialize this as False):
            dictionary[inames[i]]['resampling'] = False
            # Save if a given instrument has GP fitting ON (initialize this as False):
            dictionary[inames[i]]['GPDetrend'] = False
            # Save if transit fitting will be done for a given dataset/instrument (this is so users can fit photometry with, e.g., GPs):
            if dictype == 'lc':
                dictionary[inames[i]]['TransitFit'] = False
        if ld_laws is not None:
            # Extract limb-darkening law. If just one is given, assume same LD law for all instruments. If not, assume a
            # different law for each instrument:
            ld_laws = args.ldlaw.split(',')
            if len(ld_laws) == 1:
                for i in range(ninstruments):
                    dictionary[inames[i]]['ldlaw'] = (ld_laws[0].split('-')[-1]).split()[0].lower()
            else:
                for ld_law in ld_laws:
                    instrument,ld = ld_law.split('-')
                    dictionary[instrument.split()[0]]['ldlaw'] = ld.split()[0].lower()

        # Extract supersampling parameters. If not given for each instrument, assume all must be resampled 
        # unless everything is None in which case nothing is done.
        if instrument_supersamp is not None:
            instrument_ss = instrument_supersamp.split(',')
            n_ss = np.array(n_supersamp.split(',')).astype('int')
            exptime_ss = np.array(exptime_supersamp.split(',')).astype('double')
            for i in range(len(instrument_ss)):
                dictionary[instrument_ss[i]]['resampling'] = True
                dictionary[instrument_ss[i]]['nresampling'] = n_ss[i]
                dictionary[instrument_ss[i]]['exptimeresampling'] = exptime_ss[i]
        else:
            if n_supersamp is not None:
                n_ss = int(n_supersamp)
            else:      
                n_ss = n_supersamp
            if exptime_supersamp is not None:
                for i in range(ninstruments_lc):
                    dictionary[inames[i]]['resampling'] = True
                    dictionary[inames[i]]['nresampling'] = n_ss
                    dictionary[inames[i]]['exptimeresampling'] = exptime_ss

        # Now, if generating lightcurve dict, check whether for some photometric instruments only photometry, and not a 
        # transit, will be fit:
        if dictype == 'lc':
            for i in range(ninstruments):
                for pri in priors.keys():
                    if pri[0:2] == 'q1':
                        if inames[i] in pri.split('_'):
                            dictionary[inames[i]]['TransitFit'] = True
                            print('\t Transit fit detected for instrument ',inames[i])

        

    def __init__(self,priors,t_lc = None, y_lc = None, yerr_lc = None, instruments_lc = None,\
                 t_rv = None, y_rv = None, yerr_rv = None, instruments_rv = None,\
                 GP_regressors_lc = None, GP_instruments_lc = None, linear_regressors_lc = None, \
                 linear_instruments_lc = None, GP_regressors_rv = None, GP_instruments_rv = None,\
                 linear_regressors_rv = None, linear_instruments_rv = None,\
                 george_hodlr = False, use_dynesty = False, dynamic = False, \
                 dynesty_bound = 'multi',dynesty_sample='rwalk',dynesty_nthreads = None,\
                 out_folder = None, lcfilename = None, rvfilename = None, GPlceparamfile = None,\
                 GPrveparamfile = None, lctimedef = 'TDB', rvtimedef = 'UTC',\
                 ld_laws = 'quadratic', priorfile = None,\
                 pl = 0., pu = 1., n_live_points = 1000, ecclim = 1., n_supersamp = None, exptime_supersamp = None, \
                 instrument_supersamp = None):

        # Define cases in which data is given through files: 
        if (t_lc is None) and (t_rv is None):
            if lcfilename is not None:
                t_lc,y_lc,yerr_lc,instruments_lc,instrument_indexes_lc,ninstruments_lc,inames_lc,lm_lc_boolean,lm_lc_arguments = \
                utils.read_data(lcfilename)
            if rvfilename is not None:
                t_rv,y_rv,yerr_rv,instruments_rv,instrument_indexes_rv,ninstruments_rv,inames_rv,lm_rv_boolean,lm_rv_arguments = \
                utils.read_data(rvfilename)
            if (lcfilename is None) and (rvfilename is None): 
                raise Exception('INPUT ERROR: No complete dataset (photometric or radial-velocity) given.\n'+\
                      ' Make sure to feed times (t_lc and/or t_rv), values (y_lc and/or y_rv), \n'+\
                      ' errors (yerr_lc and/or yerr_rv) and associated instruments (instruments_lc and/or instruments_rv).')

        # If data given through direct arrays (i.e., not data files), generate some useful internal lightcurve arrays: inames_lc, which have the different lightcurve instrument names, 
        # instrument_indexes_lc (dictionary that holds, for each instrument, the indexes that have the time/lightcurve data for that particular instrument), lm_lc_boolean (dictionary of 
        # booleans; True for an instrument if it has linear regressors) and lm_lc_arguments (dictionary containing the linear regressors for each instrument):
        if (lcfilename is None) and (t_lc is not None):
            input_error_catcher(t_lc,y_lc,yerr_lc,instruments_lc,'lightcurve')
            if type(instruments_lc) is list:
                instruments_lc = np.array(instruments_lc)
            inames_lc, instrument_indexes_lc, lm_lc_boolean, lm_lc_arguments = self.data_preparation(t_lc,instruments_lc,linear_regressors_lc,linear_instruments_lc)
            ninstruments_lc = len(inames_lc)

        # Same for radial-velocity data:
        if (rvfilename is None) and (t_rv is not None):
            input_error_catcher(t_rv,y_rv,yerr_rv,instruments_rv,'radial-velocity')
            if type(instruments_rv) is list:
                instruments_rv = np.array(instruments_rv)
            inames_rv, instrument_indexes_rv, lm_rv_boolean, lm_rv_arguments = self.data_preparation(t_rv,instruments_rv,linear_regressors_rv,linear_instruments_rv)
            ninstruments_rv = len(inames_rv)

        self.results = {}
        
        # Define prefixes in case saving is turned on (i.e., user passed an out_folder):
        if use_dynesty:
            if dynamic:
                prefix = 'dynamic_dynesty_'
            else:
                prefix = 'dynesty_'
        else:
            prefix = 'multinest_'

        # If out_folder does not exist, create it, and save data to it:
        if out_folder is not None:
            if out_folder[-1] != '/':
                out_folder = out_folder + '/'
            # First, save lightcurve data:
            if not os.path.exists(out_folder):
                os.mkdir(out_folder)
            if (not os.path.exists(out_folder+'lc.dat')): 
                if lcfilename is not None:
                    os.system('cp '+lcfilename+' '+out_folder+'lc.dat')
                elif t_lc is not None:
                    self.save_data(out_folder+'lc.dat',t_lc,y_lc,yerr_lc,instruments_lc,lm_lc_boolean,lm_lc_arguments)
            # Now radial-velocity data:
            if (not os.path.exists(out_folder+'rvs.dat')):
                if rvfilename is not None:
                    os.system('cp '+rvfilename+' '+out_folder+'rvs.dat')
                elif t_rv is not None:
                    self.save_data(out_folder+'rvs.dat',t_rv,y_rv,yerr_rv,instruments_rv,lm_rv_boolean,lm_rv_arguments)
            # Next, save GP regressors:
            if (not os.path.exists(out_folder+'GP_lc_regressors.dat')):
                if GPlceparamfile is not None:
                    os.system('cp '+rvfilename+' '+out_folder+'GP_lc_regressors.dat')
                elif GP_regressors_lc is not None:
                    self.save_regressors(out_folder+'GP_lc_regressors.dat',GP_regressors_lc,GP_instruments_lc)
            if (not os.path.exists(out_folder+'GP_rv_regressors.dat')):
                if GPrveparamfile is not None:
                    os.system('cp '+rvfilename+' '+out_folder+'GP_rv_regressors.dat')
                elif GP_regressors_rv is not None:
                    self.save_regressors(out_folder+'GP_rv_regressors.dat',GP_regressors_rv,GP_instruments_rv)

        # Now check if GP regressors are not None for lightcurves and RVs. If not, then initialize them:


class lc_model(object):
    """
    Given an array of times, this kernel initializes either a lightcurve model object for use within 
    the juliet library. Example usage:

               >>> model = juliet.lc_model(times)

    :param times: (array of floats)
        An array of floats that define the times at which the model will be evaluated.

    :param ld_law: (optional, string)
        String indicating the limb-darkening law to use. Can be either ``linear``, ``quadratic``, ``squareroot``, 
        or ``logarithmic``.

    :param instrument: (optional, string)
        Instrument to be modelled with the object. Useful for cross-talk with GP modelling and the juliter 
        ``posteriors`` dictionary.

    :param linear_regressors: (optional, multi-dimensional array of floats)
        Multi-dimensional array of length ``[len(times),n]`` with ``n`` linear regressors for the ``len(times)`` 
        photometric datapoints.

    :param GP_regressors: (optional, multi-dimensional array of floats)
        Multi-dimensional array of length ``[len(times),n]`` with ``n`` GP regressors for the ``len(times)`` 
        photometric datapoints.

    :param n_ss: (optional, int)
        Int indicating the number of points to super-sample in the lightcurve.

    :param exptime_ss: (optinal, float)
        Float indicating the exposure time of the observations that will be used for the super-sampling.

    """
    def init_batman(self):
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
         params.limb_dark = self.ld_law
         if self.n_ss is None or self.exptime_ss is None:
             m = batman.TransitModel(params, self.t)
         else:
             m = batman.TransitModel(params, self.t, supersample_factor=self.n_ss, exp_time=self.exptime_ss)
         return params,m

    def set_1pl_transit_parameters(self,t0,P,p,a,inc,q1,q2):
        coeff1,coeff2 = reverse_ld_coeffs(ld_law, q1, q2)
        self.params.t0 = t0
        self.params.per = P
        self.params.rp = p
        self.params.a = a 
        self.params.inc = inc
        if ld_law == 'linear':
            self.params.u = [coeff1]
        else:
            self.params.u = [coeff1,coeff2]

    def get_npl_transit_model(self,parameter_values, n_transit, numbering_transit):
        """
        This function generates a transit model for n_transit planets given the parameter_values 
        vector and the numbering_transit array, which contains the numbering of the transiting 
        planets (e.g. n_transit = [1,3] if 'p1' and 'p3' transit).
        """
        for n in range(n_transit):
            i = numbering_transit[n]
            if self.ld_law != 'linear':
                coeff1,coeff2 = reverse_ld_coeffs(lc_dictionary[instrument]['ldlaw'],priors['q1_'+ld_iname[instr     ument]]['cvalue']
            else:
         

    def get_1pl_transit_model(self):
        return self.m.light_curve(self.params)

    def __init__(self, times, yerr = None, instrument = None, ld_law = 'quadratic', linear_regressors = None, GP_regressors = None,\
                 kernel_name = None, n_ss = None, exptime_ss = None, george_hodlr = False):
        self.t = times
        self.ld_law = ld_law
        self.n_ss = n_ss
        self.exptime_ss = exptime_ss
        self.linear_regressors = linear_regressors
        self.GP_regressors = GP_regressors
        self.kernel_name = kernel_name
        self.yerr = yerr
        self.instrument = instrument
        # Initialize lightcurve for object, inhert the batman params and m objects:
        self.params, self.m = self.init_batman()
        # Initialize the GP object:
        self.GPmodel = gaussian_process(kernel_name = self.kernel_name, X = self.GP_regressors, model_type = 'lc', \
                                        yerr = self.yerr, instrument = self.instrument, george_hodlr = george_hodlr) 

class gaussian_process(object):
    """
    Given a kernel name and a matrix (or array) of external parameters X, this object generates a Gaussian Process 
    (GP) object to use within the juliet library. Example usage:

               >>> GPmodel = juliet.gaussian_process(kernel_name='SEKernel', X=X, model_type = 'lc', yerr=errors, instrument = 'TESS')

    :param kernel_name: (string) 
        String defining the name of the kernel to be used. It can either be ``SEKernel``, `ExpSineSquaredSEKernel``, ``CeleriteQPKernel``, 
        ``CeleriteExpKernel``, ``CeleriteMatern``, ``CeleriteMaternExpKernel`` or ``CeleriteSHOKernel``.

    :param X: (multi-dimensional array of floats)
        A multi-dimensional array containing the external parameters to be used by the GP. It has to be of dimensions ``[len(yerr),p]``.

    :param model_type: (string)
        A string defining the type of data the GP will be modelling. Can be either ``lc`` or ``rv``.

    :param yerr: (optional, array of floats)
        An array of floats containing the data errors.

    :param instrument: (optional, string)
        A string indicating the name of the instrument the GP is being applied to. This string simplifies cross-talk with juliet's ``posteriors`` 
        dictionary.

    """
    def init_GP(self):
        if self.use_celerite:
            self.GP = celerite.GP(self.kernel, mean=0.0)
        else:
            # (Note no jitter kernel is given, as with george one defines this in the george.GP call):
            jitter_term = george.modeling.ConstantModel(1.)
            if self.george_hodlr:
                self.GP = george.GP(self.kernel, mean = 0.0, fit_mean = False, white_noise = jitter_term,\
                                    fit_white_noise = True, solver = george.HODLRSolver)
            else:
                self.GP = george.GP(self.kernel, mean = 0.0, fit_mean = False, white_noise = jitter_term,\
                                    fit_white_noise = True)
        self.compute_GP()

    def compute_GP(self):
        if self.yerr is not None:
            self.GP.compute(self.X, yerr = self.yerr)
        else:
            self.GP.compute(self.X)    

    def set_input_instrument(self,input_variables):
        # This function sets the "input instrument" (self.input_instrument) name for each variable (self.variables). 
        # If, for example, GP_Prot_TESS_K2_RV and GP_Gamma_TESS, and self.variables = ['Prot','Gamma'], 
        # then self.input_instrument = ['TESS_K2_RV','TESS'].
        for i in range(len(self.variables)):
            GPvariable = self.variables[i]
            for pnames in input_variables.keys():
                vec = pnames.split('_')
                if (vec[0] == 'GP') and (GPvariable in vec[1]) and (self.instrument in vec):
                    self.input_instrument.append('_'.join(vec[2:]))

    def update_parameters(self, parameter_values):
        # To update the parameters, we have to transform the juliet inputs to celerite/george inputs. Update this 
        # depending on the kernel under usage. For this, we first define a base_index variable that will define the numbering 
        # of the self.parameter_vector. The reason for this is that the dimensions of the self.parameter_vector array is 
        # different if the GP is global (i.e., self.global_GP is True --- meaning a unique GP is fitted to all instruments) or 
        # not (self.global_GP is False --- meaning a different GP per instrument is fitted). If the former, the jitter terms are 
        # modified directly by changing the self.yerr vector; in the latter, we have to manually add a jitter term in the GP parameter 
        # vector.
        base_index = 0
        if self.kernel_name == 'SEKernel':
            if not self.global_GP:
                self.parameter_vector[base_index] = np.log((parameter_values['sigma_w_'+self.instrument]['cvalue']*self.sigma_factor)**2)
                base_index += 1
            self.parameter_vector[base_index] = np.log((parameter_values['GP_sigma_'+self.input_instrument[0]]['cvalue']*self.sigma_factor)**2.)
            for i in range(self.nX):
                self.parameter_vector[base_index + 1 + i] = np.log(1./priors['GP_alpha'+str(i)+'_'+self.input_instrument[1+i]]['cvalue'])
        elif self.kernel_name == 'ExpSineSquaredSEKernel':
            if not self.global_GP:
                self.parameter_vector[base_index] = np.log((parameter_values['sigma_w_'+self.instrument]['cvalue']*self.sigma_factor)**2)
                base_index += 1
            self.parameter_vector[base_index] = np.log((parameter_values['GP_sigma_'+self.input_instrument[0]]['cvalue']*self.sigma_factor)**2.)
            self.parameter_vector[base_index + 1] = np.log(1./(parameter_values['GP_alpha_'+self.input_instrument[1]]['cvalue']))
            self.parameter_vector[base_index + 2] = parameter_values['GP_Gamma_'+self.input_instrument[2]]['cvalue']
            self.parameter_vector[base_index + 3] = np.log(parameter_values['GP_Prot_'+self.input_instrument[3]]['cvalue'])
        elif self.kernel_name == 'CeleriteQPKernel':
            self.parameter_vector[0] = np.log(parameter_values['GP_B_'+self.input_instrument[0]]['cvalue'])
            self.parameter_vector[1] = np.log(parameter_values['GP_L_'+self.input_instrument[1]]['cvalue'])
            self.parameter_vector[2] = np.log(parameter_values['GP_Prot_'+self.input_instrument[2]]['cvalue'])
            self.parameter_vector[3] = np.log(parameter_values['GP_C_'+self.input_instrument[3]]['cvalue'])
            if not self.global_GP:
                self.parameter_vector[4] = np.log(parameter_values['sigma_w_'+self.instrument]['cvalue']*self.sigma_factor)
        elif self.kernel_name == 'CeleriteExpKernel':
            self.parameter_vector[0] = np.log(parameter_values['GP_sigma_'+self.input_instrument[0]]['cvalue'])
            self.parameter_vector[1] = np.log(parameter_values['GP_timescale_'+self.input_instrument[1]]['cvalue'])
            if not self.global_GP:
                self.parameter_vector[2] = np.log(parameter_values['sigma_w_'+self.instrument]['cvalue']*self.sigma_factor)
        elif self.kernel_name == 'CeleriteMaternKernel':
            self.parameter_vector[0] = np.log(parameter_values['GP_sigma_'+self.input_instrument[0]]['cvalue'])
            self.parameter_vector[1] = np.log(parameter_values['GP_rho_'+self.input_instrument[1]]['cvalue'])
            if not self.global_GP:
                self.parameter_vector[2] = np.log(parameter_values['sigma_w_'+self.instrument]['cvalue']*self.sigma_factor)
        elif self.kernel_name == 'CeleriteMaternExpKernel':
            self.parameter_vector[0] = np.log(parameter_values['GP_sigma_'+self.input_instrument[0]]['cvalue'])
            self.parameter_vector[1] = np.log(parameter_values['GP_timescale_'+self.input_instrument[1]]['cvalue'])
            self.parameter_vector[2] = np.log(parameter_values['GP_rho_'+self.input_instrument[2]]['cvalue'])
            if not self.global_GP:
                self.parameter_vector[3] = np.log(parameter_values['sigma_w_'+self.instrument]['cvalue']*self.sigma_factor)
        elif self.kernel_name == 'CeleriteSHOKernel':
            self.parameter_vector[0] = np.log(parameter_values['GP_S0_'+self.input_instrument[0]]['cvalue'])
            self.parameter_vector[1] = np.log(parameter_values['GP_Q_'+self.input_instrument[1]]['cvalue'])
            self.parameter_vector[2] = np.log(parameter_values['GP_omega0_'+self.input_instrument[2]]['cvalue'])
            if not self.global_GP:
                self.parameter_vector[3] = np.log(parameter_values['sigma_w_'+self.instrument]['cvalue']*self.sigma_factor)
    def __init__(self,kernel, X, model_type, yerr = None, instrument = None, george_hodlr = False):
        # Save input data on the object:
        self.kernel_name = kernel_name
        # Save input predictor:
        self.X = X
        # Save shape of input predictor:
        self.nX = X.shape[1]
        # Perform changes that define the model_type. For example, the juliet input sigmas (both jitters and GP amplitudes) are 
        # given in ppm in the input files, whereas for RVs they have the same units as the input RVs. This conversion factor is 
        # defined by the model_type:
        if model_type.lower() == 'lc':
            if instrument is None:
                instrument = 'lc'
            self.sigma_factor = 1e-6
        elif model_type.lower() == 'rv':
            if instrument is None:
                instrument = 'rv'
            self.sigma_factor = 1.
        else:
            raise Exception('Model type '+model_type+' currently not supported. Only "lc" or "rv" can serve as inputs for now.')
        # Save yerr:
        self.yerr = yerr
        # Variable for the GP object:
        self.GP = None
        # Are we using celerite?
        self.use_celerite = False
        # Are we using george_hodlr?
        if george_hodlr:
            self.george_hodlr = True
        else:
            self.george_hodlr = False
        # Name of input instrument if given:
        self.instrument = instrument
        # This sets the "instrument" name for each variable (self.variables below). If, for example, GP_Prot_TESS_K2_RV and 
        # GP_Gamma_TESS, and self.variables = [Prot,Gamma], then self.instrument_variables = ['TESS_K2_RV','TESS'].
        self.input_instrument = []

        # Initialize each kernel on the GP object:
        if self.kernel_name == 'SEKernel':
            # Define names of the variables:
            self.variables = ['sigma']
            for i in range(self.nX):
                self.variables = self.variables + ['alpha'+str(i)]
            # Generate GPExpSquared base kernel:
            self.kernel = 1.*george.kernels.ExpSquaredKernel(np.ones(self.nX),ndim = self.nX, axes = range(self.nX))
            # (Note no jitter kernel is given, as with george one defines this in the george.GP call):
        elif self.kernel_name == 'ExpSineSquaredSEKernel':
            # Define names of the variables:
            self.variables = ['sigma','alpha','Gamma','Prot']
            # Generate the kernels:
            K1 = 1.*george.kernels.ExpSquaredKernel(metric = 1.0
            K2 = george.kernels.ExpSine2Kernel(gamma=1.0,log_period=1.0)
            self.kernel = K1*K2
            # (Note no jitter kernel is given, as with george one defines this in the george.GP call):
        elif self.kernel_name == 'CeleriteQPKernel':
            # Define names of the variables:
            self.variables = ['B','L','Prot','C']
            # Generate rotational kernel:
            rot_kernel = terms.TermSum(RotationTerm(log_amp=np.log(10.),\
                                                    log_timescale=np.log(10.0),\
                                                    log_period=np.log(3.0),\
                                                    log_factor=np.log(1.0)))
            # Jitter term:
            kernel_jitter = terms.JitterTerm(np.log(100*1e-6))
            # Wrap GP kernel and object:
            self.kernel = rot_kernel + kernel_jitter
            # We are using celerite:
            self.use_celerite = True
        elif self.kernel_name == 'CeleriteExpKernel':
            # Define names of the variables:
            self.variables = ['sigma','timescale']
            # Generate exponential kernel:
            exp_kernel = terms.RealTerm(log_a=np.log(10.), log_c=np.log(10.))
            # Jitter term:
            kernel_jitter = terms.JitterTerm(np.log(100*1e-6))
            # Wrap GP kernel and object:
            self.kernel = exp_kernel + kernel_jitter
            # We are using celerite:
            self.use_celerite = True
        elif self.kernel_name == 'CeleriteMaternKernel':
            # Define names of the variables:
            self.variables = ['sigma','rho']
            # Generate matern kernel:
            matern_kernel = terms.Matern32Term(log_sigma=np.log(10.), log_rho=np.log(10.))
            # Jitter term:
            kernel_jitter = terms.JitterTerm(np.log(100*1e-6))
            # Wrap GP kernel and object:
            self.kernel = matern_kernel + kernel_jitter
            # We are using celerite:
            self.use_celerite = True
        elif self.kernel_name == 'CeleriteMaternExpKernel':
            # Define names of the variables:
            self.variables = ['sigma','timescale','rho']
            # Generate matern and exponential kernels:
            matern_kernel = terms.Matern32Term(log_sigma=np.log(10.), log_rho=np.log(10.))
            exp_kernel = terms.RealTerm(log_a=np.log(10.), log_c=np.log(10.))
            # Jitter term:
            kernel_jitter = terms.JitterTerm(np.log(100*1e-6))
            # Wrap GP kernel and object:
            self.kernel = exp_kernel*matern_kernel + kernel_jitter
            # We are using celerite:
            self.use_celerite = True
        elif self.kernel_name == 'CeleriteSHOKernel':
            # Define names of the variables:
            self.variables = ['S0','Q','omega0']
            # Generate kernel:
            sho_kernel = terms.SHOTerm(log_S0=np.log(10.), log_Q=np.log(10.),log_omega0=np.log(10.))
            # Jitter term:
            kernel_jitter = terms.JitterTerm(np.log(100*1e-6))
            # Wrap GP kernel and object:
            self.kernel = sho_kernel + kernel_jitter
            # We are using celerite:
            self.use_celerite = True
        self.init_GP()
        if self.instrument.lower() in ['rv','lc']:
            # If no instrument or instrument is 'rv' or 'lc', assume GP object will fit for a global GP 
            # (e.g., global photometric signal, or global RV signal) that assumes a given GP realization for all instruments:
            self.parameter_vector = np.zeros(len(self.variables))
            self.global_GP = True
        else:
            # If GP per instrument, then there is one jitter term per instrument:
            self.parameter_vector = np.zeros(len(self.variables)+1)
            self.globel_GP = False
