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
import sys

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

__all__ = ['fit'] 

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
        If a path to a lightcurve file is given, ``t_lc``, ``y_lc``, ``yerr_lc`` and ``instruments_lc`` will be read from there. The file format is a pure 
        ascii file where times are in the first column, relative fluxes in the second, errors in the third and instrument names in the fourth.

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

    def __init__(self,priors,t_lc = None, y_lc = None, yerr_lc = None, instruments_lc = None,\
                 t_rv = None, y_rv = None, yerr_rv = None, instruments_rv = None,\
                 GP_regressors_lc = None, GP_instruments_lc = None, linear_regressors_lc = None, \
                 linear_instruments_lc = None, GP_regressors_rv = None, GP_instruments_rv = None,\
                 george_hodlr = False, use_dynesty = False, dynamic = False, \
                 dynesty_bound = 'multi',dynesty_sample='rwalk',dynesty_nthreads = None,\
                 out_folder = None, lcfilename = None, rvfilename = None, GPlceparamfile = None,\
                 GPrveparamfile = None, lctimedef = 'TDB', rvtimedef = 'UTC',\
                 ld_laws = 'quadratic', priorfile = None,\
                 pl = 0., pu = 1., n_live_points = 1000, ecclim = 1., n_supersamp = None, exptime_supersamp = None, \
                 instrument_supersamp = None):

        # Define cases in which data is given through files: 
        if ((t_lc is None) or (y_lc is None) or (yerr_lc is None)) and ((t_rv is None) or \
             (y_rv is None) or (yerr_rv is None)):
            if lcfilename is not None:
                t_lc,y_lc,yerr_lc,instruments_lc,instrument_indexes_lc,ninstruments_lc,inames_lc,lm_lc_boolean,lm_lc_arguments = \
                utils.readlc(lcfilename)
            if rvfilename is not None:
                t_rv,y_rv,yerr_rv,instruments_rv,instrument_indexes_rv,ninstruments_rv,inames_rv,lm_rv_boolean,lm_rv_arguments = \
                utils.readlc(rvfilename)
            if (lcfilename is None) and (rvfilename is None): 
                raise Exception('No complete dataset (photometric or radial-velocity) given.\n'+\
                      ' Make sure to feed times (t_lc and/or t_rv) values (y_lc and/or y_rv) \n'+\
                      ' and errors (yerr_lc and/or yerr_rv).')

        # If data given through direct arrays, generate instrument_indexes_lc (dictionary that holds indexes of arrays that correspond 
        # to a given instrument),ninstruments_lc (numver of instruments),inames_lc (names of instruments):
        if (lcfilename is None) and (t_lc is not None):
            inames_lc = []
            for i in range(len(t_lc)):
                if instruments_lc[i] not in inames_lc:
                    inames_lc.append(instruments_lc[i])
            ninstruments_lc = len(inames_lc)
            instrument_indexes_lc = {}
            for instrument in inames_lc:
                instrument_indexes_lc[instrument] = np.where(instruments_lc == instrument)[0]
            
        # Same for radial-velocity data:
        if (rvfilename is None) and (t_rv is not None):
            inames_rv = []
            for i in range(len(t_rv)):
                if instruments_rv[i] not in inames_rv:
                    inames_rv.append(instruments_rv[i])
            ninstruments_rv = len(inames_rv)
            instrument_indexes_rv = {}
            for instrument in inames_rv:
                instrument_indexes_rv[instrument] = np.where(instruments_rv == instrument)[0]

 
        # Also generate lm_lc_boolean and lm_lc_arguments in case linear regressors were passed:
        if (lcfilename is None) and (t_lc is not None) and (linear_regressors_lc is not None):
            lm_lc_boolean = {}
            lm_lc_arguments = {}
            for instrument in inames_lc:
                if instrument in linear_instruments_lc:
                    lm_lc_boolean[instrument] = True
                    lm_lc_arguments[instrument] = linear_regressors_lc[instrument_indexes_rv[instrument],:]
                else:
                    lm_lc_boolean[instrument] = False
                     

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
            if not os.path.exists(out_folder):
                os.mkdir(out_folder)
            if (not os.path.exists(out_folder+'lc.dat')): 
                if lcfilename is not None:
                    os.system('cp '+lcfilename+' '+out_folder+'lc.dat')
                elif t_lc is not None:
                    fout = open(out_folder+'lc.dat','w')
                    lm_lc_counters = {}
                    for i in range(len(t_lc)):
                        fout.write('{0:.10f} {1:.10f} {2:.10f} {3:}'.format(t_lc[i],y_lc[i],yerr_lc[i],instruments_lc[i]))
                        if lm_lc_boolean[instruments_lc[i]]:
                            if instruments_lc[i] not in lm_lc_counters.keys():
                                lm_lc_counters[instruments_lc[i]] = 0
                            for j in range(lm_lc_arguments[instruments_lc[i]].shape[1]):
                                fout.write(' {0:.10f}'.format(lm_lc_arguments[instruments_lc[i]][lm_lc_counters[instruments_lc[i]]][j]))
                            lm_lc_counters[instruments_lc[i]] += 1
                        fout.write('\n')
                    fout.close()
            if (not os.path.exists(out_folder+'rvs.dat')):
                if rvfilename is not None:
                    os.system('cp '+rvfilename+' '+out_folder+'rvs.dat')
                elif t_rv is not None:
                    fout = open(out_folder+'rvs.dat','w')
                    for i in range(len(t_rv)):
                        fout.write('{0:.10f} {1:.10f} {2:.10f} {3:}\n'.format(t_rv[i],y_rv[i],yerr_rv[i],instruments_rv[i]))
                    fout.close()
            if (not os.path.exists(out_folder+'lc_eparams.dat')):
                if GPlceparamfile is not None:
                    os.system('cp '+rvfilename+' '+out_folder+'lc_eparams.dat')
                elif GP_regressors_lc is not None:
                    fout = open(out_folder+'lc_eparams.dat.dat','w')
                    for i in range(GP_regressors_lc.shape[0]):
                        for j in range(GP_regressors_lc.shape[1]):
                            fout.write('{0:.10f} '.format(GP_regressors_lc[i,j]))
                        if GP_instruments_lc is not None:
                            fout.write('{0:}\n'.format(GP_instruments_lc[i]))
                        else:
                            fout.write('\n')
                    fout.close()
            if (not os.path.exists(out_folder+'rv_eparams.dat')):
                if GPrveparamfile is not None:
                    os.system('cp '+rvfilename+' '+out_folder+'rv_eparams.dat')
                elif GP_regressors_rv is not None:
                    fout = open(out_folder+'rv_eparams.dat.dat','w')
                    for i in range(GP_regressors_rv.shape[0]):
                        for j in range(GP_regressors_rv.shape[1]):
                            fout.write('{0:.10f} '.format(GP_regressors_rv[i,j]))
                        if GP_instruments_rv is not None:
                            fout.write('{0:}\n'.format(GP_instruments_rv[i]))
                        else:
                            fout.write('\n')
                    fout.close()    
        
        #val1,val2 = reverse_ld_coeffs('quadratic',0.1,0.5)     
        #print("yup:",val1,val2)   
