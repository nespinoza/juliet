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
    Given a dictionary with priors and a dataset, this class performs a juliet fit. Example usage:

               >>> out = juliet.fit(priors,t_lc=times,y_lc=fluxes,yerr_lc=fluxes)

    :param priors:                   A python dictionary containing each of the parameters to be fit, along with their 
                                     respective prior distributions and hyperparameters. Each key of the priors 
                                     dictionary has to have a parameter name (e.g., 'r1_p1', 'sigma_w_TESS'), and each of 
                                     those elements are, in turn, dictionaries as well containing two keys: a 'distribution' 
                                     key which defines the prior distribution of the parameter and a 'hyperparameters' key, 
                                     which contains the hyperparameters of that distribution. 

                                     e.g.: >> priors['r1_p1'] = {}
                                     >> priors['r1_p1']['distribution'] = 'Uniform'
                                     >> priors['r1_p1']['hyperparameters'] = [0.,1.]
    :type priors:                    dict

    :param t_lc:                     Array of times of the lightcurve data.
    :type t_lc:                      ndarray

    :param y_lc:                     Array of relative fluxes of the lightcurve data.     
    :type y_lc:                      ndarray

    :param yerr_lc:                  Array of errors on relative fluxes of the lightcurve data.
    :type yerr_lc:                   ndarray

    :param instruments_lc:           (Optional) Array containing the names (strings) of the instrument 
                                     corresponding to each of the t_lc, y_lc and yerr_lc datapoints. If not 
                                     given, this is set to an array of length len(t_lc), where each element is 
                                     named 'Data'.
    :type instruments_lc:            list
   
    :param GP_regressors_lc:         (Optional) Matrix containing in each column the GP regressors to use for the photometric measurements.
    :type GP_regressors_lc:          ndarray

    :param GP_instruments_lc:        (Optional) Array of length GP_regressors_lc.shape[0] indicating the instruments
                                     to which to apply the GP regression. If not given, assumed the GP will apply to all the 
                                     photometry.
    :type GP_instruments_lc:         list

    :param linear_regressors_lc:     (Optional) Matrix containing in each column linear regressors to use in the fit.
    :type linear_regressors_lc:      ndarray

    :param linear_instruments_lc:    (Optional) Array of length linear_regressors_lc.shape[0] indicating the instruments
                                     to which to apply the linear regression.

    :type linear_instruments_lc:     list

    :param GP_regressors_rv:         (Optional) Matrix containing in each column the GP regressors to use for the radial-velocities.
    :type GP_regressors_rv:          ndarray

    :param GP_instruments_rv:        (Optional) Array of length GP_regressors_lc.shape[0] indicating the instruments
                                     to which to apply the GP regression. If not given, assumed the GP will apply to all the 
                                     photometry.
    :type GP_instruments_rv:         list

    :param t_rv:                     Same as t_lc, but for the radial-velocities.
    :type t_rv:                      ndarray
   
    :param y_rv:                     Same as y_lc, but for the radial-velocities.
    :type y_rv:                      ndarray

    :param yerr_rv:                  Same as yerr_lc, but for the radial-velocities.
    :type yerr_rv:                   ndarray

    :param george_hodlr:             If True, use the HODLR solver for george Gaussian Process evaluation.
    :type george_hodlr:              bool
   
    :param use_dynesty:              If True, use dynesty instead of MultiNest for posterior sampling and evidence evaluation.
    :type use_dynesty:               bool
 
    :param dynamic:                  If True, use dynamic Nested Sampling with dynesty.
    :type dynamic:                   bool
 
    :param dynesty_bound:            Define the dynesty bound to use. Default is 'multi'.
    :type dynesty_bound:             str

    :param dynesty_sample:           Define the sampling method for dynesty. Default is 'rwalk'.
    :type dynesty_sample:            str

    :param dynesty_nthreads:         Define the number of threads to use within dynesty. Default is to use just 1.
    :type dynesty_nthreads:          int

    :param out_folder:               If a path is given, results will be saved to that path.
    :type out_folder:                str

    :param lcfilename:               If a path to a lightcurve file is given, t_lc, y_lc, yerr_lc and instruments_lc will 
                                     be read from there.
    :type lcfilename:                str

    :param rvfilename:               Same as lcfilename, but for the radial-velocities.
    :type rvfilename:                str

    :param GPlceparamfile:           If a path to a file is given, the columns of that file will be used as GP regressors for 
                                     the lightcurve fit.
    :type GPlceparamfile:            str

    :param GPrveparamfile:           Same as GPlceparamfile but for the radial-velocities.
    :type GPrveparamfile:            str

    :param lctimedef:                Time definitions for each of the lightcurve instruments. Default is TDB.
    :type lctimedef:                 list

    :param rvtimedef:                Time definitions for each of the radial-velocity instruments. Default is UTC.
    :type rvtimedef:                 list

    :param ld_laws:                  Limb-darkening law to be used for each instrument. Default is to use quadratic for all.
    :type ld_laws:                   list

    :param priorfile:                If a path to a file is given, it will be assumed this is a prior file and the prior dictionary 
                                     will be overwritten by the data in this file.
    :type priorfile:                 str

    :param pl:                       If the (r1,r2) parametrization for (b,p) is used, this defines the lower limit of the planet-to-star 
                                     radius ratio to be sampled.
    :type pl:                        float
   
    :param pu:                       Same as pl, but for the upper limit.
    :type pu:                        float

    :param n_live_points:            Number of live-points to be sampled.
    :type n_live_points:             int

    :param ecclim:                   Upper limit on the maximum eccentricity to sample.
    :type ecclim:                    float

    :param sdensity_mean:            Value of mean stellar density (in SI units), in case one wants to use it as a datapoint.
    :type sdensity_mean:             float

    :param sdensity_sigma:           Value of the standard deviation of the stellar density (in SI units), in case one wants to use it as 
                                     a datapoint.
    :type sdensity_sigma:            float

    :param n_supersamp:              Define the number of datapoints to supersample
    :type n_supersamp:               list

    :param exptime_supersamp:        Define the exposure-time of the observations for the supersampling.
    :type exptime_supersamp:         ndarray

    :param instrument_supersamp:     Define for which lightcurve instruments there will be super-sampling.
    :type instrument_supersamp:      list

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
                 pl = 0., pu = 1., n_live_points = 1000, ecclim = 1., sdensity_mean = None, \
                 sdensity_sigma = None, n_supersamp = None, exptime_supersamp = None, \
                 instrument_supersamp = None):
        if ((t_lc is  None) or (y_lc is None) or (yerr_lc is None)) and ((t_rv is None) or \
             (y_rv is None) or (yerr_rv is None)):
            if lcfilename is not None:
                t_lc,y_lc,yerr_lc,instruments_lc,instrument_indexes_lc,ninstruments_lc,inames_lc,lm_boolean,lm_arguments = \
                utils.readlc(lcfilename)
            if rvfilename is not None:
                t_rv,y_rv,yerr_rv,instruments_rv,instrument_indexes_rv,ninstruments_rv,inames_rv,lm_rv_boolean,lm_rv_arguments = \
                utils.readlc(rvfilename)
            if (lcfilename is None) and (rvfilename is None): 
                raise Exception('No complete dataset (photometric or radial-velocity) given.\n'+\
                      ' Make sure to feed times (t_lc and/or t_rv) values (y_lc and/or y_rv) \n'+\
                      ' and errors (yerr_lc and/or yerr_rv).')
        self.results = {}
        #val1,val2 = reverse_ld_coeffs('quadratic',0.1,0.5)     
        #print("yup:",val1,val2)   
