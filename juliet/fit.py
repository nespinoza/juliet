# Import batman, for lightcurve models:
import batman
# Try to import catwoman:
try: 
    import catwoman
    have_catwoman = True
except:
    have_catwoman = False
# Import radvel, for RV models:
import radvel
# Import george for detrending:
try:
    import george
except:
    print('Warning: no george installation found. No non-celerite GPs will be able to be used')
# Import celerite for detrending:
try:
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
except:
    print('Warning: no celerite installation found. No celerite GPs will be able to be used')
# Import dynesty for dynamic nested sampling:
try:
    import dynesty
    from dynesty.utils import resample_equal
    force_pymultinest = False
except:
    force_pymultinest = True
# Import multinest for (importance) nested sampling:
try:
    import pymultinest
    force_dynesty = False
except:
    force_dynesty = True

# Import generic useful classes:
import os
import sys
import numpy as np

# Define constants on the code:
G = 6.67408e-11 # Gravitational constant, mks
log2pi = np.log(2.*np.pi) # ln(2*pi)

# Import all the utils functions:
from .utils import *

__all__ = ['load','fit','gaussian_process','model'] 

class load(object):
    """
    Given a dictionary with priors (or a filename pointing to a prior file) and data either given through arrays 
    or through files containing the data, this class loads data into a juliet object which holds all the information 
    about the dataset. Example usage:

               >>> data = juliet.load(priors=priors,t_lc=times,y_lc=fluxes,yerr_lc=fluxes_errors)

    Or, also,
             
               >>> data = juliet.load(input_folder = folder)

    :param priors: (optional, dict or string)                         
        This can be either a python ``string`` or a python ``dict``. If a ``dict``, this has to contain each of 
        the parameters to be fit, along with their respective prior distributions and hyperparameters. Each key 
        of this dictionary has to have a parameter name (e.g., ``r1_p1``, ``sigma_w_TESS``), and each of 
        those elements are, in turn, dictionaries as well containing two keys: a ``distribution``
        key which defines the prior distribution of the parameter and a ``hyperparameters`` key, 
        which contains the hyperparameters of that distribution. 

        Example setup of the ``priors`` dictionary:
            >> priors = {}
            >> priors['r1_p1'] = {}
            >> priors['r1_p1']['distribution'] = 'Uniform'
            >> priors['r1_p1']['hyperparameters'] = [0.,1.]

        If a ``string``, this has to contain the filename to a proper juliet prior file; the prior ``dict`` will 
        then be generated from there. A proper prior file has in the first column the name of the parameter, 
        in the second the name of the distribution, and in the third the hyperparameters of that distribution for 
        the parameter.
 
        Note that this along with either lightcurve or RV data or a ``input_folder`` has to be given in order to properly 
        load a juliet data object.

    :param input_folder: (optional, string)
        Python ``string`` containing the path to a folder containing all the input data --- this will thus be load into a 
        juliet data object. This input folder has to contain at least a ``priors.dat`` file with the priors and either a ``lc.dat`` 
        file containing lightcurve data or a ``rvs.dat`` file containing radial-velocity data. If in this folder a ``GP_lc_regressors.dat`` 
        file or a ``GP_rv_regressors.dat`` file is found, data will be loaded into the juliet object as well.

        Note that at least this or a ``priors`` string or dictionary, along with either lightcurve or RV data has to be given 
        in order to properly load a juliet data object.

    :param t_lc: (optional, dictionary)
        Dictionary whose keys are instrument names; each of those keys is expected to have arrays with the times corresponding to those instruments.
        For example,
                                    >>> t_lc = {}
                                    >>> t_lc['TESS'] = np.linspace(0,100,100)

        Is a valid input dictionary for ``t_lc``.

    :param y_lc: (optional, dictionary)
        Similarly to ``t_lc``, dictionary whose keys are instrument names; each of those keys is expected to have arrays with the fluxes corresponding to those instruments. 
        These are expected to be consistent with the ``t_lc`` dictionaries.

    :param yerr_lc: (optional, dictionary)
        Similarly to ``t_lc``, dictionary whose keys are instrument names; each of those keys is expected to have arrays with the errors on the fluxes corresponding to those instruments. 
        These are expected to be consistent with the ``t_lc`` dictionaries. 

    :param GP_regressors_lc: (optional, dictionary) 
        Dictionary whose keys are names of instruments where a GP is to be fit. On each name/element, an array of 
        regressors of shape ``(m,n)`` containing in each column the ``n`` GP regressors to be used for 
        ``m`` photometric measurements has to be given. Note that ``m`` for a given instrument has to be of the same length 
        as the corresponding ``t_lc`` for that instrument. Also, note the order of each regressor of each instrument has to match 
        the corresponding order in the ``t_lc`` array. 
        For example,

                                    >>> GP_regressors_lc = {}
                                    >>> GP_regressors_lc['TESS'] = np.linspace(-1,1,100)
 
    :param linear_regressors_lc: (optional, dictionary)
        Similarly as for ``GP_regressors_lc``, this is a dictionary whose keys are names of instruments where a linear regression is to be fit. 
        On each name/element, an array of shape ``(q,p)`` containing in each column the ``p`` linear regressors to be used for the ``q`` 
        photometric measurements. Again, note the order of each regressor of each instrument has to match the corresponding order in the ``t_lc`` array. 
         
    :param GP_regressors_rv: (optional, dictionary)  
        Same as ``GP_regressors_lc`` but for the radial-velocity data. 

    :param linear_regressors_rv: (optional, dictionary)
        Same as ``linear_regressors_lc``, but for the radial-velocities.

    :param t_rv: (optional, dictionary)                    
        Same as ``t_lc``, but for the radial-velocities.
   
    :param y_rv: (optional, dictionary)
        Same as ``y_lc``, but for the radial-velocities.

    :param yerr_rv: (optional, dictionary)
        Same as ``yerr_lc``, but for the radial-velocities.

    :param out_folder: (optional, string) 
        If a path is given, results will be saved to that path as a ``pickle`` file, along with all inputs in the standard juliet format.

    :param lcfilename:  (optional, string)             
        If a path to a lightcurve file is given, ``t_lc``, ``y_lc``, ``yerr_lc`` and ``instruments_lc`` will be read from there. The basic file format is a pure 
        ascii file where times are in the first column, relative fluxes in the second, errors in the third and instrument names in the fourth. If more columns are given for 
        a given instrument, those will be identified as linear regressors for those instruments.

    :param rvfilename: (optional, string)               
        Same as ``lcfilename``, but for the radial-velocities.

    :param GPlceparamfile: (optional, string)          
        If a path to a file is given, the columns of that file will be used as GP regressors for the lightcurve fit. The file format is a pure ascii file 
        where regressors are given in different columns, and the last column holds the instrument name. The order of this file has to be consistent with 
        ``t_lc`` and/or the ``lcfilename`` file. If no instrument for each regressor is given, it is assumed the GP is a global GP, common to all 
        instruments.

    :param GPrveparamfile: (optional, string)          
        Same as ``GPlceparamfile`` but for the radial-velocities.

    :param lctimedef: (optional, string)               
        Time definitions for each of the lightcurve instruments. Default is to assume all instruments (in lcs and rvs) have the same time definitions. If more than one instrument is given, this string 
        should have instruments and time-definitions separated by commas, e.g., ``TESS-TDB, LCOGT-UTC``, etc.

    :param rvtimedef: (optional, string)               
        Time definitions for each of the radial-velocity instruments. Default is to assume all instruments (in lcs and rvs) have the same time definitions. If more than one instrument is given, 
        this string should have instruments and time-definitions separated by commas, e.g., ``FEROS-TDB, HARPS-UTC``, etc.

    :param ld_laws: (optional, string)                 
        Limb-darkening law to be used for each instrument. Default is ``quadratic`` for all instruments. If more than one instrument is given, 
        this string should have instruments and limb-darkening laws separated by commas, e.g., ``TESS-quadratic, LCOGT-linear``.

    :param priorfile: (optional, string)                
        If a path to a file is given, it will be assumed this is a prior file. The ``priors`` dictionary will be overwritten by the data in this 
        file. The file structure is a plain ascii file, with the name of the parameters in the first column, name of the prior distribution in the 
        second column and hyperparameters in the third column.

    :param lc_instrument_supersamp: (optional, array of strings)     
        Define for which lightcurve instruments super-sampling will be applied (e.g., in the case of long-cadence integrations). e.g., ``instrument_supersamp = ['TESS','K2']``

    :param n_supersamp: (optional, array of ints)              
        Define the number of datapoints to supersample. Order should be consistent with order in ``instrument_supersamp``. e.g., ``n_supersamp = [20,30]``.

    :param exptime_supersamp: (optional, array of floats)        
        Define the exposure-time of the observations for the supersampling. Order should be consistent with order in ``instrument_supersamp``. e.g., ``exptime_supersamp = [0.020434,0.020434]``

    :param verbose: (optional, boolean)
        If True, all outputs of the code are printed to terminal. Default is False.

    """

    def data_preparation(self,times,instruments,linear_regressors,linear_instruments):
        """
        This function generates f useful internal arrays for this class: inames which saves the instrument names, ``global_times`` 
        which is a "flattened" array of the ``times`` dictionary where all the times for all instruments are stacked, instrument_indexes, 
        which is a dictionary that has, for each instrument the indexes of the ``global_times`` corresponding to each instrument, lm_boolean which saves booleans for each 
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

    def convert_input_data(self, t, y, yerr):
        """
        This converts the input dictionaries to arrays (this is easier to handle internally within juliet; input dictionaries are just asked because 
        it is easier for the user to pass them). 
        """
        instruments = list(t.keys())
        all_times = np.array([])
        all_y = np.array([])
        all_yerr = np.array([])
        all_instruments = np.array([])
        for instrument in instruments:
            for i in range(len(t[instrument])):
                all_times = np.append(all_times,t[instrument][i])
                all_y = np.append(all_y,y[instrument][i])
                all_yerr = np.append(all_yerr,yerr[instrument][i])
                all_instruments = np.append(all_instruments,instrument)
        return all_times, all_y, all_yerr, all_instruments

    def convert_to_dictionary(self, t, y, yerr, instrument_indexes):
        """
        Convert data given in arrays to dictionaries for easier user usage
        """
        times = {}
        data = {}
        errors = {}
        for instrument in instrument_indexes.keys():
            times[instrument] = t[instrument_indexes[instrument]]
            data[instrument] = y[instrument_indexes[instrument]]
            errors[instrument] = yerr[instrument_indexes[instrument]]
        return times,data,errors

    def save_regressors(self,fname, GP_arguments, global_model):
        """
        This function saves the GP regressors to fname.
        """
        fout = open(fname,'w')
        for GP_instrument in GP_arguments.keys():
            GP_regressors = GP_arguments[GP_instrument]
            multi_dimensional = False
            if len(GP_regressors.shape) == 2:
                multi_dimensional = True
            if multi_dimensional:
                 for i in range(GP_regressors.shape[0]):
                     for j in range(GP_regressors.shape[1]):
                         fout.write('{0:.10f} '.format(GP_regressors[i,j]))
                     fout.write('{0:}\n'.format(GP_instrument))
            else:
                 for i in range(GP_regressors.shape[0]):
                     fout.write('{0:.10f} {1:}\n'.format(GP_regressors[i],GP_instrument))
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

    def save_priorfile(self,fname):
        """
        This function saves a priorfile file out to fname
        """
        fout = open(fname,'w')
        for pname in self.priors.keys():
            if self.priors[pname]['distribution'].lower() != 'fixed':
                value = ','.join(np.array(self.priors[pname]['hyperparameters']).astype(str))
            else:
                value = str(self.priors[pname]['hyperparameters'])
            fout.write('{0: <20} {1: <20} {2: <20}\n'.format(pname,self.priors[pname]['distribution'],value))
        fout.close()

    def check_global(self,name):
        for pname in self.priors.keys():
            if name in pname.split('_'):
                return True
        return False

    def append_GP(self, ndata, instrument_indexes, GP_arguments,inames):
        """
            This function appends all the GP regressors into one --- useful for the global models.
        """
        # First check if GP regressors are multi-dimensional --- check this just for the first instrument:
        if len(GP_arguments[inames[0]].shape) == 2:
            nregressors = GP_arguments[inames[0]].shape[1]
            multidimensional = True
            out = np.zeros([ndata, nregressors])
        else:
            multidimensional = False
            out = np.zeros(ndata)
        for instrument in inames:
            if multi_dimensional:
                out[instrument_indexes[instrument],:] = GP_arguments[instrument]
            else:
                out[instrument_indexes[instrument]] = GP_arguments[instrument]
        return out 

    def sort_GP(self, dictype):
        if dictype == 'lc':
            # Sort first times, fluxes, errors and the GP regressor:
            idx_sort = np.argsort(self.GP_lc_arguments['lc'])
            self.t_lc = self.t_lc[idx_sort]
            self.y_lc = self.y_lc[idx_sort]
            self.yerr_lc = self.yerr_lc[idx_sort]      
            self.GP_lc_arguments['lc'] = self.GP_lc_arguments['lc'][idx_sort]
            # Now with the sorted indices, iterate through the instrument indexes and change them according to the new 
            # ordering:
            for instrument in self.inames_lc:
                new_instrument_indexes = np.zeros(len(instrument_indexes))
                instrument_indexes = self.instrument_indexes_lc[instrument]
                counter = 0
                for i in instrument_indexes:
                    new_instrument_indexes[counter] = np.where(i == idx_sort)[0][0]
                    counter += 1
                self.instrument_indexes_lc[instrument] = new_instrument_indexes
        elif dictype == 'rv':
            # Sort first times, rvs, errors and the GP regressor:
            idx_sort = np.argsort(self.GP_rv_arguments['rv'])
            self.t_rv = self.t_rv[idx_sort]
            self.y_rv = self.y_rv[idx_sort]
            self.yerr_rv = self.yerr_rv[idx_sort]           
            self.GP_rv_arguments['rv'] = self.GP_rv_arguments['rv'][idx_sort]
            # Now with the sorted indices, iterate through the instrument indexes and change them according to the new 
            # ordering:
            for instrument in self.inames_rv:
                new_instrument_indexes = np.zeros(len(instrument_indexes))
                instrument_indexes = self.instrument_indexes_rv[instrument]
                counter = 0
                for i in instrument_indexes:
                    new_instrument_indexes[counter] = np.where(i == idx_sort)[0][0]
                    counter += 1 
                self.instrument_indexes_rv[instrument] = new_instrument_indexes
                    

    def generate_datadict(self, dictype):
        """
        This generates the options dictionary for lightcurves, RVs, and everything else you want to fit. Useful for the 
        fit, as it separaters options per instrument.

        :param dictype: (string)
            Defines the type of dictionary type. It can either be 'lc' (for the lightcurve dictionary) or 'rv' (for the 
            radial-velocity one). 
        """

        dictionary = {}
        if dictype == 'lc':
            inames = self.inames_lc
            ninstruments = self.ninstruments_lc
            instrument_supersamp = self.lc_instrument_supersamp
            n_supersamp = self.lc_n_supersamp
            exptime_supersamp = self.lc_exptime_supersamp
            numbering_planets = self.numbering_transiting_planets
            # Check if model is global based on the input prior names. If they include as instrument "rv", set to global model:
            self.global_lc_model = self.check_global('lc')
            global_model = self.global_lc_model
            if global_model and (self.GP_lc_arguments is not None):
                self.GP_lc_arguments['lc'] = append_GP(len(self.t_lc), self.instrument_indexes_lc, self.GP_lc_arguments, inames)
            GP_regressors = self.GP_lc_arguments
            
        elif dictype == 'rv':
            inames = self.inames_rv
            ninstruments = self.ninstruments_rv
            instrument_supersamp = None
            n_supersamp = None
            exptime_supersamp = None
            numbering_planets = self.numbering_rv_planets
            # Check if model is global based on the input prior names. If they include as instrument "lc", set to global model:
            self.global_lc_model = self.check_global('rv')
            global_model = self.global_lc_model
            # If global_model is True, create an additional key in the GP_regressors array that will have all the GP regressors appended:
            if global_model and (self.GP_rv_arguments is not None):
                self.GP_rv_arguments['rv'] = append_GP(len(self.t_rv), self.instrument_indexes_rv, self.GP_rv_arguments, inames)
            GP_regressors = self.GP_rv_arguments
        else:
            raise Exception('INPUT ERROR: dictype not understood. Has to be either lc or rv.')

        for i in range(ninstruments):
            instrument = inames[i]
            dictionary[instrument] = {}
            # Save if a given instrument will receive resampling (initialize this as False):
            dictionary[instrument]['resampling'] = False
            # Save if a given instrument has GP fitting ON (initialize this as False):
            dictionary[instrument]['GPDetrend'] = False
            # Save if transit fitting will be done for a given dataset/instrument (this is so users can fit photometry with, e.g., GPs):
            if dictype == 'lc':
                dictionary[instrument]['TransitFit'] = False
                dictionary[instrument]['TransitFitCatwoman'] = False

        if dictype == 'lc':
            # Extract limb-darkening law. If no limb-darkening law was given by the user, assume LD law depending on whether the user defined a prior for q1 only for a 
            # given instrument (in which that instrument is set to the linear law) or a prior for q1 and q2, in which case we assume the user 
            # wants to use a quadratic law for that instrument. If user gave one limb-darkening law, assume that law for all instruments that have priors for q1 and q2 
            # (if only q1 is given, assume linear for those instruments). If LD laws given for every instrument, extract them:
            all_ld_laws = self.ld_laws.split(',')
            if len(all_ld_laws) == 1:
                for i in range(ninstruments):
                    instrument = inames[i]
                    q1_given = False
                    q2_given = False
                    for parameter in self.priors.keys():
                        if parameter[0:2] == 'q1':
                            if instrument in parameter.split('_')[1:]:
                                q1_given = True
                        if parameter[0:2] == 'q2':
                            if instrument in parameter.split('_')[1:]:
                                q2_given = True
                    if q1_given and (not q2_given):
                        dictionary[instrument]['ldlaw'] = 'linear'
                    elif q1_given and q2_given:
                        dictionary[instrument]['ldlaw'] = (all_ld_laws[0].split('-')[-1]).split()[0].lower()
                    elif (not q1_given) and q2_given:
                        raise Exception('INPUT ERROR: it appears q1 for instrument '+instrument+' was not defined (but q2 was) in the prior file.')
            else:
                for ld_law in all_ld_laws:
                    instrument,ld = ld_law.split('-')
                    dictionary[instrument.split()[0]]['ldlaw'] = ld.split()[0].lower()

        # Extract supersampling parameters if given. 
        # For now this only allows inputs from lightcurves; TODO: add supersampling for RVs.
        if instrument_supersamp is not None and dictype == 'lc':
            for i in range(len(instrument_supersamp)):
                dictionary[instrument_supersamp[i]]['resampling'] = True
                dictionary[instrument_supersamp[i]]['nresampling'] = n_supersamp[i]
                dictionary[instrument_supersamp[i]]['exptimeresampling'] = exptime_supersamp[i]

        # Now, if generating lightcurve dict, check whether for some photometric instruments only photometry, and not a 
        # transit, will be fit. This is based on whether the user gave limb-darkening coefficients for a given photometric 
        # instrument or not. If given, transit is fit. If not, no transit is fit:
        if dictype == 'lc':
            for i in range(ninstruments):
                for pri in self.priors.keys():
                    if pri[0:2] == 'q1':
                        if inames[i] in pri.split('_'):
                            dictionary[inames[i]]['TransitFit'] = True
                            if self.verbose:
                                print('\t Transit fit detected for instrument ',inames[i])
                    if pri[0:3] == 'phi':
                        dictionary[inames[i]]['TransitFit'] = True
                        dictionary[inames[i]]['TransitFitCatwoman'] = True
                        if self.verbose:
                            print('\t Transit (catwoman) fit detected for instrument ',inames[i])

        # Now, implement noise models for each of the instrument. First check if model should be global or instrument-by-instrument, 
        # based on the input instruments given for the GP regressors.
        if global_model:
            dictionary['global_model'] = {}
            if GP_regressors is not None:
                dictionary['global_model']['GPDetrend'] = True
                dictionary['global_model']['noise_model'] = gaussian_process(self, model_type = dictype,instrument = dictype)
                if not dictionary['global_model']['noise_model'].isInit:
                    # If not initiated, most likely kernel is a celerite one. Reorder times, values, etc. This is OK --- is expected:
                    if dictype == 'lc':
                        self.sort_GP('lc')
                    elif dictype == 'rv':
                        self.sort_GP('rv')
                    # Try again:
                    dictionary['global_model']['noise_model'] = gaussian_process(self, model_type = dictype,instrument = dictype)
                    if not dictionary['global_model']['noise_model'].isInit:
                        # Check, blame the user:
                        raise Exception('INPUT ERROR: GP initialization for object for '+dictype+' global kernel failed.')
            else:
                dictionary['global_model']['GPDetrend'] = False
        else:
            for i in range(ninstruments):
                instrument = inames[i]    
                if (GP_regressors is not None) and (instrument in GP_regressors.keys()):
                    dictionary[instrument]['GPDetrend'] = True
                    dictionary[instrument]['noise_model'] =  gaussian_process(self, model_type = dictype, instrument = instrument)
                    if not dictionary[instrument]['noise_model'].isInit:
                        # Blame the user, although perhaps we could simply solve this as for the global modelling?:
                        raise Exception('INPUT ERROR: GP regressors for instrument '+instrument+' use celerite, and are not in ascending or descending order. Please, give the input in those orders --- it will not work othersie.')

        # Check which eccentricity parametrization is going to be used for each planet in the juliet numbering scheme.
        # 0 = ecc, omega  1: ecosomega,esinomega  2: sqrt(e)cosomega, sqrt(e)sinomega
        dictionary['ecc_parametrization'] = {}
        if dictype == 'lc':
            dictionary['efficient_bp'] = {}
        for i in numbering_planets:
            if 'ecosomega_p'+str(i) in self.priors.keys():
                dictionary['ecc_parametrization'][i] = 1
                if self.verbose:
                    print('\t >> ecosomega,esinomega parametrization detected for '+dictype+' planet p'+str(i))
            elif 'secosomega_p'+str(i) in self.priors.keys():
                dictionary['ecc_parametrization'][i] = 2
                if self.verbose:
                    print('\t >> sqrt(e)cosomega, sqrt(e)sinomega parametrization detected for '+dictype+' planet p'+str(i))
            else:
                dictionary['ecc_parametrization'][i] = 0
                if self.verbose:
                    print('\t >> ecc,omega parametrization detected for '+dictype+' planet p'+str(i))
            if dictype == 'lc':
                # Check if Espinoza (2018), (b,p) parametrization is on:
                if 'r1_p'+str(i) in self.priors.keys():
                    dictionary['efficient_bp'][i] = True
                    if self.verbose:
                        print('\t >> (b,p) parametrization detected for '+dictype+' planet p'+str(i))
                else:
                    dictionary['efficient_bp'][i] = False

        # Check if stellar density is in the prior:
        if dictype == 'lc':
            dictionary['fitrho'] = False
            if 'rho' in self.priors.keys():
                dictionary['fitrho'] = True
           

        # For RV dictionaries, check if RV trend will be fitted:
        if dictype == 'rv':
            dictionary['fitrvline'] = False
            dictionary['fitrvquad'] = False
            if 'rv_slope' in self.priors.keys():
                if 'rv_quad' in self.priors.keys():
                    dictionary['fitrvquad'] = True
                    if self.verbose:
                        print('\t Fitting quadratic trend to RVs.')
                else:
                    dictionary['fitrvline'] = True
                    if self.verbose:
                        print('\t Fitting linear trend to RVs.')

        # Save dictionary to self:
        if dictype == 'lc':
            self.lc_options = dictionary
        elif dictype == 'rv':
            self.rv_options = dictionary
        else:
            raise Exception('INPUT ERROR: dictype not understood. Has to be either lc or rv.')

    def set_lc_data(self,t_lc, y_lc, yerr_lc, instruments_lc,instrument_indexes_lc,ninstruments_lc,inames_lc,lm_lc_boolean,lm_lc_arguments):
            self.t_lc = t_lc.astype('float64')
            self.y_lc = y_lc
            self.yerr_lc = yerr_lc
            self.inames_lc = inames_lc
            self.instruments_lc = instruments_lc
            self.ninstruments_lc = ninstruments_lc
            self.instrument_indexes_lc = instrument_indexes_lc
            self.lm_lc_boolean = lm_lc_boolean
            self.lm_lc_arguments = lm_lc_arguments
            self.lc_data = True

    def set_rv_data(self,t_rv, y_rv, yerr_rv, instruments_rv,instrument_indexes_rv,ninstruments_rv,inames_rv,lm_rv_boolean,lm_rv_arguments):
            self.t_rv = t_rv.astype('float64')
            self.y_rv = y_rv
            self.yerr_rv = yerr_rv
            self.inames_rv = inames_rv
            self.instruments_rv = instruments_rv
            self.ninstruments_rv = ninstruments_rv
            self.instrument_indexes_rv = instrument_indexes_rv
            self.lm_rv_boolean = lm_rv_boolean
            self.lm_rv_arguments = lm_rv_arguments
            self.rv_data = True

    def save(self):
            if self.out_folder[-1] != '/': 
                self.out_folder = self.out_folder + '/'
            # First, save lightcurve data:
            if not os.path.exists(self.out_folder):
                os.mkdir(self.out_folder)
            if (not os.path.exists(self.out_folder+'lc.dat')):
                if self.lcfilename is not None:
                    os.system('cp '+self.lcfilename+' '+self.out_folder+'lc.dat')
                elif self.t_lc is not None:
                    self.save_data(self.out_folder+'lc.dat',self.t_lc,self.y_lc,self.yerr_lc,self.instruments_lc,self.lm_lc_boolean,self.lm_lc_arguments)
            # Now radial-velocity data:
            if (not os.path.exists(self.out_folder+'rvs.dat')):
                if self.rvfilename is not None:
                    os.system('cp '+self.rvfilename+' '+self.out_folder+'rvs.dat')
                elif self.t_rv is not None:
                    self.save_data(self.out_folder+'rvs.dat',self.t_rv,self.y_rv,self.yerr_rv,self.instruments_rv,self.lm_rv_boolean,self.lm_rv_arguments)
            # Next, save GP regressors:
            if (not os.path.exists(self.out_folder+'GP_lc_regressors.dat')):
                if self.GPlceparamfile is not None:
                    os.system('cp '+GPlceparamfile+' '+self.out_folder+'GP_lc_regressors.dat')
                elif self.GP_lc_arguments is not None:
                    self.save_regressors(self.out_folder+'GP_lc_regressors.dat', self.GP_lc_arguments, self.global_lc_model)
            if (not os.path.exists(self.out_folder+'GP_rv_regressors.dat')):
                if self.GPrveparamfile is not None:
                    os.system('cp '+self.GPrveparamfile+' '+self.out_folder+'GP_rv_regressors.dat')
                elif self.GP_rv_arguments is not None:
                    self.save_regressors(self.out_folder+'GP_rv_regressors.dat', self.GP_rv_arguments, self.global_rv_model)
            if (not os.path.exists(self.out_folder+'priors.dat')):
                self.prior_fname = self.out_folder+'priors.dat'
                self.save_priorfile(self.out_folder+'priors.dat')

    def fit(self, use_dynesty = False, dynamic = False, dynesty_bound = 'multi', dynesty_sample='rwalk', dynesty_nthreads = None, \
            n_live_points = 1000, ecclim = 1., delta_z_lim = 0.5, pl = 0.0, pu = 1.0):
        """
        Perhaps the most important function of the juliet data object. This function fits your data using the nested 
        sampler of choice. This returns a results object which contains all the posteriors information.
        """
        # Note this return call creates a fit *object* with the current data object. The fit class definition is below.
        return fit(self, use_dynesty = use_dynesty, dynamic = dynamic, dynesty_bound = dynesty_bound, dynesty_sample = dynesty_sample, \
                   dynesty_nthreads = dynesty_nthreads, n_live_points = n_live_points, ecclim = ecclim, delta_z_lim = delta_z_lim, \
                   pl = pl, pu = pu)

    def __init__(self,priors = None, input_folder = None, t_lc = None, y_lc = None, yerr_lc = None, \
                 t_rv = None, y_rv = None, yerr_rv = None, GP_regressors_lc = None, linear_regressors_lc = None, \
                 GP_regressors_rv = None, linear_regressors_rv = None, 
                 out_folder = None, lcfilename = None, rvfilename = None, GPlceparamfile = None,\
                 GPrveparamfile = None, lctimedef = 'TDB', rvtimedef = 'UTC',\
                 ld_laws = 'quadratic', priorfile = None, lc_n_supersamp = None, lc_exptime_supersamp = None, \
                 lc_instrument_supersamp = None, verbose = False):

        self.lcfilename = lcfilename
        self.rvfilename = rvfilename
        self.GPlceparamfile = GPlceparamfile
        self.GPrveparamfile = GPrveparamfile
        self.verbose = verbose

        # Initialize data options for lightcurves:
        self.t_lc = None
        self.y_lc = None
        self.yerr_lc = None
        self.instruments_lc = None
        self.ninstruments_lc = None
        self.inames_lc = None
        self.instrument_indexes_lc = None
        self.lm_lc_boolean = None
        self.lm_lc_arguments = None
        self.GP_lc_arguments = None
        self.lctimedef = lctimedef
        self.ld_laws = ld_laws
        self.lc_n_supersamp = lc_n_supersamp
        self.lc_exptime_supersamp = lc_exptime_supersamp
        self.lc_instrument_supersamp = lc_instrument_supersamp
        self.lc_data = False
        self.global_lc_model = False
        self.lc_options = {}

        # Initialize data options for RVs:
        self.t_rv = None
        self.y_rv = None 
        self.yerr_rv = None
        self.instruments_rv = None
        self.ninstruments_rv = None
        self.inames_rv = None
        self.instrument_indexes_rv = None
        self.lm_rv_boolean = None
        self.lm_rv_arguments = None
        self.GP_rv_arguments = None
        self.rvtimedef = rvtimedef
        self.rv_data = False
        self.global_rv_model = False
        self.rv_options = {}

        self.out_folder = None

        if input_folder is not None:
            if input_folder[-1] != '/':
                self.input_folder = input_folder + '/'
            else:
                self.input_folder = input_folder
            if os.path.exists(self.input_folder+'lc.dat'):
                lcfilename = self.input_folder+'lc.dat'
            if os.path.exists(self.input_folder+'rvs.dat'):
                rvfilename = self.input_folder+'rvs.dat'
            if (not os.path.exists(self.input_folder+'lc.dat')) and (not os.path.exists(self.input_folder+'rvs.dat')):
                raise Exception('INPUT ERROR: No lightcurve data file (lc.dat) or radial-velocity data file (rvs.dat) found in folder '+self.input_folder+\
                                '. \n Create them and try again. For details, check juliet.load?')
            if os.path.exists(self.input_folder+'GP_lc_regressors.dat'):
                GPlceparamfile = self.input_folder+'GP_lc_regressors.dat'
            if os.path.exists(self.input_folder+'GP_rv_regressors.dat'):
                GPrveparamfile = self.input_folder+'GP_rv_regressors.dat'
            if os.path.exists(self.input_folder+'priors.dat'):
                priors = self.input_folder+'priors.dat'
            else:
                raise Exception('INPUT ERROR: Prior file (priors.dat) not found in folder '+self.input_folder+'.'+\
                                'Create it and try again. For details, check juliet.load?')
        else:
            self.input_folder = None

        if type(priors) == str:
           self.prior_fname = priors
           priors,n_transit,n_rv,numbering_transit,numbering_rv,n_params =  readpriors(priors)
           # Save information stored in the prior: the dictionary, number of transiting planets, 
           # number of RV planets, numbering of transiting and rv planets (e.g., if p1 and p3 transit 
           # and all of them are RV planets, numbering_transit = [1,3] and numbering_rv = [1,2,3]). 
           # Save also number of *free* parameters (FIXED don't count here).
           self.priors = priors
           self.n_transiting_planets = n_transit
           self.n_rv_planets = n_rv
           self.numbering_transiting_planets = numbering_transit
           self.numbering_rv_planets = numbering_rv
           self.nparams = n_params
        elif type(priors) == dict:
           # Dictionary was passed, so save it.
           self.priors = priors
           # Extract same info as above if-statement but using only the dictionary:
           n_transit,n_rv,numbering_transit,numbering_rv,n_params =  readpriors(priors)
           # Save information:
           self.n_transiting_planets = n_transit
           self.n_rv_planets = n_rv
           self.numbering_transiting_planets = numbering_transit
           self.numbering_rv_planets = numbering_rv
           self.nparams = n_params
           self.prior_fname = None
        else:
           raise Exception('INPUT ERROR: Prior file is not a string or a dictionary (and it has to). Do juliet.load? for details.')

        # Define cases in which data is given through files: 
        if (t_lc is None):
            if lcfilename is not None:
                t_lc,y_lc,yerr_lc,instruments_lc,instrument_indexes_lc,ninstruments_lc,inames_lc,lm_lc_boolean,lm_lc_arguments = \
                read_data(lcfilename)

                # Save data to object:
                self.set_lc_data(t_lc, y_lc, yerr_lc, instruments_lc,instrument_indexes_lc,ninstruments_lc,inames_lc,lm_lc_boolean,lm_lc_arguments)
        if (t_rv is None): 
            if rvfilename is not None:
                t_rv,y_rv,yerr_rv,instruments_rv,instrument_indexes_rv,ninstruments_rv,inames_rv,lm_rv_boolean,lm_rv_arguments = \
                read_data(rvfilename)

                # Save data to object:
                self.set_rv_data(t_rv,y_rv,yerr_rv,instruments_rv,instrument_indexes_rv,ninstruments_rv,inames_rv,lm_rv_boolean,lm_rv_arguments)

        if (t_lc is None and t_rv is None):
            if (lcfilename is None) and (rvfilename is None): 
                raise Exception('INPUT ERROR: No complete dataset (photometric or radial-velocity) given.\n'+\
                      ' Make sure to feed times (t_lc and/or t_rv), values (y_lc and/or y_rv), \n'+\
                      ' errors (yerr_lc and/or yerr_rv).')

        # Read GP regressors if given through files or arrays. The former takes priority. First lightcurve:
        if GPlceparamfile is not None:
            self.GP_lc_arguments, self.global_lc_model = readGPeparams(GPlceparamfile)
        elif GP_regressors_lc is not None:
            self.GP_lc_arguments = GP_regressors_lc
            instruments = set(list(self.GP_lc_arguments.keys()))

        # Same thing for RVs:
        if GPrveparamfile is not None:
            self.GP_rv_arguments, self.global_rv_model = readGPeparams(GPrveparamfile)
        elif GP_regressors_rv is not None:
            self.GP_rv_arguments = GP_regressors_rv
            instruments = set(list(self.GP_rv_arguments.keys()))

        # If data given through direct arrays (i.e., not data files), generate some useful internal lightcurve arrays: inames_lc, which have the different lightcurve instrument names, 
        # instrument_indexes_lc (dictionary that holds, for each instrument, the indexes that have the time/lightcurve data for that particular instrument), lm_lc_boolean (dictionary of 
        # booleans; True for an instrument if it has linear regressors), lm_lc_arguments (dictionary containing the linear regressors for each instrument), etc.:
        if (lcfilename is None) and (t_lc is not None):
            # First check user gave all data:
            input_error_catcher(t_lc,y_lc,yerr_lc,'lightcurve')
            # Convert times to float64 (batman really hates non-float64 inputs):
            for instrument in t_lc.keys():
                t_lc[instrument] = t_lc[instrument].astype('float64') 
            # Create global arrays:
            tglobal_lc, yglobal_lc, yglobalerr_lc, instruments_lc = self.convert_input_data(t_lc, y_lc, yerr_lc)
            # Save data in a format useful for global modelling:
            inames_lc, instrument_indexes_lc, lm_lc_boolean, lm_lc_arguments = self.data_preparation(tglobal_lc,instruments_lc,linear_regressors_lc,linear_instruments_lc)
            ninstruments_lc = len(inames_lc)

            # Save data to object:
            self.set_lc_data(tglobal_lc, yglobal_lc, yglobalerr_lc, instruments_lc,instrument_indexes_lc,ninstruments_lc,inames_lc,lm_lc_boolean,lm_lc_arguments)

            # Save input dictionaries:
            self.times_lc = t_lc
            self.data_lc = y_lc
            self.errors_lc = yerr_lc
        elif t_lc is not None:
            # In this case, convert data in array-form to dictionaries, save them so user can easily use them:
            times_lc, data_lc, errors_lc = self.convert_to_dictionary(t_lc, y_lc, yerr_lc, instrument_indexes_lc)
            self.times_lc = times_lc
            self.data_lc = data_lc
            self.errors_lc = errors_lc

        # Same for radial-velocity data:
        if (rvfilename is None) and (t_rv is not None):
            input_error_catcher(t_rv,y_rv,yerr_rv,'radial-velocity')
            tglobal_rv, yglobal_rv, yglobalerr_rv, instruments_rv = self.convert_input_data(t_rv, y_rv, yerr_rv)
            inames_rv, instrument_indexes_rv, lm_rv_boolean, lm_rv_arguments = self.data_preparation(t_rv,instruments_rv,linear_regressors_rv,linear_instruments_rv)
            ninstruments_rv = len(inames_rv)

            # Save data to object:
            self.set_rv_data(tglobal_rv,yglobal_rv,yglobalerr_rv,instruments_rv,instrument_indexes_rv,ninstruments_rv,inames_rv,lm_rv_boolean,lm_rv_arguments)
            
            # Save input dictionaries:
            self.times_rv = t_rv    
            self.data_rv = y_rv    
            self.errors_rv = yerr_rv
        elif t_rv is not None:
            # In this case, convert data in array-form to dictionaries, save them so user can easily use them:
            times_rv, data_rv, errors_rv = self.convert_to_dictionary(t_rv, y_rv, yerr_rv, instrument_indexes_rv)
            self.times_rv = times_rv
            self.data_rv = data_rv
            self.errors_rv = errors_rv

        # If out_folder does not exist, create it, and save data to it:
        if out_folder is not None:
            self.out_folder = out_folder
            self.save()
        # Finally, generate datadicts, that will save information about the fits, including gaussian_process objects for each instrument that requires it 
        # (including the case of global models):  
        if t_lc is not None:
            self.generate_datadict('lc')
        if t_rv is not None:
            self.generate_datadict('rv')

class fit(object):
    """
    Given a juliet data object, this class performs a fit to the data and returns a results object to explore the 
    results. Example usage:

               >>> results = juliet.fit(data)

    :params data: (juliet object)
        An object containing all the information regarding the data to be fitted, including options of the fit. 
        Generated via juliet.load().

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

    :param n_live_points: (optional, int)            
        Number of live-points to be sampled. Default is ``500``.

    :param ecclim: (optional, float)                   
        Upper limit on the maximum eccentricity to sample. Default is ``1``.

    :param delta_z_lim: (optional, double)
        Define the convergence delta_z limit for the nested samplers. Default is 0.5.

    :param pl: (optional, float)                      
        If the ``(r1,r2)`` parametrization for ``(b,p)`` is used, this defines the lower limit of the planet-to-star radius ratio to be sampled. 
        Default is ``0``.

    :param pu: (optional, float)                    
        Same as ``pl``, but for the upper limit. Default is ``1``.

    :param ta: (optional, float)
        Time to be substracted to the input times in order to generate the linear and/or quadratic trend to be added to the model. 
        Default is 2458460.
    """

    def set_prior_transform(self):
        for pname in self.model_parameters:
            if self.data.priors[pname]['distribution'] != 'fixed':
                if self.data.priors[pname]['distribution'] == 'uniform':
                    self.transform_prior[pname] = transform_uniform
                if self.data.priors[pname]['distribution'] == 'normal':
                    self.transform_prior[pname] = transform_normal
                if self.data.priors[pname]['distribution'] == 'truncatednormal':
                    self.transform_prior[pname] = transform_truncated_normal
                if self.data.priors[pname]['distribution'] == 'jeffreys' or self.data.priors[pname]['distribution'] =='loguniform':
                    self.transform_prior[pname] = transform_loguniform
                if self.data.priors[pname]['distribution'] == 'beta':
                    self.transform_prior[pname] = transform_beta
                if self.data.priors[pname]['distribution'] == 'exponential':
                    self.transform_prior[pname] = exponential

    def prior(self, cube, ndim = None, nparams = None):
        pcounter = 0
        for pname in self.model_parameters:
            if self.data.priors[pname]['distribution'] != 'fixed':
                if self.use_dynesty:
                    self.transformed_priors[pcounter] = self.transform_prior[pname](cube[pcounter], \
                                                                             self.data.priors[pname]['hyperparameters']) 
                else:
                    cube[pcounter] = self.transform_prior[pname](cube[pcounter], \
                                                          self.data.priors[pname]['hyperparameters'])
                pcounter += 1
        if self.use_dynesty:
            return self.transformed_priors

    def loglike(self, cube, ndim=None, nparams=None):
        # Evaluate the joint log-likelihood. For this, first extract all inputs:
        pcounter = 0
        for pname in self.model_parameters:
            if self.data.priors[pname]['distribution'] != 'fixed':
                self.posteriors[pname] = cube[pcounter]
                pcounter += 1
        # Initialize log-likelihood:
        log_likelihood = 0.0

        # Evaluate photometric model first:
        if self.data.t_lc is not None:
             self.lc.generate(self.posteriors)
             if self.lc.modelOK:
                 log_likelihood += self.lc.get_log_likelihood(self.posteriors)
             else:
                 return -1e101
        # Now RV model:
        if self.data.t_rv is not None:
             self.rv.generate(self.posteriors)
             if self.rv.modelOK:
                 log_likelihood += self.rv.get_log_likelihood(self.posteriors)
             else:
                 return -1e101
      
        # Return total log-likelihood:
        return log_likelihood

    def __init__(self, data, use_dynesty = False, dynamic = False, dynesty_bound = 'multi', dynesty_sample='rwalk', dynesty_nthreads = None, \
                       n_live_points = 1000, ecclim = 1., delta_z_lim = 0.5, pl = 0.0, pu = 1.0, ta = 2458460.):

        # Define output results object:
        self.results = None
        # Save sampler inputs:
        self.use_dynesty = use_dynesty
        self.dynamic = dynamic
        self.dynesty_bound = dynesty_bound
        self.dynesty_sample = dynesty_sample
        self.dynesty_nthreads = dynesty_nthreads
        self.n_live_points = n_live_points
        self.ecclim = ecclim 
        self.delta_z_lim = delta_z_lim
        self.pl = pl
        self.pu = pu
        self.ta = ta
        # Inhert data object:
        self.data = data
        # Inhert some other fit options:
        #if self.data.t_lc is not None:
        #    if True in self.data.lc_dict['efficient_bp']:
        #        self.pu = pu
        #        self.pl = pl
        #        self.Ar = (self.pu - self.pl)/(2. + self.pl + self.pu)
        # Inhert the output folder:
        self.out_folder = data.out_folder
        self.transformed_priors = np.zeros(self.data.nparams)

        # Define prefixes in case saving is turned on (i.e., user passed an out_folder):
        if self.use_dynesty:
            if self.dynamic:
                self.sampler_prefix = 'dynamic_dynesty_'
            else:
                self.sampler_prefix = 'dynesty_'
        else:
            self.sampler_prefix = 'multinest_'

        # Generate a posteriors self that will save the current values of each of the parameters:
        self.posteriors = {}
        self.model_parameters = list(self.data.priors.keys())
        for pname in self.model_parameters:
            if self.data.priors[pname]['distribution'] == 'fixed':
                self.posteriors[pname] = self.data.priors[pname]['hyperparameters']
            else:
                self.posteriors[pname] = 0.#self.data.priors[pname]['cvalue']

        # For each of the variables in the prior that is not fixed, define an internal dictionary that will save the 
        # corresponding transformation function to the prior corresponding to that variable. Idea is that with this one 
        # simply does self.transform_prior[variable_name](value) and you get the transformed value to the 0,1 prior. 
        # This avoids having to keep track of the prior distribution on each of the interations:
        self.transform_prior = {} 
        self.set_prior_transform()

        # Generate light-curve and radial-velocity models:
        if self.data.t_lc is not None:
            self.lc = model(self.data, modeltype = 'lc', pl = self.pl, pu = self.pu, ecclim = self.ecclim, log_like_calc = True)
        if self.data.t_rv is not None:
            self.rv = model(self.data, modeltype = 'rv', ecclim = self.ecclim, ta = self.ta, log_like_calc = True)

        # Before starting, check if force_dynesty or force_pymultinest is on; change options accordingly:
        if force_dynesty and (not self.use_dynesty):
            print('PyMultinest installation not detected. Forcing dynesty as the sampler.')
            self.use_dynesty = True
        if force_pymultinest and self.use_dynesty:
            print('dynesty installation not detected. Forcing PyMultinest as the sampler.')
            self.use_dynesty = False

        # If not ran and saved already, run dynesty or MultiNest, and save posterior samples and evidences to pickle file:
        out = {}
        runMultiNest = False
        runDynesty = False
        if not self.use_dynesty:
            if self.out_folder is None:
                out_folder = os.getcwd()+'/'
                runMultiNest = True
            else:
                if (not os.path.exists(self.out_folder+'posteriors.pkl')):
                    runMultiNest = True
            if runMultiNest:
                pymultinest.run(self.loglike, self.prior, self.data.nparams, \
                                n_live_points = self.n_live_points,\
                                max_modes = 100,\
                                outputfiles_basename = self.out_folder + 'jomnest_', resume = False,\
                                verbose = self.data.verbose)
                # Run and get output:
                output = pymultinest.Analyzer(outputfiles_basename = self.out_folder + 'jomnest_', n_params = self.data.nparams)
                # Get out parameters: this matrix has (samples,n_params+1):
                posterior_samples = output.get_equal_weighted_posterior()[:,:-1]
                # Get INS lnZ:
                out['lnZ'] = output.get_stats()['global evidence']
                out['lnZerr'] = output.get_stats()['global evidence error']
                if self.out_folder is None:
                    os.system('rm '+out_folder+'jomnest_*')
        elif self.use_dynesty:
            if self.out_folder is None:
                runDynesty = True
            else:
                if self.dynamic and (not os.path.exists(self.out_folder+'_dynesty_DNS_posteriors.pkl')):
                    DynestySampler = dynesty.DynamicNestedSampler
                    runDynesty = True
                elif (not self.dynamic) and (not os.path.exists(self.out_folder+'_dynesty_NS_posteriors.pkl')):
                    DynestySampler = dynesty.NestedSampler
                    runDynesty = True
            if runDynesty:
                if self.dynesty_nthreads is None:
                    sampler = dynesty.DynamicNestedSampler(self.loglike, self.prior, self.data.nparams, nlive = self.n_live_points, \
                                                           bound = self.dynesty_bound, sample = self.dynesty_sample)
                    # Run and get output:
                    sampler.run_nested()
                    results = sampler.results
                else:
                    from multiprocessing import Pool
                    import contextlib
                    nthreads = int(self.dynesty_nthreads)
                    with contextlib.closing(Pool(processes=nthreads-1)) as executor:
                        sampler = dynesty.DynamicNestedSampler(self.loglike, self.prior, self.data.nparams, nlive = self.n_live_points, \
                                                              bound = self.dynesty_bound, sample = self.dynesty_sample, pool=executor, queue_size=nthreads)
                        sampler.run_nested()
                        results = sampler.results
                out['dynesty_output'] = results
                # Get weighted posterior:
                weights = np.exp(results['logwt'] - results['logz'][-1])
                posterior_samples = resample_equal(results.samples, weights)
                # Get lnZ:
                out['lnZ'] = results.logz[-1]
                out['lnZerr'] = results.logzerr[-1]
        if runMultiNest or runDynesty:
            out['posterior_samples'] = {}
            out['posterior_samples']['unnamed'] = posterior_samples
            # Extract parameters:
            pcounter = 0
            for pname in self.model_parameters:
                if data.priors[pname]['distribution'] != 'fixed':
                    self.posteriors[pname] = np.median(posterior_samples[:,pcounter])
                    out['posterior_samples'][pname] = posterior_samples[:,pcounter]
                    pcounter += 1
            
            if self.data.t_lc is not None:
                if True in self.data.lc_options['efficient_bp']:
                    out['pu'] = self.pu
                    out['pl'] = self.pl
            if self.data.t_rv is not None:
                if self.data.rv_options['fitrvline'] or self.data.rv_options['fitrvquad']:
                    out['ta'] = self.ta
            if runDynesty:
                if self.dynamic and (self.out_folder is not None):
                    pickle.dump(out,open(self.out_folder+'_dynesty_DNS_posteriors.pkl','wb'))
                elif (not self.dynamic) and (self.out_folder is not None):
                    pickle.dump(out,open(self.out_folder+'_dynesty_NS_posteriors.pkl','wb'))
            else:
                if self.out_folder is not None:
                    pickle.dump(out,open(self.out_folder+'posteriors.pkl','wb'))
        else:
            # Probably already ran any of the above, so read the outputs:
            if (self.use_dynesty) and (self.out_folder is not None):
                if self.dynamic:
                    if os.path.exists(self.out_folder+'_dynesty_DNS_posteriors.pkl'):
                        if data.self.verbose:
                            print('Detected (dynesty) Dynamic NS output files --- extracting...')
                        out = pickle.load(open(self.out_folder+'_dynesty_DNS_posteriors.pkl','rb'))
                else:
                    if os.path.exists(self.out_folder+'_dynesty_NS_posteriors.pkl'):
                        if data.self.verbose:
                            print('Detected (dynesty) NS output files --- extracting...')
                        out = pickle.load(open(self.out_folder+'_dynesty_NS_posteriors.pkl','rb'))
            elif self.out_folder is not None:
                if self.data.verbose:
                    print('Detected (MultiNest) NS output files --- extracting...')
                out = pickle.load(open(self.out_folder+'posteriors.pkl','rb')) 
            if len(out.keys()) == 0:
                print('Warning: no output generated or extracted. Check the fit options given to juliet.fit().')
            else:
                # For retro-compatibility, check for sigma_w_rv_instrument and add an extra variable on out 
                # for sigma_w_instrument:
                for pname in out['posterior_samples'].keys():
                    if 'sigma_w_rv' == pname[:10]:
                        instrument = pname.split('_')[-1]
                        out['posterior_samples']['sigma_w_'+instrument] = out['posterior_samples'][pname]
                # Extract parameters:
                for pname in self.posteriors.keys():
                    if data.priors[pname]['distribution'] != 'fixed':
                        self.posteriors[pname] = np.median(out['posterior_samples'][pname])
                posterior_samples = out['posterior_samples']['unnamed']
                if 'pu' in out.keys():
                    self.pu = out['pu']
                    self.pl = out['pl']
                    self.Ar = (self.pu - self.pl)/(2. + self.pl + self.pu)
                if 'ta' in out.keys():
                    self.ta = out['ta']
        # Either fit done or extracted. If doesn't exist, create the posteriors.dat file:
        if self.out_folder is not None:
            if not os.path.exists(self.out_folder+'posteriors.dat'):
                outpp = open(self.out_folder+'posteriors.dat','w')
                writepp(outpp,out, data.priors)

        # Save all results (posteriors) to the self.results object:
        self.posteriors = out
  
        # Save posteriors to lc and rv:
        if self.data.t_lc is not None:
            self.lc.set_posterior_samples(out['posterior_samples'])
        if self.data.t_rv is not None:
            self.rv.set_posterior_samples(out['posterior_samples'])

class model(object):
    """
    Given a juliet data object, this kernel generates either a lightcurve or a radial-velocity object. Example usage:

               >>> model = juliet.model(data, modeltype = 'lc')

    :param data: (juliet.load object)
        An object containing all the information about the current dataset.

    :param modeltype: (optional, string)
        String indicating whether the model to generate should be a lightcurve ('lc') or a radial-velocity ('rv') model. 

    :param pl: (optional, float)                      
        If the ``(r1,r2)`` parametrization for ``(b,p)`` is used, this defines the lower limit of the planet-to-star radius ratio to be sampled. 
        Default is ``0``.

    :param pu: (optional, float)                    
        Same as ``pl``, but for the upper limit. Default is ``1``.

    :param ecclim: (optional, float)
        This parameter sets the maximum eccentricity allowed such that a model is actually evaluated. Default is ``1``.

    :param log_like_calc: (optional, boolean)
        If True, it is assumed the model is generated to generate likelihoods values, and thus this skips the saving/calculation of the individual 
        models per planet (i.e., ``self.model['p1']``, ``self.model['p2']``, etc. will not exist). Default is False.

    """
    def init_batman(self, t, ld_law, nresampling = None, etresampling = None):
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
         if ld_law == 'linear':
             params.u = [0.5]
         else:
             params.u = [0.1,0.3]
         params.limb_dark = ld_law
         if nresampling is None or etresampling is None:
             m = batman.TransitModel(params, t)
         else:
             m = batman.TransitModel(params, t, supersample_factor=nresampling, exp_time=etresampling)
         return params,m

    def init_catwoman(self, t, ld_law, nresampling = None, etresampling = None):
         """  
         This function initializes the batman code.
         """
         params = batman.TransitParams()
         params.t0 = 0.
         params.per = 1.
         params.rp = 0.1
         params.rp2 = 0.1
         params.a = 15.
         params.inc = 87.
         params.ecc = 0.
         params.w = 90.
         params.phi = 90.
         if ld_law == 'linear':
             params.u = [0.5]
         else:
             params.u = [0.1,0.3]
         params.limb_dark = ld_law
         if nresampling is None or etresampling is None:
             m = catwoman.TransitModel(params, t)
         else:
             m = catwoman.TransitModel(params, t, supersample_factor=nresampling, exp_time=etresampling)
         return params,m

    def init_radvel(self, nplanets=1):
        return radvel.model.Parameters(nplanets,basis='per tc e w k')  

    def generate_rv_model(self, parameter_values):
        self.modelOK = True
        # Before anything continues, check the periods are chronologically ordered (this is to avoid multiple modes due to 
        # periods "jumping" between planet numbering):
        first_time = True 
        for i in self.numbering:
            if first_time:
                cP = parameter_values['P_p'+str(i)]
                first_time = False
            else:
                if cP < parameter_values['P_p'+str(i)]:
                    cP = parameter_values['P_p'+str(i)]
                else:
                    self.modelOK = False
                    return False

        # First, extract orbital parameters and save them, which will be common to all instruments:
        for n in range(self.nplanets):
            i = self.numbering[n]

            # Semi-amplitudes, t0 and P:
            K, t0, P = parameter_values['K_p'+str(i)], parameter_values['t0_p'+str(i)], parameter_values['P_p'+str(i)]

            # Extract eccentricity and omega depending on the used parametrization for each planet:
            if self.dictionary['ecc_parametrization'][i] == 0:
                ecc,omega = parameter_values['ecc_p'+str(i)], parameter_values['omega_p'+str(i)]*np.pi/180.
            elif self.dictionary['ecc_parametrization'][i] == 1:
                ecc = np.sqrt(parameter_values['ecosomega_p'+str(i)]**2+parameter_values['esinomega_p'+str(i)]**2)
                omega = np.arctan2(parameter_values['esinomega_p'+str(i)],parameter_values['ecosomega_p'+str(i)])
            else:
                ecc = parameter_values['secosomega_p'+str(i)]**2+parameter_values['sesinomega_p'+str(i)]**2
                omega = np.arctan2(parameter_values['sesinomega_p'+str(i)],parameter_values['secosomega_p'+str(i)])

            # Generate lightcurve for the current planet if ecc is OK:
            if ecc > self.ecclim:
                self.modelOK = False
                return False

            # Save them to radvel:
            self.model['radvel']['per'+str(n+1)] = radvel.Parameter(value = P)
            self.model['radvel']['tc'+str(n+1)] = radvel.Parameter(value = t0)
            self.model['radvel']['w'+str(n+1)] = radvel.Parameter(value = omega) # note given in radians
            self.model['radvel']['e'+str(n+1)] = radvel.Parameter(value = ecc)
            self.model['radvel']['k'+str(n+1)] = radvel.Parameter(value = K)

        # If log_like_calc is True (by default during juliet.fit), don't bother saving the RVs of planet p_i:
        if self.log_like_calc:
            self.model['Keplerian'] = radvel.model.RVModel(self.model['radvel']).__call__(self.t)
        else:
            self.model['Keplerian'] = radvel.model.RVModel(self.model['radvel']).__call__(self.t)
            for n in range(self.nplanets):
                i = self.numbering[n]
                self.model['p'+str(i)] = radvel.model.RVModel(self.model['radvel']).__call__(self.t,planet_num=n+1)

        # If trends are being fitted, add them to the Keplerian+Trend model:
        if self.dictionary['fitrvline']:
            self.model['Keplerian+Trend'] = self.model['Keplerian'] + parameter_values['rv_intercept'] + (self.t - self.ta)*parameter_values['rv_slope'] 
        elif self.dictionary['fitrvquad']:
            self.model['Keplerian+Trend'] = self.model['Keplerian'] + parameter_values['rv_intercept'] + (self.t - self.ta)*parameter_values['rv_slope'] + \
                                                                      ((self.t - self.ta)**2)*parameter_values['rv_quad']
        else:
            self.model['Keplerian+Trend'] = self.model['Keplerian']

        # Populate the self.model[instrument]['deterministic'] array. This hosts the full (deterministic) model for each RV instrument.
        for instrument in self.inames:
            self.model[instrument]['deterministic'] = self.model['Keplerian+Trend'][self.instrument_indexes[instrument]] + parameter_values['mu_'+instrument] 
            self.model[instrument]['deterministic_variances'] = self.errors[instrument]**2 + parameter_values['sigma_w_'+instrument]**2
            if self.lm_boolean[instrument]:
                self.model[instrument]['LM'] = np.zeros(self.ndatapoints_per_instrument[instrument])
                for i in range(self.lm_n[instrument]):
                    self.model[instrument]['LM'] += parameter_values['theta'+str(i)+'_'+instrument]*self.lm_arguments[instrument][:,i]
                self.model[instrument]['deterministic'] += self.model[instrument]['LM']
            # If the model under consideration is a global model, populate the global model dictionary:
            if self.global_model:
                self.model['global'][self.instrument_indexes[instrument]] = self.model[instrument]['deterministic']
                self.model['global_variances'][self.instrument_indexes[instrument]] = self.yerr[self.instrument_indexes[instrument]]**2 + \
                                                                                      parameter_values['sigma_w_'+instrument]**2
        
    def get_GP_plus_deterministic_model(self, parameter_values, instrument = None):
        if self.global_model:
            if self.dictionary['global_model']['GPDetrend']:
                residuals = self.y - self.model['global']
                self.dictionary['global_model']['noise_model'].set_parameter_vector(parameter_values)
                self.dictionary['global_model']['noise_model'].yerr = np.sqrt(self.model['global_variances'])
                self.dictionary['global_model']['noise_model'].compute_GP()
                # Return mean signal plus GP model:
                self.model['GP'] = self.dictionary['global_model']['noise_model'].GP.predict(residuals, self.dictionary['global_model']['noise_model'].X, \
                                                                                                   return_var=False, return_cov=False)
                return self.model['global'], self.model['GP'], elf.model['global'] + self.model['GP']
            else:
                return self.model['global'] 
        else:
            if self.dictionary[instrument]['GPDetrend']:
                residuals = self.residuals#self.data[instrument] - self.model[instrument]['deterministic']
                self.dictionary[instrument]['noise_model'].set_parameter_vector(parameter_values)
                self.model[instrument]['GP'] = self.dictionary[instrument]['noise_model'].GP.predict(residuals,self.dictionary[instrument]['noise_model'].X, \
                                               return_var=False, return_cov=False)
                return self.model[instrument]['deterministic'], self.model[instrument]['GP'], self.model[instrument]['deterministic'] + self.model[instrument]['GP']
            else:
                return self.model[instrument]['deterministic']

    def evaluate_model(self, instrument = None, parameter_values = None, resampling = None, nresampling = None, etresampling = None, \
                          all_samples = False, nsamples = 1000, return_samples = False, t = None, GPregressors = None, LMregressors = None, \
                          return_err = False, alpha = 0.68, return_components = False):
        """
        This function evaluates the current lc or rv  model given a set of parameter values. Resampling options can be changed if resampling is a boolean,
        but the object is at the end rolled-back to the default resampling definitions the user defined in the juliet.load object. For now, resampling only is 
        available for lightcurve models. TODO: add resampling for RVs.
        """
        if resampling is not None and self.modeltype == 'lc':
            if resampling:
                self.model[instrument]['params'], self.model[instrument]['m'] = self.init_batman(self.times[instrument], self.dictionary[instrument]['ldlaw'],\
                                                                                                 nresampling = nresampling,\
                                                                                                 etresampling = etresampling)
            else:
                self.model[instrument]['params'], self.model[instrument]['m'] = self.init_batman(self.times[instrument], self.dictionary[instrument]['ldlaw'])

        # Check if user gave input parameter_values dictionary. If that's the case, generate again the 
        # full lightcurve/rv model:
        if parameter_values is not None:
            # If return_components, generate the components dictionary:
            if return_components:
                self.log_like_calc = False
                components = {}
            # Now, consider two possible cases. If the user is giving a parameter_values where the dictionary contains *arrays* of values 
            # in it, then iterate through all the values in order to calculate the median model. If the dictionary contains only individual 
            # values, evaluate the model only at those values:
            parameters = list(self.priors.keys())
            input_parameters = list(parameter_values.keys())
            if type(parameter_values[input_parameters[0]]) is np.ndarray:
                # To generate a median model first generate an output_model_samples array that will save the model at each evaluation. This will 
                # save nsamples samples of the posterior model. If all_samples = True, all samples from the posterior are used for the evaluated model 
                # (this is slower, but you might not care). First create idx_samples, which will save the samples:
                nsampled = len(parameter_values[input_parameters[0]])
                if all_samples:
                    nsamples = nsampled
                    idx_samples = np.arange(nsamples)
                else:
                    idx_samples = np.random.choice(np.arange(nsampled),np.min([nsamples,nsampled]),replace=False)
                    idx_samples = idx_samples[np.argsort(idx_samples)]
                
                # Create the output_model arrays: these will save on each iteration the full model (lc/rv + GP, output_model_samples), 
                # the GP-only model (GP, output_modelGP_samples) and the lc/rv-only model (lc/rv, output_modelDET_samples) --- the latter ones 
                # will make sense only if there is a GP model:
                if t is None:
                    if self.global_model:
                        output_model_samples = np.zeros([nsamples,self.ndatapoints_all_instruments])
                    else:
                        output_model_samples = np.zeros([nsamples,self.ndatapoints_per_instrument[instrument]])
                else:
                    nt = len(t)
                    nt_original = len(self.times[instrument])
                    output_model_samples = np.zeros([nsamples,nt])
                    original_instrument_times = np.copy(self.times[instrument])
                    if self.modeltype == 'lc':
                        if self.dictionary[instrument]['TransitFit']:
                            if not self.dictionary[instrument]['TransitFitCatwoman']:
                                supersample_params,supersample_m = self.init_batman(t, self.dictionary[instrument]['ldlaw'])
                                sample_params,sample_m = self.init_batman(self.times[instrument], self.dictionary[instrument]['ldlaw'])
                            else:
                                supersample_params,supersample_m = self.init_catwoman(t, self.dictionary[instrument]['ldlaw'])
                                sample_params,sample_m = self.init_catwoman(self.times[instrument], self.dictionary[instrument]['ldlaw'])             
                    else:
                        original_t = np.copy(self.t)
                        original_instrument_indexes = np.copy(self.instrument_indexes)
                        # If not global, assume indexes for selected instrument are all the user-inputted t's
                        if not self.global_model:
                            dummy_indexes = np.arange(len(t))
                            # Set inames to just the name of the instrument under consideration:
                            self.iname = [instrument]
                            original_instrument_index = self.instrument_indexes[instrument]

                # Fill the components dictionary in case return_components is true; use the output_model_samples for the size of each component array:
                if return_components:
                    for i in self.numbering:
                        components['p'+str(i)] = np.zeros(output_model_samples.shape)
                    if self.modeltype == 'lc':
                        components['lm'] = np.zeros(output_model_samples.shape)
                        components['transit'] = np.zeros(output_model_samples.shape)
                    else:
                        components['keplerian'] = np.zeros(output_model_samples.shape)
                        components['trend'] = np.zeros(output_model_samples.shape) 
                        components['lm'] = np.zeros(output_model_samples.shape)
                        components['mu'] = np.zeros(output_model_samples.shape[0])

                # Save the original inames. If not global model, we won't care about generating the models for the other instruments:
                if not self.global_model:
                    original_inames = np.copy(self.inames)
                    # Set inames to just the name of the instrument under consideration:
                    self.inames = [instrument]

                if self.dictionary[instrument]['GPDetrend']:
                    output_modelGP_samples = np.copy(output_model_samples)
                    output_modelDET_samples = np.copy(output_model_samples)

                # Create dictionary that saves the current parameter_values to evaluate:
                current_parameter_values = dict.fromkeys(parameters)
                # First go through all parameters in the prior; fix the ones which are fixed:
                for parameter in parameters:
                    if self.priors[parameter]['distribution'] == 'fixed': 
                        current_parameter_values[parameter] = self.priors[parameter]['hyperparameters']
                # If extrapolating the model, save the current times, current GPregressors and current linear 
                # regressors: 
                if t is not None:
                    if self.dictionary[instrument]['GPDetrend']:
                        original_GPregressors = np.copy(self.dictionary[instrument]['noise_model'].X)
                        self.dictionary[instrument]['noise_model'].X = GPregressors
                    if self.lm_boolean[instrument]:
                        original_lm_arguments  = np.copy(self.lm_arguments[instrument])
                # Now iterate through all samples, through all the input parameters:
                counter = 0
                for i in idx_samples:
                    for parameter in input_parameters:
                        # Populate the current parameter_values
                        current_parameter_values[parameter] = parameter_values[parameter][i]
                    # Evaluate rv/lightcurve at the current parameter values, calculate residuals, save them:
                    if self.modeltype == 'lc':
                        self.generate_lc_model(current_parameter_values)
                    else:
                        self.generate_rv_model(current_parameter_values)
                    self.residuals = self.data[instrument] - self.model[instrument]['deterministic']
                    # If extrapolating (t is not None), evaluate the extrapolated model with a lightcurve/rv model 
                    # considering the input times and not the current dataset times:
                    if t is not None:
                        if self.modeltype == 'lc':
                            if self.dictionary[instrument]['TransitFit']:
                                self.model[instrument]['params'], self.model[instrument]['m'] = supersample_params,supersample_m
                            if self.lm_boolean[instrument]:
                                self.lm_arguments[instrument] = LMregressors
                            self.model[instrument]['ones'] = np.ones(nt)
                            self.ndatapoints_per_instrument[instrument] = nt 
                            self.generate_lc_model(current_parameter_values)    
                        else:
                            self.t = t
                            if not self.global_model:
                                self.times[instrument] = t
                                self.instrument_indexes[instrument] = dummy_indexes
                            self.generate_rv_model(current_parameter_values)

                    if self.dictionary[instrument]['GPDetrend']:
                        output_modelDET_samples[counter,:], output_modelGP_samples[counter,:], output_model_samples[counter,:] = \
                                                                 self.get_GP_plus_deterministic_model(current_parameter_values, \
                                                                                                         instrument = instrument)
                    else:
                        output_model_samples[counter,:] = self.get_GP_plus_deterministic_model(current_parameter_values, \
                                                                                                  instrument = instrument)

                    if return_components:
                        if self.modeltype == 'lc':
                            transit = 0.
                            for i in self.numbering:
                                components['p'+str(i)][counter,:] = self.model[instrument]['p'+str(i)]
                                transit += (components['p'+str(i)][counter,:] - 1.)
                            components['transit'][counter,:] = 1. + transit
                        else:
                            for i in self.numbering:
                                components['p'+str(i)][counter,:] = self.model['p'+str(i)] 
                            components['trend'][counter,:] = self.model['Keplerian+Trend'] - self.model['Keplerian']
                            components['keplerian'][counter,:] = self.model['Keplerian']
                            components['mu'][counter] = current_parameter_values['mu_'+instrument]
                        if self.lm_boolean[instrument]:
                            if self.modeltype == 'lc':
                                components['lm'][counter,:] = self.model[instrument]['LM']
                            else:
                                components['lm'][counter,:] = self.model[instrument]['LM']

                    # Rollback in case t is not None:
                    if t is not None:
                        self.times[instrument] = original_instrument_times
                        if self.modeltype == 'lc':
                            if self.dictionary[instrument]['TransitFit']:
                                self.model[instrument]['params'], self.model[instrument]['m'] = sample_params,sample_m
                            if self.lm_boolean[instrument]:
                                self.lm_arguments[instrument] = original_lm_arguments
                            self.model[instrument]['ones'] = np.ones(nt_original)
                        else:
                            self.t = original_t
                            if not self.global_model:
                                self.instrument_indexes[instrument] = original_instrument_index 
                        self.ndatapoints_per_instrument[instrument] = nt_original
                        
                    counter += 1
                # If return_error is on, return upper and lower sigma (68% CI) of the model(s):
                if return_err:
                    m_output_model, u_output_model, l_output_model = np.zeros(output_model_samples.shape[1]),\
                                                                     np.zeros(output_model_samples.shape[1]),\
                                                                     np.zeros(output_model_samples.shape[1])

                    if self.dictionary[instrument]['GPDetrend']:
                        LCm_output_model, LCu_output_model, LCl_output_model = np.copy(m_output_model), np.copy(u_output_model), \
                                                                               np.copy(l_output_model) 

                        GPm_output_model, GPu_output_model, GPl_output_model = np.copy(m_output_model), np.copy(u_output_model), \
                                                                               np.copy(l_output_model)
                    for i in range(output_model_samples.shape[1]):
                        m_output_model[i], u_output_model[i], l_output_model[i] = get_quantiles(output_model_samples[:,i], alpha = alpha)
                        if self.dictionary[instrument]['GPDetrend']:
                            mLC_output_model[i], uLC_output_model[i], lLC_output_model[i] = get_quantiles(output_modelDET_samples[:,i], alpha = alpha)
                            mGP_output_model[i], uGP_output_model[i], lGP_output_model[i] = get_quantiles(output_modelGP_samples[:,i], alpha = alpha)
                    if self.dictionary[instrument]['GPDetrend']: 
                        self.model[instrument]['deterministic'], self.model[instrument]['GP'] = mLC_output_model, mGP_output_model
                        self.model[instrument]['deterministic_uerror'], self.model[instrument]['GP_uerror'] = uLC_output_model, uGP_output_model
                        self.model[instrument]['deterministic_lerror'], self.model[instrument]['GP_lerror'] = lLC_output_model, lGP_output_model
                else:
                    output_model = np.median(output_model_samples,axis=0)
                    if self.dictionary[instrument]['GPDetrend']:
                        self.model[instrument]['deterministic'], self.model[instrument]['GP'] = np.median(output_modelDET_samples,axis=0), \
                                                                                       np.median(output_modelGP_samples,axis=0)

                # If return_components is true, generate the median models for each part of the full model:
                if return_components:
                    for k in components.keys():
                        components[k] = np.median(components[k],axis=0)
            else:
                self.generate_lc_model(parameter_values)
                output_model = self.get_GP_plus_deterministic_model(parameter_values, instrument = instrument)
                if return_components:
                    if self.modeltype == 'lc':
                        transit = 0.
                        for i in self.numbering:
                            components['p'+str(i)] = self.model[instrument]['p'+str(i)]
                            transit += (components['p'+str(i)] - 1.)
                        components['transit'] = 1. + transit
                    else:
                        for i in self.numbering:
                            components['p'+str(i)] = self.model['p'+str(i)]
                        components['trend'] = self.model['Keplerian+Trend'] - self.model['Keplerian']
                        components['keplerian'] = self.model['Keplerian']
                        components['mu'] = parameter_values['mu_'+instrument]
                    if self.lm_boolean[instrument]:
                        components['lm'] = self.model[instrument]['LM']
        else:
         
            x = self.evaluate_model(instrument = instrument, parameter_values = self.posteriors, resampling = resampling, \
                                              nresampling = nresampling, etresampling = etresampling, all_samples = all_samples, \
                                              nsamples = nsamples, return_samples = return_samples, t = t, GPregressors = GPregressors, \
                                              LMregressors = LMregressors, return_err = return_err, return_components = return_components)
            if return_samples:
                if return_err:
                    if return_components:
                        output_model_samples, m_output_model, u_output_model, l_output_model = x
                    else:
                        output_model_samples, m_output_model, u_output_model, l_output_model, components = x
                else:
                    if return_components:
                        output_model_samples,output_model,components = x
                    else:
                        output_model_samples,output_model = x
            else:
                if return_err:
                    if return_components:
                        m_output_model, u_output_model, l_output_model, components = x
                    else:
                        m_output_model, u_output_model, l_output_model = x
                else:
                    if return_components:
                        output_model, components = x
                    else:
                        output_model = x

        if resampling is not None: 
             # get lc, return, then turn all back to normal:
             if self.dictionary[instrument]['resampling']:
                 self.model[instrument]['params'], self.model[instrument]['m'] = self.init_batman(self.times[instrument], self.dictionary[instrument]['ldlaw'],\
                                                                                                  nresampling = self.dictionary[instrument]['nresampling'],\
                                                                                                  etresampling = self.dictionary[instrument]['exptimeresampling'])
             else:
                 self.model[instrument]['params'], self.model[instrument]['m'] = self.init_batman(self.times[instrument], self.dictionary[instrument]['ldlaw'])
        if return_samples:
            if return_err:
                if return_components:
                    return output_model_samples, m_output_model, u_output_model, l_output_model, components
                else:
                    return output_model_samples, m_output_model, u_output_model, l_output_model
            else:
                if return_components:
                    return output_model_samples,output_model, components
                else:
                    return output_model_samples,output_model
        else:
            if return_err:
                if return_components:
                    return m_output_model, u_output_model, l_output_model, components
                else:
                    return m_output_model, u_output_model, l_output_model
            else:
                if return_components:
                    return output_model, components
                else:
                    return output_model

    def generate_lc_model(self, parameter_values):
        self.modelOK = True
        # Before anything continues, check the periods are chronologically ordered (this is to avoid multiple modes due to 
        # periods "jumping" between planet numbering):
        first_time = True
        for i in self.numbering:
            if first_time:
                cP = parameter_values['P_p'+str(i)]
                first_time = False
            else:
                if cP < parameter_values['P_p'+str(i)]:
                    cP = parameter_values['P_p'+str(i)]
                else:
                    self.modelOK = False
                    return False

        # Start loop to populate the self.model[instrument]['deterministic_model'] array, which will host the complete lightcurve for a given 
        # instrument (including flux from all the planets). Do the for loop per instrument for the parameter extraction, so in the 
        # future we can do, e.g., wavelength-dependant rp/rs.
        for instrument in self.inames:
            # Set full array to ones by copying:
            self.model[instrument]['M'] = np.copy(self.model[instrument]['ones'])
            # If transit fit is on, then model the transit lightcurve:
            if self.dictionary[instrument]['TransitFit']:
                # Extract and set the limb-darkening coefficients for the instrument:
                if self.dictionary[instrument]['ldlaw'] != 'linear':
                    coeff1, coeff2 = reverse_ld_coeffs(self.dictionary[instrument]['ldlaw'], parameter_values['q1_'+self.ld_iname[instrument]],\
                                                       parameter_values['q2_'+self.ld_iname[instrument]])
                else:
                    coeff1 = parameter_values['q1_'+self.ld_iname[instrument]]

                # Now loop through all the planets, getting the model for each:
                for i in self.numbering:
                    if self.dictionary['efficient_bp'][i]:
                        if not self.dictionary['fitrho']:
                            a,r1,r2,t0,P = parameter_values['a_p'+str(i)], parameter_values['r1_p'+str(i)],\
                                           parameter_values['r2_p'+str(i)], parameter_values['t0_p'+str(i)],\
                                           parameter_values['P_p'+str(i)]
                        else:
                            rho,r1,r2,t0,P = parameter_values['rho'], parameter_values['r1_p'+str(i)],\
                                             parameter_values['r2_p'+str(i)], parameter_values['t0_p'+str(i)],\
                                             parameter_values['P_p'+str(i)] 
                            a = ((rho*G*((P*24.*3600.)**2))/(3.*np.pi))**(1./3.)
                        if r1 > self.Ar:
                            b,p = (1+self.pl)*(1. + (r1-1.)/(1.-self.Ar)),\
                                  (1-r2)*self.pl + r2*self.pu
                        else:
                            b,p = (1. + self.pl) + np.sqrt(r1/self.Ar)*r2*(self.pu-self.pl),\
                                  self.pu + (self.pl-self.pu)*np.sqrt(r1/self.Ar)*(1.-r2)
                    else:
                       if not self.dictionary['fitrho']:
                           if not self.dictionary[instrument]['TransitFitCatwoman']:
                               a,b,p,t0,P = parameter_values['a_p'+str(i)], parameter_values['b_p'+str(i)],\
                                            parameter_values['p_p'+str(i)], parameter_values['t0_p'+str(i)],\
                                            parameter_values['P_p'+str(i)]
                           else:
                               a,b,p1,p2,t0,P,phi = parameter_values['a_p'+str(i)], parameter_values['b_p'+str(i)],\
                                            parameter_values['p1_p'+str(i)], parameter_values['p2_p'+str(i)], \
                                            parameter_values['t0_p'+str(i)], parameter_values['P_p'+str(i)], \
                                            parameter_values['phi_p'+str(i)]
                               p = np.min([p1,p2])
                       else:
                           if not self.dictionary[instrument]['TransitFitCatwoman']:
                                rho,b,p,t0,P = parameter_values['rho'], parameter_values['b_p'+str(i)],\
                                               parameter_values['p_p'+str(i)], parameter_values['t0_p'+str(i)],\
                                               parameter_values['P_p'+str(i)]
                           else:
                                rho,b,p1,p2,t0,P,phi = parameter_values['rho'], parameter_values['b_p'+str(i)],\
                                               parameter_values['p1_p'+str(i)], parameter_values['p2_p'+str(i)],\
                                               parameter_values['t0_p'+str(i)], parameter_values['P_p'+str(i)],\
                                               parameter_values['phi_p'+str(i)]
                                p = np.min([p1,p2])
                           a = ((rho*G*((P*24.*3600.)**2))/(3.*np.pi))**(1./3.)

                    # Now extract eccentricity and omega depending on the used parametrization for each planet:
                    if self.dictionary['ecc_parametrization'][i] == 0:
                        ecc,omega = parameter_values['ecc_p'+str(i)], parameter_values['omega_p'+str(i)]
                    elif self.dictionary['ecc_parametrization'][i] == 1:
                        ecc = np.sqrt(parameter_values['ecosomega_p'+str(i)]**2+parameter_values['esinomega_p'+str(i)]**2)
                        omega = np.arctan2(parameter_values['esinomega_p'+str(i)],parameter_values['ecosomega_p'+str(i)])*180./np.pi
                    else:
                        ecc = parameter_values['secosomega_p'+str(i)]**2+parameter_values['sesinomega_p'+str(i)]**2
                        omega = np.arctan2(parameter_values['sesinomega_p'+str(i)],parameter_values['secosomega_p'+str(i)])*180./np.pi

                    # Generate lightcurve for the current planet if ecc is OK:
                    if ecc > self.ecclim:
                        self.modelOK = False
                        return False
                    else:
                        ecc_factor = (1. + ecc*np.sin(omega * np.pi/180.))/(1. - ecc**2)
                        inc_inv_factor = (b/a)*ecc_factor
                        if not (b>1.+p or inc_inv_factor >=1.):
                            self.model[instrument]['params'].t0 = t0
                            self.model[instrument]['params'].per = P
                            self.model[instrument]['params'].a = a
                            self.model[instrument]['params'].inc = np.arccos(inc_inv_factor)*180./np.pi
                            self.model[instrument]['params'].ecc = ecc 
                            self.model[instrument]['params'].w = omega
                            if not self.dictionary[instrument]['TransitFitCatwoman']:
                                self.model[instrument]['params'].rp = p
                            else:
                                self.model[instrument]['params'].rp = p1
                                self.model[instrument]['params'].rp2 = p2
                                self.model[instrument]['params'].phi = phi
                            if self.dictionary[instrument]['ldlaw'] is not 'linear':
                               self.model[instrument]['params'].u = [coeff1, coeff2]
                            else:
                               self.model[instrument]['params'].u = [coeff1]
                            # If log_like_calc is True (by default during juliet.fit), don't bother saving the lightcurve of planet p_i:
                            if self.log_like_calc:
                                self.model[instrument]['M'] += self.model[instrument]['m'].light_curve(self.model[instrument]['params']) - 1.
                            else:
                                self.model[instrument]['p'+str(i)] = self.model[instrument]['m'].light_curve(self.model[instrument]['params'])
                                self.model[instrument]['M'] += self.model[instrument]['p'+str(i)] - 1.
                        else:
                            self.modelOK = False   
                            return False 
                    
            # Once either the transit model is generated or after populating the full_model with ones if no transit fit is on, 
            # convert the lightcurve so it complies with the juliet model accounting for the dilution and the mean out-of-transit flux:
            D, M = parameter_values['mdilution_'+self.mdilution_iname[instrument]], parameter_values['mflux_'+instrument]
            self.model[instrument]['M'] = (self.model[instrument]['M']*D + (1. - D))*(1./(1. + D*M))

            # Now, if a linear model was defined, generate it and add it to the full model:
            if self.lm_boolean[instrument]:
                self.model[instrument]['LM'] = np.zeros(self.ndatapoints_per_instrument[instrument])
                for i in range(self.lm_n[instrument]):
                    self.model[instrument]['LM'] += parameter_values['theta'+str(i)+'_'+instrument]*self.lm_arguments[instrument][:,i]
                self.model[instrument]['deterministic'] = self.model[instrument]['M'] + self.model[instrument]['LM']
            else:
                self.model[instrument]['deterministic'] = self.model[instrument]['M']
            self.model[instrument]['deterministic_variances'] = self.errors[instrument]**2 + (parameter_values['sigma_w_'+instrument]*1e-6)**2
            # Finally, if the model under consideration is a global model, populate the global model dictionary:
            if self.global_model:
                self.model['global'][self.instrument_indexes[instrument]] = self.model[instrument]['deterministic']
                self.model['global_variances'][self.instrument_indexes[instrument]] = self.yerr[self.instrument_indexes[instrument]]**2 + \
                                                                                      (parameter_values['sigma_w_'+instrument]*1e-6)**2

    def gaussian_log_likelihood(self, residuals, variances):
        taus = 1./variances
        return -0.5*(len(residuals)*log2pi+np.sum(-np.log(taus)+taus*(residuals**2)))

    def get_log_likelihood(self, parameter_values):
        if self.global_model:
            residuals = self.y - self.model['global']
            if self.dictionary['global_model']['GPDetrend']:
                self.dictionary['global_model']['noise_model'].set_parameter_vector(parameter_values)
                self.dictionary['global_model']['noise_model'].yerr = np.sqrt(self.model['global_variances'])
                self.dictionary['global_model']['noise_model'].compute_GP()
                return self.dictionary['global_model']['noise_model'].GP.log_likelihood(residuals)
            else:
                self.gaussian_log_likelihood(residuals,self.model['global_variances'])
        else:
            log_like = 0.0
            for instrument in self.inames:
                residuals = self.data[instrument] - self.model[instrument]['deterministic']
                if self.dictionary[instrument]['GPDetrend']:
                    self.dictionary[instrument]['noise_model'].set_parameter_vector(parameter_values)
                    log_like += self.dictionary[instrument]['noise_model'].GP.log_likelihood(residuals)
                else:
                    log_like += self.gaussian_log_likelihood(residuals,self.model[instrument]['deterministic_variances'])
            return log_like 

    def set_posterior_samples(self, posterior_samples):
        self.posteriors = posterior_samples
        self.median_posterior_samples = {}
        for parameter in self.posteriors.keys():
            if parameter is not 'unnamed':
                self.median_posterior_samples[parameter] = np.median(self.posteriors[parameter])
        for parameter in self.priors:
            if self.priors[parameter]['distribution'] == 'fixed':
                self.median_posterior_samples[parameter] = self.priors[parameter]['hyperparameters']
        try:
            self.generate(self.median_posterior_samples)
        except:
            print('Warning: model evaluated at the posterior median did not compute properly.')

    def __init__(self, data, modeltype, pl = 0.0, pu = 1.0, ecclim = 1., ta = 2458460., log_like_calc = False):
        # Inhert the priors dictionary from data:
        self.priors = data.priors
        # Define the ecclim value:
        self.ecclim = ecclim
        # Define ta:
        self.ta = ta
        # Save the log_like_calc boolean:
        self.log_like_calc = log_like_calc
        # Define variable that at each iteration defines if the model is OK or not (not OK means something failed in terms of the 
        # parameter space being explored):
        self.modelOK = True
        # Define a variable that will save the posterior samples:
        self.posteriors = None
        self.median_posterior_samples = None
        # Number of datapoints per instrument variable:
        self.ndatapoints_per_instrument = {}
        if modeltype == 'lc':
            self.modeltype = 'lc'
            # Inhert times, fluxes, errors, indexes, etc. from data.
            # FYI, in case this seems confusing: self.t, self.y and self.yerr save what we internally call 
            # "global" data-arrays. These have the data from all the instruments stacked into an array; to recover 
            # the data for a given instrument, one uses the self.instrument_indexes dictionary. On the other hand, 
            # self.times, self.data and self.errors are dictionaries that on each key have the data of a given instrument.      
            # Calling dictionaries is faster than calling indexes of arrays, so we use the latter in general to evaluate models.
            self.t = data.t_lc
            self.y = data.y_lc
            self.yerr = data.yerr_lc
            self.times = data.times_lc
            self.data = data.data_lc
            self.errors = data.errors_lc
            self.instruments = data.instruments_lc
            self.ninstruments = data.ninstruments_lc
            self.inames = data.inames_lc
            self.instrument_indexes = data.instrument_indexes_lc
            self.lm_boolean = data.lm_lc_boolean
            self.lm_arguments = data.lm_lc_arguments
            self.lm_n = {}
            self.pl = pl
            self.pu = pu
            self.Ar = (self.pu - self.pl)/(2. + self.pl + self.pu)
            self.global_model = data.global_lc_model
            self.dictionary = data.lc_options
            self.numbering = data.numbering_transiting_planets
            self.numbering.sort()
            self.nplanets = len(self.numbering)
            self.model = {}
            # First, if a global model, generate array that will save this:
            if self.global_model:
                self.model['global'] = np.zeros(len(self.t))
                self.model['global_errors'] = np.zeros(len(self.t))
            # If limb-darkening or dilution factors will be shared by different instruments, set the correct variable name for each:
            self.ld_iname = {}
            self.mdilution_iname = {}
            self.ndatapoints_all_instruments = 0.
            for instrument in self.inames:
                self.model[instrument] = {}
                # Extract number of datapoints per instrument:
                self.ndatapoints_per_instrument[instrument] = len(self.instrument_indexes[instrument])
                self.ndatapoints_all_instruments += self.ndatapoints_per_instrument[instrument]
                # Extract number of linear model terms per instrument:
                if self.lm_boolean[instrument]:
                    self.lm_n[instrument] = self.lm_arguments[instrument].shape[1]
                # An array of ones to copy around:
                self.model[instrument]['ones'] = np.ones(len(self.instrument_indexes[instrument]))
                # Generate internal model variables of interest to the user. First, the lightcurve model in the notation of juliet (Mi) 
                # (full lightcurve plus dilution factors and mflux):
                self.model[instrument]['M'] = np.ones(len(self.instrument_indexes[instrument]))
                # Linear model (in the notation of juliet, LM):
                self.model[instrument]['LM'] = np.zeros(len(self.instrument_indexes[instrument]))
                # Now, generate dictionary that will save the final full, deterministic model (M + LM):
                self.model[instrument]['deterministic'] = np.zeros(len(self.instrument_indexes[instrument]))
                # Same for the errors:
                self.model[instrument]['deterministic_errors'] = np.zeros(len(self.instrument_indexes[instrument]))
                if self.dictionary[instrument]['TransitFit']:
                    # First, take the opportunity to initialize transit lightcurves for each instrument:
                    if self.dictionary[instrument]['resampling']:
                        if not self.dictionary[instrument]['TransitFitCatwoman']:
                            self.model[instrument]['params'], self.model[instrument]['m'] = self.init_batman(self.times[instrument], self.dictionary[instrument]['ldlaw'],\
                                                                                                         nresampling = self.dictionary[instrument]['nresampling'],\
                                                                                                         etresampling = self.dictionary[instrument]['exptimeresampling'])
                        else:
                            self.model[instrument]['params'], self.model[instrument]['m'] = self.init_catwoman(self.times[instrument], self.dictionary[instrument]['ldlaw'],\
                                                                                                         nresampling = self.dictionary[instrument]['nresampling'],\
                                                                                                         etresampling = self.dictionary[instrument]['exptimeresampling'])
                    else:
                        if not self.dictionary[instrument]['TransitFitCatwoman']:
                            self.model[instrument]['params'], self.model[instrument]['m'] = self.init_batman(self.times[instrument], \
                                                                                                             self.dictionary[instrument]['ldlaw'])
                        else:
                            self.model[instrument]['params'], self.model[instrument]['m'] = self.init_catwoman(self.times[instrument], \
                                                                                                               self.dictionary[instrument]['ldlaw'])
                    # Individual transit lightcurves for each planet:
                    for i in self.numbering:
                        self.model[instrument]['p'+str(i)] = np.ones(len(self.instrument_indexes[instrument]))
                    # Now proceed with instrument namings:
                    for pname in self.priors.keys():
                        # Check if variable name is a limb-darkening coefficient:
                        if pname[0:2] == 'q1':
                            vec = pname.split('_')
                            if len(vec)>2:
                                if instrument in vec:
                                    self.ld_iname[instrument] = '_'.join(vec[1:])
                            else:
                                if instrument in vec:
                                    self.ld_iname[instrument] = vec[1]
                        # Check if it is a dilution factor:
                        if pname[0:9] == 'mdilution':
                            vec = pname.split('_')
                            if len(vec)>2:
                                if instrument in vec:
                                    self.mdilution_iname[instrument] = '_'.join(vec[1:])
                            else:
                                self.mdilution_iname[instrument] = vec[1]
                else:
                    # Now proceed with instrument namings:
                    for pname in self.priors.keys():
                        # Check if it is a dilution factor:
                        if pname[0:9] == 'mdilution':
                            vec = pname.split('_')
                            if len(vec)>2:
                                if instrument in vec:
                                    self.mdilution_iname[instrument] = '_'.join(vec[1:])
                            else:        
                                self.mdilution_iname[instrument] = vec[1]
            # Set the model-type to M(t):
            self.evaluate = self.evaluate_model
            self.generate = self.generate_lc_model
        elif modeltype == 'rv':
            self.modeltype = 'rv'
            # Inhert times, RVs, errors, indexes, etc. from data:
            self.t = data.t_rv
            self.y = data.y_rv
            self.yerr = data.yerr_rv
            self.times = data.times_rv
            self.data = data.data_rv
            self.errors = data.errors_rv
            self.instruments = data.instruments_rv
            self.ninstruments = data.ninstruments_rv
            self.inames = data.inames_rv
            self.instrument_indexes = data.instrument_indexes_rv
            self.lm_boolean = data.lm_rv_boolean
            self.lm_arguments = data.lm_rv_arguments
            self.lm_n = {}
            self.global_model = data.global_rv_model
            self.dictionary = data.rv_options
            self.numbering = data.numbering_rv_planets
            self.numbering.sort()
            self.nplanets = len(self.numbering)
            self.model = {}
            self.ndatapoints_all_instruments = 0.
            # First, if a global model, generate array that will save this:
            if self.global_model:
                self.model['global'] = np.zeros(len(self.t))
                self.model['global_errors'] = np.zeros(len(self.t))
            # Initialize radvel:
            self.model['radvel'] = self.init_radvel(nplanets=self.nplanets)
            # First go around all planets to compute the full RV models:
            for i in self.numbering:
                self.model['p'+str(i)] = np.ones(len(self.t))
            # Now variable to save full RV Keplerian model:
            self.model['Keplerian'] = np.ones(len(self.t))
            # Same for Keplerian + trends:
            self.model['Keplerian+Trend'] = np.ones(len(self.t))
            # Go around each instrument:
            for instrument in self.inames:
                self.model[instrument] = {}
                # Extract number of datapoints per instrument:
                self.ndatapoints_per_instrument[instrument] = len(self.instrument_indexes[instrument])
                self.ndatapoints_all_instruments += self.ndatapoints_per_instrument[instrument]
                # Extract number of linear model terms per instrument:
                if self.lm_boolean[instrument]:
                    self.lm_n[instrument] = self.lm_arguments[instrument].shape[1]
                # Generate internal model variables of interest to the user. First, the RV model in the notation of juliet (Mi) 
                # (full RV model plus offset velocity, plus trend):
                self.model[instrument]['M'] = np.ones(len(self.instrument_indexes[instrument]))
                # Linear model (in the notation of juliet, LM):
                self.model[instrument]['LM'] = np.zeros(len(self.instrument_indexes[instrument]))
                # Now, generate dictionary that will save the final full model (M + LM):
                self.model[instrument]['deterministic'] = np.zeros(len(self.instrument_indexes[instrument]))
                # Same for the errors:
                self.model[instrument]['deterministic_errors'] = np.zeros(len(self.instrument_indexes[instrument]))
                # Individual keplerians for each planet:
                for i in self.numbering:
                    self.model[instrument]['p'+str(i)] = np.ones(len(self.instrument_indexes[instrument]))
                # An array of ones to copy around:
                self.model[instrument]['ones'] = np.ones(len(self.t[self.instrument_indexes[instrument]]))
            # Set the model-type to M(t):
            self.evaluate = self.evaluate_model
            self.generate = self.generate_rv_model
        else:
            raise Exception('Model type "'+lc+'" not recognized. Currently it can only be "lc" for a light-curve model or "rv" for radial-velocity model.')
       
class gaussian_process(object):
    """
    Given a juliet data object (created via juliet.load), a model type (i.e., is this a GP for a RV or lightcurve dataset) and 
    an instrument name, this object generates a Gaussian Process (GP) object to use within the juliet library. Example usage:

               >>> GPmodel = juliet.gaussian_process(data, model_type = 'lc', instrument = 'TESS')

    :param data (juliet.load object)
        Object containing all the information about the current dataset. This will help in determining the type of kernel 
        the input instrument has and also if the instrument has any errors associated with it to initialize the kernel.

    :param model_type: (string)
        A string defining the type of data the GP will be modelling. Can be either ``lc`` (for photometry) or ``rv`` (for radial-velocities).

    :param instrument: (string)
        A string indicating the name of the instrument the GP is being applied to. This string simplifies cross-talk with juliet's ``posteriors`` 
        dictionary.

    :param george_hodlr: (optional, boolean)
        If True, this uses George's HODLR solver (faster).

    """
 
    def get_kernel_name(self,priors):
        # First, check all the GP variables in the priors file that are of the form GP_variable_instrument1_instrument2_...:
        variables_that_match = []
        for pname in priors.keys():
            vec = pname.split('_')
            if (vec[0] == 'GP') and (self.instrument in vec):
                variables_that_match = variables_that_match + [vec[1]]
        # Now we have all the variables that match the current instrument in variables_that_match. Check which of the 
        # implemented GP models gives a perfect match to all the variables; that will give us the name of the kernel:
        n_variables_that_match = len(variables_that_match)
        if n_variables_that_match  == 0:
            raise Exception('Input error: it seems instrument '+self.instrument+' has no defined priors in the prior file for a Gaussian Process. Check the prior file and try again.')

        for kernel_name in self.all_kernel_variables.keys():
            counter = 0
            for variable_name in self.all_kernel_variables[kernel_name]:
                if variable_name in variables_that_match:
                    counter += 1
            if (n_variables_that_match == counter) and (len(self.all_kernel_variables[kernel_name]) == n_variables_that_match):
                return kernel_name

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
        # If, for example, GP_Prot_TESS_K2_rv and GP_Gamma_TESS, and self.variables = ['Prot','Gamma'], 
        # then self.input_instrument = ['TESS_K2_rv','TESS'].
        for i in range(len(self.variables)):
            GPvariable = self.variables[i]
            for pnames in input_variables.keys():
                vec = pnames.split('_')
                if (vec[0] == 'GP') and (GPvariable in vec[1]) and (self.instrument in vec):
                    self.input_instrument.append('_'.join(vec[2:]))

    def set_parameter_vector(self, parameter_values):
        # To update the parameters, we have to transform the juliet inputs to celerite/george inputs. Update this 
        # depending on the kernel under usage. For this, we first define a base_index variable that will define the numbering 
        # of the self.parameter_vector. The reason for this is that the dimensions of the self.parameter_vector array is 
        # different if the GP is global (i.e., self.global_GP is True --- meaning a unique GP is fitted to all instruments) or 
        # not (self.global_GP is False --- meaning a different GP per instrument is fitted). If the former, the jitter terms are 
        # modified directly by changing the self.yerr vector; in the latter, we have to manually add a jitter term in the GP parameter 
        # vector. This base_index is only important for the george kernels though --- an if statement suffices for the celerite ones.
        base_index = 0
        if self.kernel_name == 'SEKernel':
            if not self.global_GP:
                self.parameter_vector[base_index] = np.log((parameter_values['sigma_w_'+self.instrument]*self.sigma_factor)**2)
                base_index += 1
            self.parameter_vector[base_index] = np.log((parameter_values['GP_sigma_'+self.input_instrument[0]]*self.sigma_factor)**2.)
            for i in range(self.nX):
                self.parameter_vector[base_index + 1 + i] = np.log(1./priors['GP_alpha'+str(i)+'_'+self.input_instrument[1+i]])
        elif self.kernel_name == 'ExpSineSquaredSEKernel':
            if not self.global_GP:
                self.parameter_vector[base_index] = np.log((parameter_values['sigma_w_'+self.instrument]*self.sigma_factor)**2)
                base_index += 1
            self.parameter_vector[base_index] = np.log((parameter_values['GP_sigma_'+self.input_instrument[0]]*self.sigma_factor)**2.)
            self.parameter_vector[base_index + 1] = np.log(1./(parameter_values['GP_alpha_'+self.input_instrument[1]]))
            self.parameter_vector[base_index + 2] = parameter_values['GP_Gamma_'+self.input_instrument[2]]
            self.parameter_vector[base_index + 3] = np.log(parameter_values['GP_Prot_'+self.input_instrument[3]])
        elif self.kernel_name == 'CeleriteQPKernel':
            self.parameter_vector[0] = np.log(parameter_values['GP_B_'+self.input_instrument[0]])
            self.parameter_vector[1] = np.log(parameter_values['GP_L_'+self.input_instrument[1]])
            self.parameter_vector[2] = np.log(parameter_values['GP_Prot_'+self.input_instrument[2]])
            self.parameter_vector[3] = np.log(parameter_values['GP_C_'+self.input_instrument[3]])
            if not self.global_GP:
                self.parameter_vector[4] = np.log(parameter_values['sigma_w_'+self.instrument]*self.sigma_factor)
        elif self.kernel_name == 'CeleriteExpKernel':
            self.parameter_vector[0] = np.log(parameter_values['GP_sigma_'+self.input_instrument[0]])
            self.parameter_vector[1] = np.log(parameter_values['GP_timescale_'+self.input_instrument[1]])
            if not self.global_GP:
                self.parameter_vector[2] = np.log(parameter_values['sigma_w_'+self.instrument]*self.sigma_factor)
        elif self.kernel_name == 'CeleriteMaternKernel':
            self.parameter_vector[0] = np.log(parameter_values['GP_sigma_'+self.input_instrument[0]])
            self.parameter_vector[1] = np.log(parameter_values['GP_rho_'+self.input_instrument[1]])
            if not self.global_GP:
                self.parameter_vector[2] = np.log(parameter_values['sigma_w_'+self.instrument]*self.sigma_factor)
        elif self.kernel_name == 'CeleriteMaternExpKernel':
            self.parameter_vector[0] = np.log(parameter_values['GP_sigma_'+self.input_instrument[0]])
            self.parameter_vector[1] = np.log(parameter_values['GP_timescale_'+self.input_instrument[1]])
            self.parameter_vector[3] = np.log(parameter_values['GP_rho_'+self.input_instrument[2]])
            if not self.global_GP:
                self.parameter_vector[4] = np.log(parameter_values['sigma_w_'+self.instrument]*self.sigma_factor)
        elif self.kernel_name == 'CeleriteSHOKernel':
            self.parameter_vector[0] = np.log(parameter_values['GP_S0_'+self.input_instrument[0]])
            self.parameter_vector[1] = np.log(parameter_values['GP_Q_'+self.input_instrument[1]])
            self.parameter_vector[2] = np.log(parameter_values['GP_omega0_'+self.input_instrument[2]])
            if not self.global_GP:
                self.parameter_vector[3] = np.log(parameter_values['sigma_w_'+self.instrument]*self.sigma_factor)
        self.GP.set_parameter_vector(self.parameter_vector) 

    def __init__(self, data, model_type, instrument, george_hodlr = True):
        self.isInit = False
        self.model_type = model_type.lower()
        # Perform changes that define the model_type. For example, the juliet input sigmas (both jitters and GP amplitudes) are 
        # given in ppm in the input files, whereas for RVs they have the same units as the input RVs. This conversion factor is 
        # defined by the model_type:
        if self.model_type == 'lc':
            if instrument is None:
                instrument = 'lc'
            self.sigma_factor = 1e-6
        elif self.model_type == 'rv':
            if instrument is None:
                instrument = 'rv'
            self.sigma_factor = 1.
        else:
            raise Exception('Model type '+model_type+' currently not supported. Only "lc" or "rv" can serve as inputs for now.')

        # Name of input instrument if given:
        self.instrument = instrument
        
        # Extract information from the data object:
        if self.model_type == 'lc':
            # Save input predictor:
            if instrument == 'lc':
                self.X = data.GP_lc_arguments['lc']
            else:
                self.X = data.GP_lc_arguments[instrument]
            # Save errors (if any):
            if data.yerr_lc is not None:
                if instrument != 'lc':
                     self.yerr = data.yerr_lc[data.instrument_indexes_lc[instrument]]
                else:
                     self.yerr = data.yerr_lc
            else:
                self.yerr = None
        elif self.model_type == 'rv':
            # Save input predictor:
            if instrument == 'rv':
                self.X = data.GP_rv_arguments['rv']
            else:
                self.X = data.GP_rv_arguments[instrument]
            # Save errors (if any):
            if data.yerr_rv is not None:
                if instrument != 'rv':
                    self.yerr = data.yerr_rv[data.instrument_indexes_rv[instrument]]
                else:
                    self.yerr = data.yerr_rv
            else:
                self.yerr = None

        # Fix sizes of regressors if wrong:
        if len(self.X.shape) == 2:
            if self.X.shape[1] != 1:
                self.nX = self.X.shape[1]
            else:
                self.X = self.X[:,0]
                self.nX = 1
        else:
            self.nX = 1

        # Define all possible kernels available by the object:
        self.all_kernel_variables = {}
        self.all_kernel_variables['SEKernel'] = ['sigma'] 
        for i in range(self.nX):
            self.all_kernel_variables['SEKernel'] = self.all_kernel_variables['SEKernel'] + ['alpha'+str(i)]
        self.all_kernel_variables['ExpSineSquaredSEKernel'] = ['sigma','alpha','Gamma','Prot']
        self.all_kernel_variables['CeleriteQPKernel'] = ['B','L','Prot','C']
        self.all_kernel_variables['CeleriteExpKernel'] = ['sigma','timescale']
        self.all_kernel_variables['CeleriteMaternKernel'] = ['sigma','rho']
        self.all_kernel_variables['CeleriteMaternExpKernel'] = ['sigma','timescale','rho']
        self.all_kernel_variables['CeleriteSHOKernel'] = ['S0','Q','omega0']

        # Find kernel name (and save it to self.kernel_name):
        self.kernel_name = self.get_kernel_name(data.priors)
        # Initialize variable for the GP object:
        self.GP = None
        # Are we using celerite?
        self.use_celerite = False
        # Are we using george_hodlr?
        if george_hodlr:
            self.george_hodlr = True
        else:
            self.george_hodlr = False
        # Initialize variable that sets the "instrument" name for each variable (self.variables below). If, for example, 
        # GP_Prot_TESS_K2_RV and GP_Gamma_TESS, and self.variables = [Prot,Gamma], then self.instrument_variables = ['TESS_K2_RV','TESS'].
        self.input_instrument = []

        # Initialize each kernel on the GP object. First, set the variables to the ones defined above. Then initialize the 
        # actual kernel:
        self.variables = self.all_kernel_variables[self.kernel_name]
        phantomvariable = 0
        if self.kernel_name == 'SEKernel':
            # Generate GPExpSquared base kernel:
            self.kernel = 1.*george.kernels.ExpSquaredKernel(np.ones(self.nX),ndim = self.nX, axes = range(self.nX))
            # (Note no jitter kernel is given, as with george one defines this in the george.GP call):
        elif self.kernel_name == 'ExpSineSquaredSEKernel':
            # Generate the kernels:
            K1 = 1.*george.kernels.ExpSquaredKernel(metric = 1.0)
            K2 = george.kernels.ExpSine2Kernel(gamma=1.0,log_period=1.0)
            self.kernel = K1*K2
            # (Note no jitter kernel is given, as with george one defines this in the george.GP call):
        elif self.kernel_name == 'CeleriteQPKernel':
            # Generate rotational kernel:
            rot_kernel = terms.TermSum(RotationTerm(log_amp=np.log(10.),\
                                                    log_timescale=np.log(10.0),\
                                                    log_period=np.log(3.0),\
                                                    log_factor=np.log(1.0)))
            # Jitter term:
            kernel_jitter = terms.JitterTerm(np.log(100*1e-6))
            # Wrap GP kernel and object:
            if self.instrument in ['rv','lc']:
                self.kernel = rot_kernel 
            else:
                self.kernel = rot_kernel + kernel_jitter
            # We are using celerite:
            self.use_celerite = True
        elif self.kernel_name == 'CeleriteExpKernel':
            # Generate exponential kernel:
            exp_kernel = terms.RealTerm(log_a=np.log(10.), log_c=np.log(10.))
            # Jitter term:
            kernel_jitter = terms.JitterTerm(np.log(100*1e-6))
            # Wrap GP kernel and object:
            if self.instrument in ['rv','lc']:
                self.kernel = exp_kernel
            else:
                self.kernel = exp_kernel + kernel_jitter
            # We are using celerite:
            self.use_celerite = True
        elif self.kernel_name == 'CeleriteMaternKernel':
            # Generate matern kernel:
            matern_kernel = terms.Matern32Term(log_sigma=np.log(10.), log_rho=np.log(10.))
            # Jitter term:
            kernel_jitter = terms.JitterTerm(np.log(100*1e-6))
            # Wrap GP kernel and object:
            if self.instrument in ['rv','lc']:
                 self.kernel = matern_kernel
            else:
                 self.kernel = matern_kernel + kernel_jitter
            # We are using celerite:
            self.use_celerite = True
        elif self.kernel_name == 'CeleriteMaternExpKernel':
            # Generate matern and exponential kernels:
            matern_kernel = terms.Matern32Term(log_sigma=np.log(10.), log_rho=np.log(10.))
            exp_kernel = terms.RealTerm(log_a=np.log(10.), log_c=np.log(10.))
            # Jitter term:
            kernel_jitter = terms.JitterTerm(np.log(100*1e-6))
            # Wrap GP kernel and object:
            if self.instrument in ['rv','lc']:
                self.kernel = exp_kernel*matern_kernel
            else:
                self.kernel = exp_kernel*matern_kernel + kernel_jitter
            # We add a phantom variable because we want to leave index 2 without value ON PURPOSE: the idea is 
            # that here, that is always 0 (because this defines the log(sigma) of the matern kernel in the 
            # multiplication, which we set to 1).
            phantomvariable = 1
            # We are using celerite:
            self.use_celerite = True
        elif self.kernel_name == 'CeleriteSHOKernel':
            # Generate kernel:
            sho_kernel = terms.SHOTerm(log_S0=np.log(10.), log_Q=np.log(10.),log_omega0=np.log(10.))
            # Jitter term:
            kernel_jitter = terms.JitterTerm(np.log(100*1e-6))
            # Wrap GP kernel and object:
            if self.instrument in ['rv','lc']:
                self.kernel = sho_kernel
            else:
                self.kernel = sho_kernel + kernel_jitter
            # We are using celerite:
            self.use_celerite = True
        # Check if use_celerite is True; if True, check that the regressor is ordered. If not, don't do the self.init_GP():
        if self.use_celerite:
            idx_sorted = np.argsort(self.X)
            diff1 = np.sum(self.X - self.X[idx_sorted])
            diff2 = np.sum(self.X - self.X[idx_sorted[::-1]])
            if diff1 == 0 or diff1 == 0:
                self.init_GP()
                self.isInit = True
        else:
            self.init_GP()
            self.isInit = True

        if self.instrument in ['rv','lc']:
            # If instrument is 'rv' or 'lc', assume GP object will fit for a global GP 
            # (e.g., global photometric signal, or global RV signal) that assumes a given 
            # GP realization for all instruments (but allows different jitters for each 
            # instrument, added in quadrature to the self.yerr):
            self.parameter_vector = np.zeros(len(self.variables)+panthomvariable)
            self.global_GP = True
        else:
            # If GP per instrument, then there is one jitter term per instrument directly added in the model:
            self.parameter_vector = np.zeros(len(self.variables)+1+phantomvariable)
            self.global_GP = False
        self.set_input_instrument(data.priors)
