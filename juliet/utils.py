import numpy as np
from astropy.io import fits
import pickle
import batman
import radvel
# Try to import catwoman:
try: 
    import catwoman
    have_catwoman = True 
except:
    have_catwoman = False

def init_batman(t, ld_law, nresampling = None, etresampling = None):
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

def init_catwoman(t, ld_law, nresampling = None, etresampling = None):
     """  
     This function initializes the catwoman code.
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

def init_radvel(nplanets=1):
    return radvel.model.Parameters(nplanets,basis='per tc e w k')  

def mag_to_flux(m,merr):
    """ 
    Convert magnitude to relative fluxes. 
    """
    fluxes = np.zeros(len(m))
    fluxes_err = np.zeros(len(m))
    for i in range(len(m)):
        dist = 10**(-np.random.normal(m[i],merr[i],1000)/2.51)
        fluxes[i] = np.mean(dist)
        fluxes_err[i] = np.sqrt(np.var(dist))
    return fluxes,fluxes_err

def get_TESS_data(filename, fluxtype = 'PDCSAP_FLUX'):
    """
    Given a filename, this function returns an array of times,
    fluxes and errors on those fluxes.
    """
    # Manipulate the fits file:
    data = fits.getdata(filename)

    # Identify zero-flux values to take them out of the data arrays:
    idx = np.where((data[fluxtype]!=0.)&(~np.isnan(data[fluxtype])))[0]

    # Return median-normalized flux:
    return data['TIME'][idx],data[fluxtype][idx]/np.median(data[fluxtype][idx]), \
           data[fluxtype+'_ERR'][idx]/np.median(data[fluxtype][idx])

import os
try:
    has_astroquery = True
    from astroquery.mast import Observations
except:
    has_astroquery = False

def get_all_TESS_data(object_name, radius = ".02 deg", get_PDC = True, get_all = False, get_lightcurves_only = True):
    """ 
    Given a planet name, this function returns a dictionary of times, fluxes and 
    errors on fluxes in a juliet-friendly format for usage. The function does an 
    astroquery to MAST using a default radius of .02 deg around the target name. If get_PDC is True, 
    this function returns PDC fluxes. False returns SAP fluxes. If get_all is true, this function 
    returns a dictionary that in addition to the times, fluxes and errors, returns other 
    metadata.
    """
    if not has_astroquery:
        print("Error on using juliet function `get_all_TESS_data`: astroquery.mast not found.")
    obs_table = Observations.query_object(object_name,radius=radius)
    out_dict = {}
    times = {}
    fluxes = {}
    fluxes_errors = {}
    for i in range(len(obs_table['dataURL'])):
        if 's_lc.fits' in obs_table['dataURL'][i]:
            fname = obs_table['dataURL'][i].split('/')[-1]
            metadata = fname.split('-')
            if len(metadata) == 5:
                # Extract metadata:
                sector = int(metadata[1].split('s')[-1])
                ticid = int(metadata[2])
                # Download files:
                data_products = Observations.get_product_list(obs_table[i])
                if get_lightcurves_only:
                    want = data_products['description'] == "Light curves"
                else:
                    want = (data_products['description'] == "Light curves") or (data_products['description'] == "Target pixel files")
                manifest = Observations.download_products(data_products[want])
                # Read lightcurve file:
                d,h = fits.getdata('mastDownload/TESS/'+fname[:-8]+'/'+fname,header=True)
                t,fs,fserr,f,ferr = d['TIME']+h['BJDREFI'],d['SAP_FLUX'],d['SAP_FLUX_ERR'],\
                                    d['PDCSAP_FLUX'],d['PDCSAP_FLUX_ERR']
                idx_goodpdc = np.where((f != 0.)&(~np.isnan(f)))[0]
                idx_goodsap = np.where((fs != 0.)&(~np.isnan(fs)))[0]
                # Save to output dictionary:
                if 'TIC' not in out_dict.keys():
                    out_dict['TIC'] = ticid
                out_dict[sector] = {}
                out_dict[sector]['TIME_PDCSAP_FLUX'] = t[idx_goodpdc]
                out_dict[sector]['PDCSAP_FLUX'] = f[idx_goodpdc]
                out_dict[sector]['PDCSAP_FLUX_ERR'] = ferr[idx_goodpdc]
                out_dict[sector]['TIME_SAP_FLUX'] = t[idx_goodsap]
                out_dict[sector]['SAP_FLUX'] = fs[idx_goodsap]
                out_dict[sector]['SAP_FLUX_ERR'] = fserr[idx_goodsap]
                if get_PDC:
                    times['TESS'+str(sector)] = t[idx_goodpdc]
                    med = np.median(f[idx_goodpdc])
                    fluxes['TESS'+str(sector)] = f[idx_goodpdc]/med
                    fluxes_errors['TESS'+str(sector)] = ferr[idx_goodpdc]/med
                else:
                    times['TESS'+str(sector)] = t[idx_goodsap]
                    med = np.median(fs[idx_goodsap])
                    fluxes['TESS'+str(sector)] = fs[idx_goodsap]/med
                    fluxes_errors['TESS'+str(sector)] = fserr[idx_goodsap]/med
                # Remove downloaded folder:
                os.system('rm -r mastDownload')
    if get_all:
        return out_dict, times, fluxes, fluxes_errors
    else:
        return times, fluxes, fluxes_errors


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

def reverse_q_coeffs(ld_law, u1, u2): 
    if ld_law == 'quadratic':
        q1 = (u1 + u2)**2
        q2 = (u1/2.)/(u1 + u2)
    elif ld_law=='squareroot':
        q1 = (u1 + u2)**2 
        q2 = (u2/2.)/(u1 + u2)
    elif ld_law=='logarithmic':
        q1 = (1. - u2)**2
        q2 = (1. - u1)/(1. - u2)
    return q1,q2

def convert_ld_coeffs(ld_law, coeff1, coeff2):
    if ld_law == 'quadratic':
        q1 = (coeff1 + coeff2)**2
        q2 = coeff1/(2.*(coeff1+coeff2))
    elif ld_law=='squareroot':
        q1 = (coeff1 + coeff2)**2
        q2 = coeff2/(2.*(coeff1+coeff2))
    elif ld_law=='logarithmic':
        q1 = (1-coeff2)**2
        q2 = (1.-coeff1)/(1.-coeff2)
    return q1,q2

def reverse_bp(r1,r2,pl,pu):
    Ar = (pu - pl)/(2. + pl + pu)
    nsamples = len(r1)
    p = np.zeros(nsamples)
    b = np.zeros(nsamples)
    for i in range(nsamples):
        if r1[i] > Ar:
            b[i],p[i] = (1+pl)*(1. + (r1[i]-1.)/(1.-Ar)),\
                        (1-r2[i])*pl + r2[i]*pu
        else:
            b[i],p[i] = (1. + pl) + np.sqrt(r1[i]/Ar)*r2[i]*(pu-pl),\
                        pu + (pl-pu)*np.sqrt(r1[i]/Ar)*(1.-r2[i])
    return b,p

from scipy.stats import gamma,norm,beta,truncnorm
import numpy as np

# Prior transforms for nested samplers:
def transform_uniform(x, hyperparameters):
    a, b = hyperparameters
    return a + (b-a)*x

def transform_loguniform(x, hyperparameters):
    a, b = hyperparameters
    la = np.log(a)
    lb = np.log(b)
    return np.exp(la + x * (lb - la))

def transform_normal(x, hyperparameters):
    mu, sigma = hyperparameters
    return norm.ppf(x, loc=mu, scale=sigma)

def transform_beta(x, hyperparameters):
    a, b = hyperparameters
    return beta.ppf(x, a, b)

def transform_exponential(x, hyperparameters):
    a = hyperparameters
    return gamma.ppf(x, a)

def transform_truncated_normal(x, hyperparameters):
    mu, sigma, a, b = hyperparameters
    ar, br = (a - mu) / sigma, (b - mu) / sigma
    return truncnorm.ppf(x, ar, br, loc=mu, scale=sigma)

def transform_modifiedjeffreys(x, hyperparameters):
    turn, hi = hyperparameters
    return turn * (np.exp( (x + 1e-10) * np.log(hi/turn + 1)) - 1)

# Log-prior evaluations for MCMCs:
def evaluate_uniform(x, hyperparameters):
    a, b = hyperparameters
    if x > a and x < b:
        return np.log(1./(b - a))
    else: 
        return -np.inf

def evaluate_loguniform(x, hyperparameters):
    a, b = hyperparameters
    if x > a and x < b:
        la = np.log(a)
        lb = np.log(b)
        return np.log(1./(x * (lb - la)))
    else:
        return -np.inf

def evaluate_normal(x, hyperparameters):
    mu, sigma = hyperparameters
    return norm.logpdf(x, loc=mu, scale=sigma)

def evaluate_beta(x, hyperparameters):
    a, b = hyperparameters
    if a > 0 and b < 1:
        return beta.logpdf(x, a, b)
    else:
        return -np.inf

def evaluate_exponential(x, hyperparameters):
    a = hyperparameters
    return gamma.logpdf(x, a)

def evaluate_truncated_normal(x, hyperparameters):
    mu, sigma, a, b = hyperparameters
    if x > a and x < b:
        ar, br = (a - mu) / sigma, (b - mu) / sigma
        return truncnorm.logpdf(x, ar, br, loc=mu, scale=sigma)
    else:
        return -np.inf

def evaluate_modifiedjeffreys(x, hyperparameters):
    turn, hi = hyperparameters
    if x > 0 and x < hi:
        return np.log(1.) - np.log(x + turn) + np.log(1.) - np.log( np.log( (turn + hi) / turn ) )
    else:
        return -np.inf

def input_error_catcher(t,y,yerr,datatype):
    if datatype == 'lightcurve':
        dname = 'lc'
    else:
        dname = 'rv'
    if (y is None):
        raise Exception('INPUT ERROR: No '+datatype+' data was fed to juliet. \n'+\
                        ' Make sure to pass data (y_'+dname+') and errors (yerr_'+dname+').')
    if (yerr is None):
        raise Exception('INPUT ERROR: No ERRORS (yerr_'+dname+') on the '+datatype+' data were fed to juliet. \n'+\
                        ' Make sure to pass data (y_'+dname+') and errors (yerr_'+dname+')..')

def read_data(fname):
    fin = open(fname,'r')
    ts = np.array([])
    fs = np.array([])
    ferrs = np.array([])
    instruments = np.array([])
    # Arguments of an optional linear model. This will save the regression matrix "X" in a model of the form X*theta = y, where theta
    # are the coefficients:
    lm_arguments = {}
    # This will save a True or False for each instrument --- True if there are 
    # inputs and therefore we want a linear model, False if not:
    lm_boolean = {}
    instrument_names = []
    while True:
        line = fin.readline()
        if line != '':
            all_vals = line.split()
            t,f,ferr,instrument = all_vals[0:4]
            lm_variables = all_vals[4:]
            ts = np.append(ts,np.double(t))
            fs = np.append(fs,np.double(f))
            ferrs = np.append(ferrs,np.double(ferr))
            instruments = np.append(instruments,instrument.split()[0])
            if instrument.split()[0] not in instrument_names:
                instrument_names.append(instrument.split()[0])
                if len(lm_variables)>0:
                    lm_arguments[instrument.split()[0]] = np.array([])
                    lm_boolean[instrument.split()[0]] = True
                else:
                    lm_boolean[instrument.split()[0]] = False
            if lm_boolean[instrument.split()[0]]:
                if len(lm_arguments[instrument.split()[0]]) == 0:
                   lm_arguments[instrument.split()[0]] = np.array(lm_variables).astype(np.double)
                else:
                   lm_arguments[instrument.split()[0]] = np.vstack((lm_arguments[instrument.split()[0]],\
                                                              np.array(lm_variables).astype(np.double)))
        else:
            break
    # Identify instrument indeces:
    indexes = {}
    for instrument in instrument_names:
        indexes[instrument] = np.where(instruments == instrument)[0]
    return ts,fs,ferrs,instruments,indexes,len(instrument_names),instrument_names,lm_boolean,lm_arguments

def readGPeparams(fname):
    fin = open(fname,'r')
    GPDictionary = {}
    ftime = True
    global_model = False
    while True:
        line = fin.readline()
        if line != '':
            if line[0] != '#':
                vector = line.split()
                variables,instrument = vector[:-1],vector[-1].split()[0]
                if ftime:
                    if instrument == 'rv' or instrument == 'lc':
                        global_model = True
                    else:
                        global_model = False
                    appended_vector = np.double(np.array(variables))
                else:
                    appended_vector = np.double(np.array(variables))
                    
                if ftime:
                    ftime = False
                if instrument in GPDictionary.keys():
                    GPDictionary[instrument] = np.vstack((GPDictionary[instrument],appended_vector))
                else:
                    GPDictionary[instrument] = {}
                    GPDictionary[instrument] = appended_vector
        else:
            break
    return GPDictionary,global_model

def readpriors(priorname):
    """
    This function takes either a string or a dict and spits out information about the prior. If a string, it 
    reads a prior file. If a dict, it assumes the input dictionary has already defined all the variables and 
    distributions and simply spits out information about the system (e.g., number of transiting planets, RV 
    planets, etc.)
    """
    input_dict = False
    if type(priorname) == str:
        fin = open(priorname)
        priors = {}
        starting_point = {}
    else:
        counter = -1
        priors = priorname
        input_dict = True
        all_parameters = list(priors.keys())
        n_allkeys = len(all_parameters)
    n_transit = 0
    n_rv = 0
    n_params = 0
    numbering_transit = np.array([])
    numbering_rv = np.array([])
    while True:
        if not input_dict:
            line = fin.readline()
        else:
            # Dummy variable so we enter the while:
            line = 'nc'
            counter += 1
        if line != '': 
            if line[0] != '#':
                if not input_dict:
                    prior_vector = line.split()
                    if len(prior_vector) == 3:
                        parameter, prior_name, vals = prior_vector
                        has_sp = False
                    else:
                        parameter, prior_name, vals, sp = prior_vector
                        has_sp = True
                    parameter = parameter.split()[0]
                    # For retro-compatibility, if parameter is of the form sigma_w_rv_instrument change to
                    # sigma_w_instrument:
                    if parameter[:10] == 'sigma_w_rv':
                        instrument = parameter.split('_')[-1]
                        parameter = 'sigma_w_'+instrument
                    prior_name = prior_name.split()[0]
                    vals = vals.split()[0]
                    priors[parameter] = {}
                    if has_sp:
                        starting_point[parameter] = np.double(sp)
                else:
                    param = all_parameters[counter]
                    parameter,prior_name = param,priors[param]['distribution']
                pvector = parameter.split('_')
                # Check if parameter/planet is from a transiting planet:
                if pvector[0] == 'r1' or pvector[0] == 'p' or pvector[0] == 'phi':
                    pnumber = int(pvector[1][1:])
                    numbering_transit = np.append(numbering_transit,pnumber)
                    n_transit += 1
                # Check if parameter/planet is from a RV planet:
                if pvector[0] == 'K':
                    pnumber = int(pvector[1][1:])
                    numbering_rv = np.append(numbering_rv,pnumber)
                    n_rv += 1
                #if parameter == 'r1_p'+str(n_transit+1) or parameter == 'p_p'+str(n_transit+1):
                #    numbering_transit = np.append(numbering_transit,n_transit+1)
                #    n_transit += 1
                #if parameter == 'K_p'+str(n_rv+1):
                #    numbering_rv = np.append(numbering_rv,n_rv+1)
                #    n_rv += 1
                if prior_name.lower() == 'fixed':
                    if not input_dict:
                        priors[parameter]['distribution'] = prior_name.lower()
                        priors[parameter]['hyperparameters'] = np.double(vals)
                        priors[parameter]['cvalue'] = np.double(vals)
                else:
                    n_params += 1
                    if not input_dict:
                        priors[parameter]['distribution'] = prior_name.lower()
                        if priors[parameter]['distribution'] != 'truncatednormal':
                            v1,v2 = vals.split(',')
                            priors[parameter]['hyperparameters'] = [np.double(v1),np.double(v2)]
                        else:
                            v1,v2,v3,v4 = vals.split(',')
                            priors[parameter]['hyperparameters'] = [np.double(v1),np.double(v2),np.double(v3),np.double(v4)]
                        priors[parameter]['cvalue'] = 0.
        else:
            break
        if input_dict:
            if counter == n_allkeys-1:
                break
    if not input_dict:
        if len(starting_point.keys()) == 0:
            starting_point = None
        return priors, n_transit, n_rv, numbering_transit.astype('int'), numbering_rv.astype('int'), n_params, starting_point
    else:
        return n_transit, n_rv, numbering_transit.astype('int'), numbering_rv.astype('int'), n_params

def get_phases(t,P,t0):
    """
    Given input times, a period (or posterior dist of periods)
    and time of transit center (or posterior), returns the 
    phase at each time t.
    """
    if type(t) is not float:
        phase = ((t - np.median(t0))/np.median(P)) % 1
        ii = np.where(phase>=0.5)[0]
        phase[ii] = phase[ii]-1.0
    else:
        phase = ((t - np.median(t0))/np.median(P)) % 1
        if phase>=0.5:
            phase = phase - 1.0
    return phase

def get_quantiles(dist,alpha = 0.68, method = 'median'):
    """
    get_quantiles function
    DESCRIPTION
        This function returns, in the default case, the parameter median and the error% 
        credibility around it. This assumes you give a non-ordered 
        distribution of parameters.
    OUTPUTS
        Median of the parameter,upper credibility bound, lower credibility bound
    """
    ordered_dist = dist[np.argsort(dist)]
    param = 0.0
    # Define the number of samples from posterior
    nsamples = len(dist)
    nsamples_at_each_side = int(nsamples*(alpha/2.)+1)
    if(method == 'median'):
       med_idx = 0
       if(nsamples%2 == 0.0): # Number of points is even
          med_idx_up = int(nsamples/2.)+1
          med_idx_down = med_idx_up-1
          param = (ordered_dist[med_idx_up]+ordered_dist[med_idx_down])/2.
          return param,ordered_dist[med_idx_up+nsamples_at_each_side],\
                 ordered_dist[med_idx_down-nsamples_at_each_side]
       else:
          med_idx = int(nsamples/2.)
          param = ordered_dist[med_idx]
          return param,ordered_dist[med_idx+nsamples_at_each_side],\
                 ordered_dist[med_idx-nsamples_at_each_side]

def bin_data(x,y,n_bin):
    x_bins = []
    y_bins = []
    y_err_bins = []
    for i in range(0,len(x),n_bin):
        x_bins.append(np.median(x[i:i+n_bin-1]))
        y_bins.append(np.median(y[i:i+n_bin-1]))
        y_err_bins.append(np.sqrt(np.var(y[i:i+n_bin-1]))/np.sqrt(len(y[i:i+n_bin-1])))
    return np.array(x_bins),np.array(y_bins),np.array(y_err_bins)

def writepp(fout,posteriors, priors):
    if 'pu' in posteriors:
        pu = posteriors['pu']
        pl = posteriors['pl']
        Ar = (pu - pl)/(2. + pl + pu)

    fout.write('# {0:18} \t \t {1:12} \t \t {2:12} \t \t {3:12}\n'.format('Parameter Name','Median','Upper 68 CI','Lower 68 CI'))
    for pname in posteriors['posterior_samples'].keys():
      if pname != 'unnamed' and pname != 'loglike':
        val,valup,valdown = get_quantiles(posteriors['posterior_samples'][pname])
        usigma = valup-val
        dsigma = val - valdown
        fout.write('{0:18} \t \t {1:.10f} \t \t {2:.10f} \t \t {3:.10f}\n'.format(pname,val,usigma,dsigma))
        if pname.split('_')[0] == 'r2':
            par,planet = pname.split('_')
            r1 = posteriors['posterior_samples']['r1_'+planet]
            r2 = posteriors['posterior_samples']['r2_'+planet]
            b,p = np.zeros(len(r1)),np.zeros(len(r1))
            for i in range(len(r1)):
                if r1[i] > Ar:
                    b[i],p[i] = (1+pl)*(1. + (r1[i]-1.)/(1.-Ar)),\
                                (1-r2[i])*pl + r2[i]*pu
                else:
                    b[i],p[i] = (1. + pl) + np.sqrt(r1[i]/Ar)*r2[i]*(pu-pl),\
                                pu + (pl-pu)*np.sqrt(r1[i]/Ar)*(1.-r2[i])
            val,valup,valdown = get_quantiles(p)
            usigma = valup-val
            dsigma = val - valdown
            fout.write('{0:18} \t \t {1:.10f} \t \t {2:.10f} \t \t {3:.10f}\n'.format('p_'+planet,val,usigma,dsigma))
            val,valup,valdown = get_quantiles(b)
            usigma = valup-val
            dsigma = val - valdown
            fout.write('{0:18} \t \t {1:.10f} \t \t {2:.10f} \t \t {3:.10f}\n'.format('b_'+planet,val,usigma,dsigma))
            # Calculate inclination:
            if 'ecosomega_'+planet in posteriors['posterior_samples']:
                iplanet = planet[1:]
                ecc = np.sqrt(posteriors['posterior_samples']['ecosomega_p'+str(iplanet)]**2+posteriors['posterior_samples']['esinomega_p'+str(iplanet)]**2)
                omega = np.arctan2(posteriors['posterior_samples']['esinomega_p'+str(iplanet)],\
                               posteriors['posterior_samples']['ecosomega_p'+str(iplanet)])
            elif 'secosomega_'+planet in posteriors['posterior_samples']:
                iplanet = planet[1:]
                ecc = posteriors['posterior_samples']['secosomega_p'+str(iplanet)]**2+posteriors['posterior_samples']['sesinomega_p'+str(iplanet)]**2
                omega = np.arctan2(posteriors['posterior_samples']['sesinomega_p'+str(iplanet)],\
                                   posteriors['posterior_samples']['secosomega_p'+str(iplanet)])
            elif 'ecc_'+planet in posteriors['posterior_samples']:
                ecc = posteriors['posterior_samples']['ecc_'+planet]
                omega = posteriors['posterior_samples']['omega_'+planet]*np.pi/180.
            else:
                 ecc = 0.
                 omega = 90.
            ecc_factor = (1. + ecc*np.sin(omega))/(1. - ecc**2)
            if 'rho' in posteriors['posterior_samples']:
                G = 6.67408e-11
                if 'P_'+planet in posteriors['posterior_samples']:
                    a = ((posteriors['posterior_samples']['rho']*G*((posteriors['posterior_samples']['P_'+planet]*24.*3600.)**2))/(3.*np.pi))**(1./3.)
                else:
                    a = ((posteriors['posterior_samples']['rho']*G*((priors['P_'+planet]['hyperparameters']*24.*3600.)**2))/(3.*np.pi))**(1./3.)
            else:
                if 'a_'+planet in posteriors['posterior_samples']:
                    a = posteriors['posterior_samples']['a_'+planet]
                else:
                    a = priors['a_'+planet]['hyperparameters']
            inc_inv_factor = (b/a)*ecc_factor
            inc = np.arccos(inc_inv_factor)*180./np.pi
            val,valup,valdown = get_quantiles(inc)
            usigma = valup-val
            dsigma = val - valdown
            fout.write('{0:18} \t \t {1:.10f} \t \t {2:.10f} \t \t {3:.10f}\n'.format('inc_'+planet,val,usigma,dsigma))

        if pname.split('_')[0] == 'P':
            if 'rho' in posteriors['posterior_samples']:
                par,planet = pname.split('_')
                G = 6.67408e-11
                if 'P_'+planet in posteriors['posterior_samples']:
                    a = ((posteriors['posterior_samples']['rho']*G*((posteriors['posterior_samples']['P_'+planet]*24.*3600.)**2))/(3.*np.pi))**(1./3.)
                else:
                    a = ((posteriors['posterior_samples']['rho']*G*((priors['P_'+planet]['hyperparameters']*24.*3600.)**2))/(3.*np.pi))**(1./3.)
                val,valup,valdown = get_quantiles(a)
                usigma = valup-val
                dsigma = val - valdown
                fout.write('{0:18} \t \t {1:.10f} \t \t {2:.10f} \t \t {3:.10f}\n'.format('a_'+planet,val,usigma,dsigma))
        if pname.split('_')[0] == 'ecosomega':
            par,planet = pname.split('_')
            iplanet = planet[1:]
            ecc = np.sqrt(posteriors['posterior_samples']['ecosomega_p'+str(iplanet)]**2+posteriors['posterior_samples']['esinomega_p'+str(iplanet)]**2)
            omega = np.arctan2(posteriors['posterior_samples']['esinomega_p'+str(iplanet)],\
                               posteriors['posterior_samples']['ecosomega_p'+str(iplanet)])*(180/np.pi)

            val,valup,valdown = get_quantiles(ecc)
            usigma = valup-val
            dsigma = val - valdown
            fout.write('{0:18} \t \t {1:.10f} \t \t {2:.10f} \t \t {3:.10f}\n'.format('ecc_'+planet,val,usigma,dsigma))

            idx = np.where(omega>0.)[0]
            val,valup,valdown = get_quantiles(omega[idx])
            usigma = valup-val
            dsigma = val - valdown
            fout.write('{0:18} \t \t {1:.10f} \t \t {2:.10f} \t \t {3:.10f}\n'.format('omega_'+planet,val,usigma,dsigma))

        if pname.split('_')[0] == 'secosomega':
            par,planet = pname.split('_')
            iplanet = planet[1:]
            ecc = posteriors['posterior_samples']['secosomega_p'+str(iplanet)]**2+posteriors['posterior_samples']['sesinomega_p'+str(iplanet)]**2
            omega = np.arctan2(posteriors['posterior_samples']['sesinomega_p'+str(iplanet)],\
                               posteriors['posterior_samples']['secosomega_p'+str(iplanet)])*(180/np.pi)

            val,valup,valdown = get_quantiles(ecc)
            usigma = valup-val
            dsigma = val - valdown
            fout.write('{0:18} \t \t {1:.10f} \t \t {2:.10f} \t \t {3:.10f}\n'.format('ecc_'+planet,val,usigma,dsigma))

            idx = np.where(omega>0.)[0]
            val,valup,valdown = get_quantiles(omega[idx])
            usigma = valup-val
            dsigma = val - valdown
            fout.write('{0:18} \t \t {1:.10f} \t \t {2:.10f} \t \t {3:.10f}\n'.format('omega_'+planet,val,usigma,dsigma))
    fout.close()

from astropy.time import Time as APYTime
def convert_time(conv_string,t):
    input_t,output_t = conv_string.split('->')
    if input_t != output_t:
        tobj = APYTime(t, format = 'jd', scale = input_t)
        # print('new_t = tobj.'+output_t+'.jd')
        # exec('new_t = tobj.'+output_t+'.jd')
        if output_t == 'utc':
            return tobj.utc.jd
        else:
            return t
        # return new_t
    else:
        return t

def generate_priors(params,dists,hyperps):
    priors = {}
    for param, dist, hyperp in zip(params, dists, hyperps):
        priors[param] = {}
        priors[param]['distribution'], priors[param]['hyperparameters'] = dist, hyperp
    return priors

def read_AIJ_tbl(fname):
    """
    This function reads in an AstroAIJ table and returns a dictionary with all the parameters in numpy arrays.
    """
    fin = open(fname,'r')
    firstime = True
    out_dict = {}
    while True:
        line = fin.readline()
        if line != '':
            vec = line.split()
            if firstime:
                out_dict['index'] = np.array([])
                for i in range(len(vec)):
                    out_dict[vec[i]] = np.array([])
                firstime = False
                parameter_vector = ['index'] + vec
            else:
                for i in range(len(vec)):
                    try:
                        out_dict[parameter_vector[i]] = np.append(out_dict[parameter_vector[i]],np.double(vec[i]))
                    except:
                        out_dict[parameter_vector[i]] = np.append(out_dict[parameter_vector[i]],np.nan)
        else:
            break
    return out_dict
