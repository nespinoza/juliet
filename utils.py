from scipy.stats import gamma,norm,beta,truncnorm
import numpy as np

def transform_uniform(x,a,b):
    return a + (b-a)*x

def transform_loguniform(x,a,b):
    la=np.log(a)
    lb=np.log(b)
    return np.exp(la + x*(lb-la))

def transform_normal(x,mu,sigma):
    return norm.ppf(x,loc=mu,scale=sigma)

def transform_beta(x,a,b):
    return beta.ppf(x,a,b)

def transform_exponential(x,a=1.):
    return gamma.ppf(x, a)

def transform_truncated_normal(x,mu,sigma,a=0.,b=1.):
    ar, br = (a - mu) / sigma, (b - mu) / sigma
    return truncnorm.ppf(x,ar,br,loc=mu,scale=sigma)

def readlc(fname):
    fin = open(fname,'r')
    ts = np.array([])
    fs = np.array([])
    ferrs = np.array([])
    instruments = np.array([])
    instrument_names = []
    while True:
        line = fin.readline()
        if line != '':
            t,f,ferr,instrument = line.split()[0:4]
            ts = np.append(ts,np.double(t))
            fs = np.append(fs,np.double(f))
            ferrs = np.append(ferrs,np.double(ferr))
            instruments = np.append(instruments,instrument.split()[0])
            if instrument.split()[0] not in instrument_names:
                instrument_names.append(instrument.split()[0])
        else:
            break
    # Identify instrument indeces:
    indexes = {}
    for instrument in instrument_names:
        indexes[instrument] = np.where(instruments == instrument)[0]
    return ts,fs,ferrs,instruments,indexes,len(instrument_names),instrument_names

def readeparams(fname,RV=False):
    fin = open(fname,'r')
    GPDictionary = {}
    ftime = True
    while True:
        line = fin.readline()
        if line != '': 
            if line[0] != '#':
                vector = line.split()
                if RV:
                    variables = vector
                    if ftime:
                        GPDictionary['variables'] = np.double(np.array(variables))
                        ftime = False 
                    else:
                        GPDictionary['variables'] = np.vstack((GPDictionary['variables'],np.double(np.array(variables))))
                else:
                    variables,instrument = vector[:-1],vector[-1].split()[0]
                    if instrument in GPDictionary.keys():
                        GPDictionary[instrument]['variables'] = np.vstack((GPDictionary[instrument]['variables'],np.double(np.array(variables))))
                    else:
                        GPDictionary[instrument] = {}
                        GPDictionary[instrument]['variables'] = np.double(np.array(variables))
        else:
            break
    return GPDictionary

def readpriors(priorname):
    fin = open(priorname)
    priors = {}
    n_transit = 0
    n_rv = 0
    n_params = 0
    numbering_transit = np.array([])
    numbering_rv = np.array([])
    while True:
        line = fin.readline()
        if line != '': 
            if line[0] != '#':
                out = line.split()
                parameter,prior_name,vals = line.split()
                parameter = parameter.split()[0]
                prior_name = prior_name.split()[0]
                vals = vals.split()[0]
                priors[parameter] = {}
                pvector = parameter.split('_')
                # Check if parameter/planet is from a transiting planet:
                if pvector[0] == 'r1' or pvector[0] == 'p':
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
                    priors[parameter]['type'] = prior_name.lower()
                    priors[parameter]['value'] = np.double(vals)
                    priors[parameter]['cvalue'] = np.double(vals)
                else:
                    n_params += 1
                    priors[parameter]['type'] = prior_name.lower()
                    v1,v2 = vals.split(',')
                    priors[parameter]['value'] = [np.double(v1),np.double(v2)]
                    priors[parameter]['cvalue'] = 0.
        else:
            break
    return priors,n_transit,n_rv,numbering_transit.astype('int'),numbering_rv.astype('int'),n_params

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


