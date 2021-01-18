import numpy as np

########## LEARNING RATE FUNCTIONS ##########

def cnst(aux, t, fbase, fweight, n_f, n_u, sim_pars):
    '''Constant learning rate for non-zero features.'''
    new_lrate = np.array(n_u*[fbase[t, :].tolist()]).transpose()*sim_pars['lrate']
    return new_lrate
cnst.par_names = ['lrate']

def hrmn(aux, t, fbase, fweight, n_f, n_u, sim_pars):
    '''Harmonic learning rate for non-zero features.'''
    denom = sim_pars['extra_counts'] + aux.data['f_counts'][t, :]
    new_lrate = 1/denom # learning rates harmonically decay with the number of times each feature has been observed
    abv_min = new_lrate > 0.01
    new_lrate = new_lrate*abv_min + 0.01*(1 - abv_min)
    new_lrate = new_lrate.reshape((n_f, 1))*np.array(n_u*[fbase[t, :].tolist()]).transpose() # learning rate depends on feature (row), but not outcome (column)
    return new_lrate
hrmn.par_names = ['extra_counts']

def from_aux_norm(aux, t, fbase, fweight, n_f, n_u, sim_pars):
    '''
    Produce weights that normalize features (e.g. CompAct).
    So long as the base features are 0 or 1, this is equivalent to EXIT-style feature weighting.
    '''
    atn_gain = aux.data['atn'][t, :]*fbase[t, :]
    atn_gain[atn_gain < 0.01] = 0.01 # Added this in to make this consistent with the R code.
    norm = sum(atn_gain**sim_pars['metric'])**(1/sim_pars['metric'])
    norm_atn = atn_gain/norm
    new_lrate = sim_pars['lrate']*norm_atn.reshape((n_f, 1))*np.array(n_u*[fbase[t, :].tolist()]).transpose() # learning rate depends on feature (row), but not outcome (column)
    return new_lrate
from_aux_norm.par_names = ['lrate', 'metric']

def from_aux_feature(aux, t, fbase, fweight, n_f, n_u, sim_pars):
    '''
    Learning rate determined by 'aux' (variable name 'atn') and the 'lrate' parameter.
    Depends only on feature.
    '''
    atn = aux.data['atn'][t, :] # current attention ('atn')
    abv_min = atn > 0.01
    blw_max = atn < 0.99
    atn_bounded = atn*abv_min*blw_max + 0.01*(1 - abv_min) + 0.99*(1 - blw_max)
    new_lrate = sim_pars['lrate']*atn_bounded.reshape((n_f, 1))*np.array(n_u*[fbase[t, :].tolist()]).transpose() # learning rate depends on feature (row), but not outcome (column)
    return new_lrate
from_aux_feature.par_names = ['lrate']

def from_aux_direct(aux, t, fbase, fweight, n_f, n_u, sim_pars):
    '''
    Learning rate taken directly from 'aux' (variable name 'gain').
    Does not depend on any 'lrate' model parameter.
    Depends on both features and outcomes.
    '''
    atn = aux.data['gain'][t, :, :]
    abv_min = atn > 0.01
    blw_max = atn < 0.99
    new_lrate = atn*abv_min*blw_max + 0.01*(1 - abv_min) + 0.99*(1 - blw_max)
    return new_lrate
from_aux_direct.par_names = []
    