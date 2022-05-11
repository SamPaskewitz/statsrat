import numpy as np

def cnst(aux, t, delta, fbase, fweight, n_f, n_y, sim_pars):
    '''Constant learning rate for non-zero features.'''
    new_lrate = np.array(n_y*[fbase[t, :].tolist()]).transpose()*sim_pars['lrate']
    return new_lrate
cnst.par_names = ['lrate']

def pos_neg(aux, t, delta, fbase, fweight, n_f, n_y, sim_pars):
    '''Separate learning rates for positive and negative prediction error.'''
    is_pos = delta > 0
    lrate_par_to_use = is_pos*sim_pars['lrate_pos'] + (1 - is_pos)*sim_pars['lrate_neg']
    new_lrate = np.array(n_y*[fbase[t, :].tolist()]).transpose()*lrate_par_to_use
    return new_lrate
pos_neg.par_names = ['lrate_pos', 'lrate_neg']

def power(aux, t, delta, fbase, fweight, n_f, n_y, sim_pars):
    '''
    Power function learning rate for non-zero features.
    Learning rates decay with the number of times each feature has been observed.
    '''
    denom = (aux.data['f_counts'][t, :] + 1)**sim_pars['power']
    new_lrate = 0.5/denom + sim_pars['lrate_min']
    new_lrate = new_lrate.reshape((n_f, 1))*np.array(n_y*[fbase[t, :].tolist()]).transpose() # learning rate depends on feature (row), but not outcome (column)
    return new_lrate
power.par_names = ['power', 'lrate_min']

def from_aux_norm2(aux, t, delta, fbase, fweight, n_f, n_y, sim_pars):
    '''
    Produce weights that normalize features (e.g. CompAct) according to a Euclidean metric.
    This is a version of 'from_aux_norm' with the metric fixed at 2.
    So long as the base features are 0 or 1, this is equivalent to EXIT-style feature weighting.
    '''
    atn_gain = aux.data['atn'][t, :]*fbase[t, :]
    atn_gain[atn_gain < 0.01] = 0.01 # Added this in to make this consistent with the R code.
    norm = sum(atn_gain**2)**(1/2)
    norm_atn = atn_gain/norm
    new_lrate = sim_pars['lrate']*norm_atn.reshape((n_f, 1))*np.array(n_y*[fbase[t, :].tolist()]).transpose() # learning rate depends on feature (row), but not outcome (column)
    return new_lrate
from_aux_norm2.par_names = ['lrate']

def from_aux_norm(aux, t, delta, fbase, fweight, n_f, n_y, sim_pars):
    '''
    Produce weights that normalize features (e.g. CompAct).
    So long as the base features are 0 or 1, this is equivalent to EXIT-style feature weighting.
    '''
    atn_gain = aux.data['atn'][t, :]*fbase[t, :]
    atn_gain[atn_gain < 0.01] = 0.01 # Added this in to make this consistent with the R code.
    norm = sum(atn_gain**sim_pars['metric'])**(1/sim_pars['metric'])
    norm_atn = atn_gain/norm
    new_lrate = sim_pars['lrate']*norm_atn.reshape((n_f, 1))*np.array(n_y*[fbase[t, :].tolist()]).transpose() # learning rate depends on feature (row), but not outcome (column)
    return new_lrate
from_aux_norm.par_names = ['lrate', 'metric']

def power_from_aux_norm(aux, t, delta, fbase, fweight, n_f, n_y, sim_pars):
    '''
    Combines the from_aux_norm (CompAct style) learning rate with power law learning
    rates (learning rates decrease each time a feature is observed).  Both styles of
    learning rate are multiplied together.
    '''
    # from_aux_norm part of learning rate
    atn_gain = aux.data['atn'][t, :]*fbase[t, :]
    atn_gain[atn_gain < 0.01] = 0.01 # Added this in to make this consistent with the R code.
    norm = sum(atn_gain**sim_pars['metric'])**(1/sim_pars['metric'])
    norm_atn = atn_gain/norm
    from_aux_norm_lrate = norm_atn.reshape((n_f, 1))*np.array(n_y*[fbase[t, :].tolist()]).transpose()
    # power law part of learning rate
    denom = (aux.data['f_counts'][t, :] + 1)**sim_pars['power']
    power_lrate = 0.5/denom + sim_pars['lrate_min']
    power_lrate = power_lrate.reshape((n_f, 1))*np.array(n_y*[fbase[t, :].tolist()]).transpose()
    return from_aux_norm_lrate*power_lrate
power_from_aux_norm.par_names = ['power', 'lrate_min', 'metric']

def from_aux_feature(aux, t, delta, fbase, fweight, n_f, n_y, sim_pars):
    '''
    Learning rate determined by 'aux' (variable name 'atn') and the 'lrate' parameter.
    Depends only on feature.
    '''
    atn = aux.data['atn'][t, :] # current attention ('atn')
    new_lrate = sim_pars['lrate']*atn.reshape((n_f, 1))*np.array(n_y*[fbase[t, :].tolist()]).transpose() # learning rate depends on feature (row), but not outcome (column)
    return new_lrate
from_aux_feature.par_names = ['lrate']

def from_aux_feature_simple(aux, t, delta, fbase, fweight, n_f, n_y, sim_pars):
    '''
    Learning rate determined by 'aux' (variable name 'atn'), with no 'lrate' parameter.
    Depends only on feature.
    '''
    atn = aux.data['atn'][t, :] # current attention ('atn')
    new_lrate = atn.reshape((n_f, 1))*np.array(n_y*[fbase[t, :].tolist()]).transpose() # learning rate depends on feature (row), but not outcome (column)
    return new_lrate
from_aux_feature_simple.par_names = []

def from_aux_direct(aux, t, delta, fbase, fweight, n_f, n_y, sim_pars):
    '''
    Learning rate taken directly from 'aux' (variable name 'gain').
    Does not depend on any 'lrate' model parameter.
    Depends on both features and outcomes.
    '''
    return aux.data['gain'][t, :, :]
from_aux_direct.par_names = []
    