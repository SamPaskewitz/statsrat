import numpy as np
import pandas as pd

def cnst(state, n, env, sim_pars):
    '''Constant learning rate for non-zero features.'''
    new_lrate = np.tile(state['fbase'].reshape([n['f'], 1]), n['y'])*sim_pars['lrate']
    return new_lrate
cnst.pars = pd.DataFrame({'min': 0.0, 'max': 1.0, 'default': 0.2}, index = ['lrate'])

def pos_neg(state, n, env, sim_pars):
    '''Separate learning rates for positive and negative prediction error.'''
    is_pos = delta > 0
    lrate_par_to_use = is_pos*sim_pars['lrate_pos'] + (1 - is_pos)*sim_pars['lrate_neg']
    new_lrate = np.array(n['y']*[state['fbase'].tolist()]).transpose()*lrate_par_to_use
    return new_lrate
pos_neg.pars = pd.DataFrame([{'min': 0.0, 'max': 1.0, 'default': 0.2}, {'min': 0.0, 'max': 1.0, 'default': 0.2}], index = ['lrate_pos', 'lrate_neg'])

def power(state, n, env, sim_pars):
    '''
    Power function learning rate for non-zero features.
    Learning rates decay with the number of times each feature has been observed.
    
    Notes
    -----
    The learning rate for a stimulus feature feature is:
    $$\\lambda_i = f_i(\frac{1}{2 (N_i + 1)^p} + \\lambda_{\\min})$$
    
    Where
    $N_i$ = the number of times the learner has observed that feature
    $p$ = the ``power'' parameter
    $\\lambda_{\\min}$ = the ``lrate_min'' parameter
    $f_i$ = the feature value
    '''
    denom = (state['f_counts'] + 1)**sim_pars['power']
    new_lrate = 0.5/denom + sim_pars['lrate_min']
    new_lrate = new_lrate.reshape((n['f'], 1))*np.array(n['y']*[state['fbase'].tolist()]).transpose() # learning rate depends on feature (row), but not outcome (column)
    return new_lrate
power.pars = pd.DataFrame([{'min': 0.0, 'max': 2.0, 'default': 0.5}, {'min': 0.0, 'max': 0.5, 'default': 0.1}], index = ['power', 'lrate_min'])

def from_aux_norm2(state, n, env, sim_pars):
    '''
    Produce weights that normalize features (e.g. CompAct) according to a Euclidean metric.
    This is a version of 'from_aux_norm' with the metric fixed at 2.
    So long as the base features are 0 or 1, this is equivalent to EXIT-style feature weighting.
    '''
    atn_gain = state['atn']*state['fbase']
    atn_gain[atn_gain < 0.01] = 0.01 # Added this in to make this consistent with the R code.
    norm = sum(atn_gain**2)**(1/2)
    norm_atn = atn_gain/norm
    new_lrate = sim_pars['lrate']*norm_atn.reshape((n['f'], 1))*np.array(n['y']*[state['fbase'].tolist()]).transpose() # learning rate depends on feature (row), but not outcome (column)
    return new_lrate
from_aux_norm2.pars = pd.DataFrame({'min': 0.0, 'max': 1.0, 'default': 0.2}, index = ['lrate'])

def from_aux_norm(state, n, env, sim_pars):
    '''
    Produce weights that normalize features (e.g. CompAct).
    So long as the base features are 0 or 1, this is equivalent to EXIT-style feature weighting.
    '''
    atn_gain = state['atn']*state['fbase']
    atn_gain[atn_gain < 0.01] = 0.01 # Added this in to make this consistent with the R code.
    norm = sum(atn_gain**sim_pars['metric'])**(1/sim_pars['metric'])
    norm_atn = atn_gain/norm
    new_lrate = sim_pars['lrate']*norm_atn.reshape((n['f'], 1))*np.array(n['y']*[state['fbase'].tolist()]).transpose() # learning rate depends on feature (row), but not outcome (column)
    return new_lrate
from_aux_norm.pars = pd.DataFrame([{'min': 0.0, 'max': 1.0, 'default': 0.2}, {'min': 0.1, 'max': 10, 'default': 2}], index = ['lrate', 'metric'])

def power_from_aux_norm(state, n, env, sim_pars):
    '''
    Combines the from_aux_norm (CompAct style) learning rate with power law learning
    rates (learning rates decrease each time a feature is observed).  Both styles of
    learning rate are multiplied together.
    '''
    # from_aux_norm part of learning rate
    atn_gain = state['atn']*state['fbase']
    atn_gain[atn_gain < 0.01] = 0.01 # Added this in to make this consistent with the R code.
    norm = sum(atn_gain**sim_pars['metric'])**(1/sim_pars['metric'])
    norm_atn = atn_gain/norm
    from_aux_norm_lrate = norm_atn.reshape((n['f'], 1))*np.array(n['y']*[state['fbase'].tolist()]).transpose()
    # power law part of learning rate
    denom = (aux.data['f_counts'][t, :] + 1)**sim_pars['power']
    power_lrate = 0.5/denom + sim_pars['lrate_min']
    power_lrate = power_lrate.reshape((n['f'], 1))*np.array(n['y']*[state['fbase'].tolist()]).transpose()
    return from_aux_norm_lrate*power_lrate
power_from_aux_norm.pars = pd.DataFrame([{'min': 0.0, 'max': 2.0, 'default': 0.5}, {'min': 0.0, 'max': 0.5, 'default': 0.1}, {'min': 0.1, 'max': 10, 'default': 2}], index = ['power', 'lrate_min', 'metric'])

def from_aux_feature(state, n, env, sim_pars):
    '''
    Learning rate determined by 'aux' (variable name 'atn') and the 'lrate' parameter.
    Depends only on feature.
    '''
    atn = state['atn'] # current attention ('atn')
    new_lrate = sim_pars['lrate']*atn.reshape((n['f'], 1))*np.array(n['y']*[state['fbase'].tolist()]).transpose() # learning rate depends on feature (row), but not outcome (column)
    return new_lrate
from_aux_feature.pars = pd.DataFrame({'min': 0.0, 'max': 1.0, 'default': 0.2}, index = ['lrate'])

def from_aux_feature_simple(state, n, env, sim_pars):
    '''
    Learning rate determined by 'aux' (variable name 'atn'), with no 'lrate' parameter.
    Depends only on feature.
    '''
    atn = state['atn'] # current attention ('atn')
    new_lrate = atn.reshape((n['f'], 1))*np.array(n['y']*[state['fbase'].tolist()]).transpose() # learning rate depends on feature (row), but not outcome (column)
    return new_lrate
from_aux_feature_simple.pars = None

def from_aux_direct(state, n, env, sim_pars):
    '''
    Learning rate taken directly from 'aux' (variable name 'gain').
    Does not depend on any 'lrate' model parameter.
    Depends on both features and outcomes.
    '''
    return aux.data['gain'][t, :, :]
from_aux_direct.pars = None   