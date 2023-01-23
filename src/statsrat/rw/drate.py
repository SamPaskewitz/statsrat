import numpy as np
import pandas as pd

'''
Functions for defining decay rates.

zero: Decay rate of zero (i.e. no weight decay).

cnst: Constant decay rate.

only_neg: Decay rate is constant for negative weights, and zero for positive weights.

hrmn: Harmonic decay rate.
'''

def zero(state, n, env, sim_pars):
    '''Decay rate of zero (i.e. no weight decay).'''
    new_drate = np.zeros((n['f'], n['y']))
    return new_drate
zero.pars = None

def cnst(state, n, env, sim_pars):
    '''Constant decay rate.'''
    new_drate = np.ones((n['f'], n['y']))*sim_pars['drate']
    return new_drate
cnst.par_names = ['drate']
cnst.pars = pd.DataFrame({'min': 0.0, 'max': 0.5, 'default': 0.25}, index = ['drate'])

def only_neg(state, n, env, sim_pars):
    '''Decay rate is constant for negative weights, and zero for positive weights.'''
    is_non_neg = state['w'] >= 0
    new_drate = np.ones((n['f'], n['y']))*sim_pars['drate']
    new_drate[is_non_neg] = 0
    return new_drate
only_neg.pars = pd.DataFrame({'min': 0.0, 'max': 0.5, 'default': 0.25}, index = ['drate'])

def hrmn(state, n, env, sim_pars):
    '''Harmonic decay rate.'''
    denom = sim_pars['extra_counts'] + aux.data['f_counts'][t, :]
    new_lrate = 1/denom # decay rates harmonically decay with the number of times each feature has been observed
    abv_min = new_lrate > 0.01
    new_lrate = new_lrate*abv_min + 0.01*(1 - abv_min)
    new_lrate = new_lrate.reshape((n['f'], 1))*np.ones((n['y'], 1))
    return new_lrate
hrmn.pars = pd.DataFrame({'min': 0.0, 'max': 10.0, 'default': 2.0}, index = ['extra_counts'])