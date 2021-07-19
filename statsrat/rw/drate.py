import numpy as np

'''
Functions for defining decay rates.

zero: Decay rate of zero (i.e. no weight decay).

cnst: Constant decay rate.

only_neg: Decay rate is constant for negative weights, and zero for positive weights.

hrmn: Harmonic decay rate.
'''

def zero(aux, t, w, n_f, n_u, sim_pars):
    '''Decay rate of zero (i.e. no weight decay).'''
    new_drate = np.zeros((n_f, n_u))
    return new_drate
zero.par_names = []

def cnst(aux, t, w, n_f, n_u, sim_pars):
    '''Constant decay rate.'''
    new_drate = np.ones((n_f, n_u))*sim_pars['drate']
    return new_drate
cnst.par_names = ['drate']

def only_neg(aux, t, w, n_f, n_u, sim_pars):
    '''Decay rate is constant for negative weights, and zero for positive weights.'''
    is_non_neg = w[t, :, :] >= 0
    new_drate = np.ones((n_f, n_u))*sim_pars['drate']
    new_drate[is_non_neg] = 0
    return new_drate
only_neg.par_names = ['drate']

def hrmn(aux, t, w, n_f, n_u, sim_pars):
    '''Harmonic decay rate.'''
    denom = sim_pars['extra_counts'] + aux.data['f_counts'][t, :]
    new_lrate = 1/denom # decay rates harmonically decay with the number of times each feature has been observed
    abv_min = new_lrate > 0.01
    new_lrate = new_lrate*abv_min + 0.01*(1 - abv_min)
    new_lrate = new_lrate.reshape((n_f, 1))*np.ones((n_u, 1))
    return new_lrate
hrmn.par_names = ['extra_counts']