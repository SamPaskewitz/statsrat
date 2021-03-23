import numpy as np

def constant(n_t, sim_pars):
    '''
    The temporal kernel is simply 1 for all time steps (i.e. there is no decay).
    This reproduces the original CRP.
    '''
    return np.ones(n_t)
constant.par_names = []

def exponential(n_t, sim_pars):
    '''Exponential decay.'''
    return np.exp(-sim_pars['gamma']*np.arange(n_t))
exponential.par_names = ['gamma']

def power(n_t, sim_pars):
    '''Power law decay (with a constant power of 1).'''
    K = np.ones(n_t)
    K[1:n_t] = 1/np.arange(1, n_t)
    return K
power.par_names = []