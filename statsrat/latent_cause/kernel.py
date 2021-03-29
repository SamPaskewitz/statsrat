import numpy as np

def constant(t, time, sim_pars):
    '''
    The temporal kernel is simply 1 for all time steps (i.e. there is no decay).
    This reproduces the original CRP.
    '''
    return np.ones(t)
constant.par_names = []

def exponential(t, time, sim_pars):
    '''Exponential decay.'''
    return np.exp(-sim_pars['gamma']*(time[t] - time[0:t]))
exponential.par_names = ['gamma']

def power(t, time, sim_pars):
    '''Power law decay (with a constant power of 1).'''
    K = 1/(time[t] - time[0:t])
    return K
power.par_names = []