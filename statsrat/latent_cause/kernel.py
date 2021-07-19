import numpy as np

'''
Temporal kernels for the distance dependent CRP (Blei & Frazier, 2011; Zhu, Ghahramani & Lafferty).
'''

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
    '''Power law decay.'''
    K = (time[t] - time[0:t])**(-sim_pars['power'])
    return K
power.par_names = ['power']