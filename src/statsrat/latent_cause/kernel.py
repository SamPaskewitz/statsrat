import numpy as np
import pandas as pd

'''
Temporal kernels for the distance dependent CRP (Blei & Frazier, 2011; Zhu, Ghahramani & Lafferty).
'''

def constant(t, N, time, sim_pars):
    '''
    The temporal kernel is simply 1 for all time steps (i.e. there is no decay).
    This reproduces the original CRP.
    '''
    return np.ones((t, N))
constant.pars = None

def exponential(t, N, time, sim_pars):
    '''Exponential decay.'''
    K = np.zeros((t, N))
    for i in range(N):
        K[:, i] = np.exp(-sim_pars['gamma']*(time[t] - time[0:t]))
    return K
exponential.pars = pd.DataFrame({'min': 0.0, 'max': 5.0, 'default': 1.0, 'description': 'decay rate for exponential SCRP; higher -> favors more recent latent causes'}, index = ['gamma'])

def power(t, N, time, sim_pars):
    '''Power law decay.'''
    K = np.zeros((t, N))
    for i in range(N):
        K[:, i] = (time[t] - time[0:t])**(-sim_pars['power'])
    return K
power.pars = pd.DataFrame({'min': 0.0, 'max': 5.0, 'default': 1.0, 'description': 'decay rate for power law SCRP; higher -> favors more recent latent causes'}, index = ['power'])

def power_asymptote(t, N, time, sim_pars):
    '''Power law decay with an asymptote greater than zero.'''
    K = np.zeros((t, N))
    for i in range(N):
        K[:, i] = (time[t] - time[0:t])**(-sim_pars['power']) + sim_pars['kernel_asymptote']
    return K
power_asymptote.pars = pd.DataFrame([{'min': 0.0, 'max': 5.0, 'default': 1.0, 'description': 'decay rate for power law SCRP; higher -> favors more recent latent causes'}, {'min': 0.0, 'max': 2.0, 'default': 0.5, 'description': 'asymptote for kernel'}], index = ['power', 'kernel_asymptote'])

def power_clusters(t, N, time, sim_pars):
    '''
    Power law decay with a different decay rate for each cluster (latent cause).
    The decay rate increases for each new cluster, making clusters discovered
    earlier more probable as time goes on.
    '''
    K = np.zeros((t, N))
    for i in range(N):
        decay = sim_pars['power']*(i + 1)
        K[:, i] = (time[t] - time[0:t])**(-decay)
    return K
power_clusters.pars = pd.DataFrame({'min': 0.0, 'max': 5.0, 'default': 1.0, 'description': 'decay rate for power law SCRP; higher -> favors more recent latent causes'}, index = ['power'])

def refractory_period(t, N, time, sim_pars):
    '''
    The kernel rapidly decreases, then gradual increases again.
    This produces a refractory period after a latent cause has been active
    that makes it temporarily less likely to be active again.
    
    In particular, the power law decay rate is non-zero for a number
    of trials defined by the 'window' parameter, then decreases to 0.
    '''
    K = np.zeros((t, N))
    decay = np.zeros(t)
    decay[time[0:t] <= sim_pars['window']] = sim_pars['power']
    for i in range(N):
        K[:, i] = (time[t] - time[0:t])**(-decay)
    return K
refractory_period.pars = pd.DataFrame([{'min': 0.0, 'max': 5.0, 'default': 1.0, 'description': 'decay rate for power law SCRP; higher -> favors more recent latent causes'}, {'min': 0.0, 'max': 1000.0, 'default': 100.0, 'description': 'window determining refractory period for kernel'}], index = ['power', 'window'])