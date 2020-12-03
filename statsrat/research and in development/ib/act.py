import numpy as np

def normalize(rtrv, sim_pars):
    '''Activation is simply normalized retrieval strength.'''
    act = rtrv/np.sum(rtrv)
    return act

def k_max(rtrv, sim_pars):
    '''The k memory traces with the largest retrieval strengths get an
    activation of 1.  All other activations are 0.'''
    n_trace = len(rtrv)
    act = np.zeros(n_trace)
    ind = np.argsort(rtrv)[-sim_pars['k']:] # indices of the k largest retrieval strengths
    act[ind] = 1/pars.k
    return act
    