import numpy as np

def each(x, x_mem, delta, aux, t, sim_pars):
    '''Encode all memory traces.'''
    return 1

def unique(x, x_mem, delta, aux, t, sim_pars):
    '''Only encode memory traces when the exemplar hasn't been previously encountered.'''
    dif = np.sum(x - x_memory, axis = 1)
    if 0 in dif:
        encd = 0 # exemplar has been previously encountered
    else:
        encd = 1 # exemplar hasn't been previously encountered
    return encd

def wrong(x, x_mem, delta, aux, t, sim_pars):
    '''Only encode memory traces when the largest prediction error exceeds a threshold.'''
    error_size = np.abs(delta[t, :])
    if any(error_size <= sim_pars['threshold']):
        encd = 0
    else:
        encd = 1
    return encd