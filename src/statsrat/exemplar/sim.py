import numpy as np

'''
Functions to define exemplar similarity.

Gaussian: Gaussian similarity function.
'''

def Gaussian(x, x_ex, atn, sim_pars):
    '''
    Gaussian similarity function.
    '''
    squared_metric = np.sum(atn*(x - x_ex)**2, axis = 1)
    return np.exp(-sim_pars['decay_rate']*squared_metric)
Gaussian.par_names = ['decay_rate']

def city_block(x, x_ex, atn, sim_pars):
    '''
    Similarity based on city block (L1) distance.
    '''
    l1_distance = np.sum(atn*np.abs(x - x_ex), axis = 1)
    return np.exp(-sim_pars['decay_rate']*l1_distance)
city_block.par_names = ['decay_rate']