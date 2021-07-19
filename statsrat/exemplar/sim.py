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