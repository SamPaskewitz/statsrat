import numpy as np
import pandas as pd

# Functions to define exemplar similarity.

def Gaussian(state, n, env, sim_pars):
    '''
    Gaussian similarity function.
    '''
    squared_metric = np.sum(state['atn']*(env['x'] - state['x_ex'])**2, axis = 1)
    return np.exp(-sim_pars['decay_rate']*squared_metric)
Gaussian.pars = pd.DataFrame({'min': 0.0, 'max': 10.0, 'default': 0.5}, index = ['decay_rate'])

def city_block(state, n, env, sim_pars):
    '''
    Similarity based on city block (L1) distance.
    '''
    l1_distance = np.sum(state['atn']*np.abs(env['x'] - state['x_ex']), axis = 1)
    return np.exp(-sim_pars['decay_rate']*l1_distance)
city_block.pars = pd.DataFrame({'min': 0.0, 'max': 10.0, 'default': 0.5}, index = ['decay_rate'])