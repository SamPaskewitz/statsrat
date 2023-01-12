import numpy as np
import pandas as pd

def equal_to_sim(state, n, env, sim_pars):
    '''Retrieval strength is equal to similarity.'''
    return state['ex_seen_yet']*state['sim']
equal_to_sim.pars = None

def normalized_sim(state, n, env, sim_pars):
    '''Retrieval strength is normalized similarity.'''
    numerator = state['ex_seen_yet']*state['sim']
    return numerator/numerator.sum()
normalized_sim.pars = None

def normalized_sim_ex_counts(state, n, env, sim_pars):
    '''
    Retrieval strength is similarity times exemplar counts, normalized.
    
    Notes
    -----
    Combined with the right u_ex_update function (viz. ex_mean), this
    reproduces instance based learning with a normalized similarity retrieval
    function across instances.
    
    One can add a fixed initial value ('nu') to ex_counts.
    '''
    numerator = state['ex_seen_yet']*state['sim']*(state['ex_counts'] + sim_pars['nu'])
    return numerator/numerator.sum()
normalized_sim_ex_counts.pars = pd.DataFrame({'min': 0.0, 'max': 10.0, 'default': 0.0}, index = ['nu']) # extra counts for ex_mean