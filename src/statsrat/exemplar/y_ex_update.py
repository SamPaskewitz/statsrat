import numpy as np
import pandas as pd

# Functions to specify how y_ex (exemplar -> outcome associations) are updated.

def from_rtrv(state, n, env, sim_pars):
    """
    Learning rates for exemplars are equal to retrieval strength
    times a constant (lrate_par).
    
    Notes
    -----
    This (minus the 'humble teachers',
    and with retrieval strength equal to similarity) is the form of 
    learning used in ALCOVE (Kruschke, 1992).
    """
    lrate = sim_pars['lrate_par']*state['rtrv'] # learning rates for exemplars
    delta = env['y'] - state['y_hat'] # prediction error (total)
    return np.outer(lrate, env['y_lrn']*delta)
from_rtrv.pars = pd.DataFrame({'min': 0.0, 'max': 1.0, 'default': 0.5}, index = ['lrate_par'])

def from_rtrv_indv_delta(state, n, env, sim_pars):
    """
    Learning rates for exemplars are equal to retrieval strength
    times a constant (lrate_par), and prediction errors are for each
    individual exemplar rather than for common.
    """
    lrate = sim_pars['lrate_par']*state['rtrv'] # learning rates for exemplars
    update = np.zeros((n['ex'], n['y']))
    for i in range(n['ex']):
        delta = env['y'] - state['y_ex'][i] # prediction error (only for exemplar i)
        update[i, :] = lrate[i]*delta
    return update
from_rtrv_indv_delta.pars = pd.DataFrame({'min': 0.0, 'max': 1.0, 'default': 0.5}, index = ['lrate_par'])

def only_max(state, n, env, sim_pars):
    """
    Only the most similar exemplar has a non-zero learning rate, which is constant.
    
    Notes
    -----
    This is the form of learning assumed by Ghirlanda (2015) when showing the equivalence
    between exemplar and RW family models.
    """
    selector = np.zeros(n['ex'])
    selector[np.argmax(state['sim'])] = 1
    lrate = sim_pars['lrate_par']*selector # learning rates for exemplars
    delta = env['y'] - state['y_hat'] # prediction error (total)
    return np.outer(lrate, env['y_lrn']*delta)
only_max.pars = pd.DataFrame({'min': 0.0, 'max': 1.0, 'default': 0.5}, index = ['lrate_par'])

def ex_mean(state, n, env, sim_pars):
    """
    Each y_ex is simply the mean of y when that exemplar is present.
    
    Notes
    -----
    Combined with the right type of retrieval function, this is equivalent
    to instance based learning: instances with the same cues (x)
    are simply grouped together as a single exemplar.
    
    One can add a fixed initial value ('nu') to ex_counts.
    """
    index = np.argmax(state['sim'])
    lrate = np.zeros(n['ex'])
    lrate[index] = 1/(state['ex_counts'][index] + sim_pars['nu'])
    delta = env['y'] - state['y_ex'][index] # prediction error (only for current exemplar)
    return np.outer(lrate, env['y_lrn']*delta)
ex_mean.pars = pd.DataFrame({'min': 0.0, 'max': 10.0, 'default': 0.0}, index = ['nu']) # extra counts for ex_mean