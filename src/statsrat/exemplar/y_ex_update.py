import numpy as np
import pandas as pd

'''
Functions to specify how y_ex (exemplar -> outcome associations) are updated.

from_rtrv: Learning rates for exemplars are equal to retrieval strength
    times a constant (lrate_par).

from_rtrv_indv_delta: Learning rates for exemplars are equal to retrieval strength
    times a constant (lrate_par), and prediction errors are for each
    individual exemplar rather than for common.

only_max: Only the most similar exemplar has a non-zero learning rate, which is constant.

ex_mean: Each y_ex is simply the mean of u when that exemplar is present.
'''

def from_rtrv(sim, rtrv, y, y_hat, y_lrn, y_ex, ex_counts, n_ex, n_y, sim_pars):
    """
    Learning rates for exemplars are equal to retrieval strength
    times a constant (lrate_par).
    
    Notes
    -----
    This (minus the 'humble teachers',
    and with retrieval strength equal to similarity) is the form of 
    learning used in ALCOVE (Kruschke, 1992).
    """
    lrate = sim_pars['lrate_par']*rtrv # learning rates for exemplars
    delta = y - y_hat # prediction error (total)
    return np.outer(lrate, y_lrn*delta)
from_rtrv.pars = pd.DataFrame({'min': 0.0, 'max': 1.0, 'default': 0.5}, index = ['lrate_par'])

def from_rtrv_indv_delta(sim, rtrv, y, y_hat, y_lrn, y_ex, ex_counts, n_ex, n_y, sim_pars):
    """
    Learning rates for exemplars are equal to retrieval strength
    times a constant (lrate_par), and prediction errors are for each
    individual exemplar rather than for common.
    """
    lrate = sim_pars['lrate_par']*rtrv # learning rates for exemplars
    update = np.zeros((n_ex, n_y))
    for i in range(n_ex):
        delta = y - y_ex[i] # prediction error (only for exemplar i)
        update[i, :] = lrate[i]*delta
    return update
from_rtrv_indv_delta.pars = pd.DataFrame({'min': 0.0, 'max': 1.0, 'default': 0.5}, index = ['lrate_par'])

def only_max(sim, rtrv, y, y_hat, y_lrn, y_ex, ex_counts, n_ex, n_y, sim_pars):
    """
    Only the most similar exemplar has a non-zero learning rate, which is constant.
    
    Notes
    -----
    This is the form of learning assumed by Ghirlanda (2015) when showing the equivalence
    between exemplar and RW family models.
    """
    selector = np.zeros(n_ex)
    selector[np.argmax(sim)] = 1
    lrate = sim_pars['lrate_par']*selector # learning rates for exemplars
    delta = y - y_hat # prediction error (total)
    return np.outer(lrate, y_lrn*delta)
only_max.pars = pd.DataFrame({'min': 0.0, 'max': 1.0, 'default': 0.5}, index = ['lrate_par'])

def ex_mean(sim, rtrv, y, y_hat, y_lrn, y_ex, ex_counts, n_ex, n_y, sim_pars):
    """
    Each y_ex is simply the mean of u when that exemplar is present.
    
    Notes
    -----
    Combined with the right type of retrieval function, this is equivalent
    to instance based learning: instances with the same cues (x)
    are simply grouped together as a single exemplar.
    
    One can add a fixed initial value ('nu') to ex_counts.
    """
    index = np.argmax(sim)
    lrate = np.zeros(n_ex)
    lrate[index] = 1/(ex_counts[index] + sim_pars['nu'])
    delta = y - y_ex[index] # prediction error (only for current exemplar)
    return np.outer(lrate, y_lrn*delta)
ex_mean.pars = pd.DataFrame({'min': 0.0, 'max': 10.0, 'default': 0.0}, index = ['nu']) # extra counts for ex_mean