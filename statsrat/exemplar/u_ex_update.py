import numpy as np

def from_sim(sim, u, u_hat, u_lrn, u_ex, ex_counts, n_ex, sim_pars):
    """
    Learning rates for exemplars are equal to retrieval strength
    (normalized similarity) times a constant (lrate_par).
    This (minus the 'humble teachers') is the form of learning used
    in ALCOVE (Kruschke, 1992).
    """
    lrate = sim_pars['lrate_par']*sim # learning rates for exemplars
    delta = u - u_hat # prediction error (total)
    return np.outer(lrate, u_lrn*delta)
from_sim.par_names = ['lrate_par']

def only_max(sim, u, u_hat, u_lrn, u_ex, ex_counts, n_ex, sim_pars):
    """
    Only the most similar exemplar has a non-zero learning rate, which is constant.
    This is the form of learning assumed by Ghirlanda (2015) when showing the equivalence
    between exemplar and RW family models.
    """
    selector = np.zeros(n_ex)
    selector[np.argmax(sim)] = 1
    lrate = sim_pars['lrate_par']*selector # learning rates for exemplars
    delta = u - u_hat # prediction error (total)
    return np.outer(lrate, u_lrn*delta)
only_max.par_names = ['lrate_par']

def ex_mean(sim, u, u_hat, u_lrn, u_ex, ex_counts, n_ex, sim_pars):
    """
    Each u_ex is simply the exponentially weight moving average of u when that exemplar is present.
    """
    selector = np.zeros(n_ex)
    selector[np.argmax(sim)] = 1
    lrate = sim_pars['lrate_par']*selector # learning rates for exemplars
    delta = u - u_ex[np.argmax(sim)] # prediction error (only for current exemplar)
    return np.outer(lrate, u_lrn*delta)
ex_mean.par_names = ['lrate_par']