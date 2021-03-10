import numpy as np

def from_rtrv(sim, rtrv, u, u_hat, u_lrn, u_ex, ex_counts, n_ex, n_u, sim_pars):
    """
    Learning rates for exemplars are equal to retrieval strength
    times a constant (lrate_par).  This (minus the 'humble teachers',
    and with retrieval strength equal to similarity) is the form of 
    learning used in ALCOVE (Kruschke, 1992).
    """
    lrate = sim_pars['lrate_par']*rtrv # learning rates for exemplars
    delta = u - u_hat # prediction error (total)
    return np.outer(lrate, u_lrn*delta)
from_rtrv.par_names = ['lrate_par']

def from_rtrv_indv_delta(sim, rtrv, u, u_hat, u_lrn, u_ex, ex_counts, n_ex, n_u, sim_pars):
    """
    Learning rates for exemplars are equal to retrieval strength
    times a constant (lrate_par), and prediction errors are for each
    individual exemplar rather than for common.
    """
    lrate = sim_pars['lrate_par']*rtrv # learning rates for exemplars
    update = np.zeros((n_ex, n_u))
    for i in range(n_ex):
        delta = u - u_ex[i] # prediction error (only for exemplar i)
        update[i, :] = lrate[i]*delta
    return update
from_rtrv_indv_delta.par_names = ['lrate_par']

def only_max(sim, rtrv, u, u_hat, u_lrn, u_ex, ex_counts, n_ex, n_u, sim_pars):
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

def ex_mean(sim, rtrv, u, u_hat, u_lrn, u_ex, ex_counts, n_ex, n_u, sim_pars):
    """
    Each u_ex is simply the mean of u when that exemplar is present.
    Combined with the right type of retrieval function, this is equivalent
    to instance based learning: instances with the same cues (x)
    are simply grouped together as a single exemplar.
    """
    index = np.argmax(sim)
    lrate = np.zeros(n_ex)
    lrate[index] = 1/ex_counts[index]
    delta = u - u_ex[index] # prediction error (only for current exemplar)
    return np.outer(lrate, u_lrn*delta)
ex_mean.par_names = []