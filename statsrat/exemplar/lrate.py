import numpy as np

def from_rtrv(rtrv, n_ex, sim_pars):
    """
    Learning rates for exemplars are equal to retrieval strength
    (normalized similarity) times a constant (lrate_par).
    """
    return sim_pars['lrate_par']*rtrv # FIX THIS
from_rtrv.par_names = ['lrate_par']

def only_max(rtrv, n_ex, sim_pars):
    """
    Only the most similar exemplar has a non-zero learning rate, which is constant.
    """
    selector = np.zeros(n_ex)
    selector[np.argmax(rtrv)] = 1
    return sim_pars['lrate_par']*selector # FIX THIS
only_max.par_names = ['lrate_par']