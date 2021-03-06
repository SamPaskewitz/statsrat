import numpy as np

def from_sim(sim, n_ex, sim_pars):
    """
    Learning rates for exemplars are equal to retrieval strength
    (normalized similarity) times a constant (lrate_par).
    """
    return sim_pars['lrate_par']*sim
from_sim.par_names = ['lrate_par']

def only_max(sim, n_ex, sim_pars):
    """
    Only the most similar exemplar has a non-zero learning rate, which is constant.
    """
    selector = np.zeros(n_ex)
    selector[np.argmax(sim)] = 1
    return sim_pars['lrate_par']*selector
only_max.par_names = ['lrate_par']