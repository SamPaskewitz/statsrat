import numpy as np

def equal_to_sim(sim, ex_counts, sim_pars):
    '''Retrieval strength is equal to similarity.'''
    return sim
equal_to_sim.par_names = []

def normalized_sim(sim, ex_counts, sim_pars):
    '''Retrieval strength is normalized similarity.'''
    return sim/sim.sum()
normalized_sim.par_names = []

def normalized_sim_ex_counts(sim, ex_counts, sim_pars):
    '''
    Retrieval strength is similarity times exemplar counts, normalized.
    Combined with the right u_ex_update function (viz. ex_mean), this
    reproduces instance based learning with a normalized similarity retrieval
    function across instances.
    '''
    numerator = sim*ex_counts
    return numerator/numerator.sum()
normalized_sim_ex_counts.par_names = []