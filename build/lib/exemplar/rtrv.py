import numpy as np

def equal_to_sim(sim, ex_counts, ex_seen_yet, sim_pars):
    '''Retrieval strength is equal to similarity.'''
    return ex_seen_yet*sim
equal_to_sim.par_names = []

def normalized_sim(sim, ex_counts, ex_seen_yet, sim_pars):
    '''Retrieval strength is normalized similarity.'''
    numerator = ex_seen_yet*sim
    return numerator/numerator.sum()
normalized_sim.par_names = []

def normalized_sim_ex_counts(sim, ex_counts, ex_seen_yet, sim_pars):
    '''
    Retrieval strength is similarity times exemplar counts, normalized.
    
    Notes
    -----
    Combined with the right u_ex_update function (viz. ex_mean), this
    reproduces instance based learning with a normalized similarity retrieval
    function across instances.
    
    One can add a fixed initial value ('nu') to ex_counts.
    '''
    numerator = ex_seen_yet*sim*(ex_counts + sim_pars['nu'])
    return numerator/numerator.sum()
normalized_sim_ex_counts.par_names = ['nu']