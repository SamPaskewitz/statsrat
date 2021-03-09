import numpy as np

def equal_to_sim(sim, sim_pars):
    '''Retrieval strength is equal to similarity.'''
    return sim
equal_to_sim.par_names = []

def normalized_sim(sim, sim_pars):
    '''Retrieval strength is normalized similarity.'''
    return sim/sim.sum()
normalized_sim.par_names = []