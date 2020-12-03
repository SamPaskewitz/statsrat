import numpy as np

def identity(sim_pars, probs):
    '''Decision weights are simply probabilities.'''
    return probs