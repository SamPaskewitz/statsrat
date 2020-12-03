import numpy as np

def power(sim_pars, rwds):
    '''Power law utility function.'''
    utility = rwds**sim_pars['alpha']
    # FIX FOR NEGATIVE RWDS
    return utility

def power_asym(sim_pars, rwds):
    '''Power law that differs for gains and losses.'''
    # FINISH