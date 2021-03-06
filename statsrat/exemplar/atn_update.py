import numpy as np

def null(sim, delta, n_x, n_u, n_ex, sim_pars):
    '''
    Don't update attention (it remains constant).
    '''
    return 0
null.par_names = []