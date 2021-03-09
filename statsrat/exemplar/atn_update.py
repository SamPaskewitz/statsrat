import numpy as np

def null(sim, u, u_hat, u_lrn, u_ex, n_x, n_u, ex_counts, n_ex, sim_pars):
    '''
    Don't update attention (it remains constant).
    '''
    return 0
null.par_names = []