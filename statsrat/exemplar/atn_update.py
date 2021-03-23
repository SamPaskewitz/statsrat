import numpy as np

def null(sim, x, u, u_psb, rtrv, u_hat, u_lrn, x_ex, u_ex, n_x, n_u, ex_counts, n_ex, sim_pars):
    '''
    Don't update attention (it remains constant).
    '''
    return 0
null.par_names = []

def gradient_ngsec(sim, x, u, u_psb, rtrv, u_hat, u_lrn, x_ex, u_ex, n_x, n_u, ex_counts, n_ex, sim_pars):
    '''
    Gradient descent on total squared error (assuming separate attention weights for each exemplar)
    when rtrv = normalized_sim_ex_counts and sim = Gaussian.
    '''
    delta = u - u_hat
    # use loops to keep things simple for now
    update = sim_pars['atn_lrate_par']*sim_pars['decay_rate']*np.ones((n_ex, n_x))
    for m in range(n_ex):
        for n in range(n_x):
            sq_dist = (x[n] - x_ex[m, n])**2
            error_factor = np.sum(delta*(u_hat - u_ex[m, :]))
            update[m, n] *= rtrv[m]*sq_dist*error_factor
    return update
gradient_ngsec.par_names = ['atn_lrate_par']