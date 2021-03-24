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

def gradient_ngsec_common(sim, x, u, u_psb, rtrv, u_hat, u_lrn, x_ex, u_ex, n_x, n_u, ex_counts, n_ex, sim_pars):
    '''
    Gradient descent on total squared error (assuming common attention weights across exemplars)
    when rtrv = normalized_sim_ex_counts and sim = Gaussian.
    '''
    delta = u - u_hat
    # use loops to keep things simple for now
    update = -sim_pars['atn_lrate_par']*sim_pars['decay_rate']*np.ones((n_ex, n_x))
    for n in range(n_x):
        sq_dist = (x[n] - x_ex[:, n])**2
        rwsd = np.sum(rtrv*sq_dist) # retrieval weighted sum of sq_dist
        foo = u_ex*(rtrv*(sq_dist - rwsd)).reshape((n_ex, 1))
        ex_factor = np.sum(foo, axis = 0)
        update[:, n] *= np.sum(delta*ex_factor)
    return update
gradient_ngsec_common.par_names = ['atn_lrate_par']