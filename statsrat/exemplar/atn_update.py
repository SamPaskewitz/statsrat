import numpy as np

def null(sim, x, u, u_psb, rtrv, u_hat, u_lrn, x_ex, u_ex, n_x, n_u, ex_seen_yet, ex_counts, n_ex, sim_pars):
    '''
    Don't update attention (it remains constant).
    '''
    return 0
null.par_names = []

def gradient_ngsec(sim, x, u, u_psb, rtrv, u_hat, u_lrn, x_ex, u_ex, n_x, n_u, ex_seen_yet, ex_counts, n_ex, sim_pars):
    '''
    Gradient descent on total squared error (assuming separate attention weights for each exemplar)
    when rtrv = normalized_sim_ex_counts and sim = Gaussian.
    
    Notes
    -----
    I have double checked that the math is correct (SP, 4/14/2021).
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

def gradient_ngsec_common(sim, x, u, u_psb, rtrv, u_hat, u_lrn, x_ex, u_ex, n_x, n_u, ex_seen_yet, ex_counts, n_ex, sim_pars):
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

def gradient_ngsec_both(sim, x, u, u_psb, rtrv, u_hat, u_lrn, x_ex, u_ex, n_x, n_u, ex_seen_yet, ex_counts, n_ex, sim_pars):
    '''
    Gradient descent on total squared error when rtrv = normalized_sim_ex_counts and sim = Gaussian.
    Attention weights have two parts: one that is common across exemplars (for each cue) and one
    that is unique to each exemplar/cue.
    '''
    delta = u - u_hat
    # update for common part of weights
    update_c = -sim_pars['atn_lrate_par']*sim_pars['decay_rate']*np.ones((n_ex, n_x))
    for n in range(n_x):
        sq_dist = (x[n] - x_ex[:, n])**2
        rwsd = np.sum(rtrv*sq_dist) # retrieval weighted sum of sq_dist
        foo = u_ex*(rtrv*(sq_dist - rwsd)).reshape((n_ex, 1))
        ex_factor = np.sum(foo, axis = 0)
        update_c[:, n] *= np.sum(delta*ex_factor)
    
    # update for separate part of weights
    update_s = sim_pars['atn_lrate_par']*sim_pars['decay_rate']*np.ones((n_ex, n_x))
    for m in range(n_ex):
        for n in range(n_x):
            sq_dist = (x[n] - x_ex[m, n])**2
            error_factor = np.sum(delta*(u_hat - u_ex[m, :]))
            update_s[m, n] *= rtrv[m]*sq_dist*error_factor
    
    return update_c + update_s
gradient_ngsec_both.par_names = ['atn_lrate_par']

def heuristic(sim, x, u, u_psb, rtrv, u_hat, u_lrn, x_ex, u_ex, n_x, n_u, ex_seen_yet, ex_counts, n_ex, sim_pars):
    '''
    Heuristic designed to adjust attention toward relevant stimuli.
    Each exemplar has a separate set of attention weights.
    Only the current exemplar's weights are adjusted.
    '''
    current = sim == 1 # assume that current exemplar has a similarity of 1, and no others do
    update = np.zeros((n_ex, n_x))
    for m in range(n_ex):
        if ex_seen_yet[m]:
            sq_y_dist = np.sum((u_ex[m, :] - u)**2)
            for n in range(n_x):
                sq_x_dist = (x_ex[m, n] - x[n])**2
                update[current, n] += sim_pars['atn_lrate_par']*sq_x_dist*sq_y_dist
    return update
heuristic.par_names = ['atn_lrate_par']