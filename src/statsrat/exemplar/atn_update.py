import numpy as np
import pandas as pd

def null(state, n, env, sim_pars):
    '''
    Don't update attention (it remains constant).
    '''
    return 0
null.pars = None

def gradient_ngsec(state, n, env, sim_pars):
    '''
    Gradient descent on total squared error (assuming separate attention weights for each exemplar)
    when rtrv = normalized_sim_ex_counts and sim = Gaussian.
    
    Notes
    -----
    I have double checked that the math is correct (SP, 4/14/2021).
    '''
    delta = env['y'] - state['y_hat']
    # use loops to keep things simple for now
    update = sim_pars['atn_lrate_par']*sim_pars['decay_rate']*np.ones((n['ex'], n['x']))
    for m in range(n['ex']):
        for l in range(n['x']):
            sq_dist = (env['x'][l] - state['x_ex'][m, l])**2
            error_factor = np.sum(delta*(state['y_hat'] - state['y_ex'][m, :]))
            update[m, l] *= state['rtrv'][m]*sq_dist*error_factor
    return update
gradient_ngsec.pars = pd.DataFrame({'min': 0.0, 'max': 1.0, 'default': 0.5}, index = ['atn_lrate_par']) # learning rate for attention updates

def gradient_ngsec_common(state, n, env, sim_pars):
    '''
    Gradient descent on total squared error (assuming common attention weights across exemplars)
    when rtrv = normalized_sim_ex_counts and sim = Gaussian.
    '''
    delta = env['y'] - state['y_hat']
    # use loops to keep things simple for now
    update = -sim_pars['atn_lrate_par']*sim_pars['decay_rate']*np.ones((n['ex'], n['x']))
    for m in range(n['x']):
        sq_dist = (env['x'][m] - state['x_ex'][:, m])**2
        rwsd = np.sum(state['rtrv']*sq_dist) # retrieval weighted sum of sq_dist
        foo = state['y_ex']*(state['rtrv']*(sq_dist - rwsd)).reshape((n['ex'], 1))
        ex_factor = np.sum(foo, axis = 0)
        update[:, m] *= np.sum(delta*ex_factor)
    return update
gradient_ngsec_common.pars = pd.DataFrame({'min': 0.0, 'max': 1.0, 'default': 0.5}, index = ['atn_lrate_par']) # learning rate for attention updates

def gradient_ngsec_both(state, n, env, sim_pars):
    '''
    Gradient descent on total squared error when rtrv = normalized_sim_ex_counts and sim = Gaussian.
    Attention weights have two parts: one that is common across exemplars (for each cue) and one
    that is unique to each exemplar/cue.
    '''
    delta = env['y'] - state['y_hat']
    # update for common part of weights
    update_c = -sim_pars['atn_lrate_par']*sim_pars['decay_rate']*np.ones((n['ex'], n['x']))
    for m in range(n['x']):
        sq_dist = (env['x'][m] - state['x_ex'][:, m])**2
        rwsd = np.sum(state['rtrv']*sq_dist) # retrieval weighted sum of sq_dist
        foo = state['y_ex']*(state['rtrv']*(sq_dist - rwsd)).reshape((n['ex'], 1))
        ex_factor = np.sum(foo, axis = 0)
        update_c[:, m] *= np.sum(delta*ex_factor)
    
    # update for separate part of weights
    update_s = sim_pars['atn_lrate_par']*sim_pars['decay_rate']*np.ones((n['ex'], n['x']))
    for l in range(n['ex']):
        for m in range(n['x']):
            sq_dist = (env['x'][m] - state['x_ex'][l, m])**2
            error_factor = np.sum(delta*(state['y_hat'] - state['y_ex'][l, :]))
            update_s[l, m] *= state['rtrv'][l]*sq_dist*error_factor
    
    return update_c + update_s
gradient_ngsec_both.pars = pd.DataFrame({'min': 0.0, 'max': 1.0, 'default': 0.5}, index = ['atn_lrate_par']) # learning rate for attention updates

def gradient_norm_cityblock_common(state, n, env, sim_pars):
    '''
    Gradient descent on total squared error (assuming common attention weights across exemplars)
    when rtrv = normalized_sim_ex_counts and sim = city_block (based on L1 distance).
    '''
    delta = env['y'] - state['y_hat']
    # use loops to keep things simple for now
    update = -sim_pars['atn_lrate_par']*sim_pars['decay_rate']*np.ones((n['ex'], n['x']))
    for m in range(n['x']):
        abs_dif = np.abs(env['x'][m] - state['x_ex'][:, m])
        rwsd = np.sum(state['rtrv']*abs_dif) # retrieval weighted sum of sq_dist
        foo = state['y_ex']*(state['rtrv']*(abs_dif - rwsd)).reshape((n['ex'], 1))
        ex_factor = np.sum(foo, axis = 0)
        update[:, m] *= np.sum(delta*ex_factor)
    return update
gradient_norm_cityblock_common.pars = pd.DataFrame({'min': 0.0, 'max': 1.0, 'default': 0.5}, index = ['atn_lrate_par']) # learning rate for attention updates

def heuristic(state, n, env, sim_pars):
    '''
    Heuristic designed to adjust attention toward relevant stimuli.
    Each exemplar has a separate set of attention weights.
    Only the current exemplar's weights are adjusted.
    '''
    current = sim == 1 # assume that current exemplar has a similarity of 1, and no others do
    update = np.zeros((n['ex'], n['x']))
    for l in range(n['ex']):
        if ex_seen_yet[l]:
            sq_y_dist = np.sum((state['y_ex'][l, :] - env['y'])**2)
            for m in range(n['x']):
                sq_x_dist = (state['x_ex'][l, m] - env['x'][m])**2
                update[current, m] += sim_pars['atn_lrate_par']*sq_x_dist*sq_y_dist
    return update
heuristic.pars = pd.DataFrame({'min': 0.0, 'max': 1.0, 'default': 0.5}, index = ['atn_lrate_par']) # learning rate for attention updates