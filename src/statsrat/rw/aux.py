import numpy as np
import pandas as pd

########## GENERAL PURPOSE AUXILIARY LEARNING OBJECTS ##########

# Defines auxilliary learning (e.g. for selective attention) in the form of classes.

def basic(state, n, env, sim_pars, purpose):
    '''Basic auxiliary learning (keeps track of feature counts).'''
    if purpose == 'initialize':
        new_state = {'f_counts': np.zeros(n['f'])}
        new_state_dims = {'f_counts': ['f_name']}
        new_state_sizes = {'f_counts': [n['f']]}
        return new_state, new_state_dims, new_state_sizes

    elif purpose == 'compute':
        state['f_counts'][state['f_x'] != 0] += 1
        return state
    
    elif purpose == 'update':
        return state
basic.pars = None
        
def drva(state, n, env, sim_pars, purpose):
    '''Derived attention (Le Pelley et al, 2016).'''
    if purpose == 'initialize':
        new_state = {'atn': np.zeros(n['f'])}
        new_state_dims = {'atn': ['f_name']}
        new_state_sizes = {'atn': [n['f']]}
        return new_state, new_state_dims, new_state_sizes

    elif purpose == 'compute':
        abs_w_sum = np.sum(abs(state['w']), axis = 1)
        abv_min = abs_w_sum >= sim_pars['atn_min']
        blw_max = abs_w_sum < 1
        state['atn'] = abs_w_sum*abv_min*blw_max + sim_pars['atn_min']*(1 - abv_min) + 1*(1 - blw_max)
        return state
    
    elif purpose == 'update':
        return state
drva.pars = pd.DataFrame({'min': 0.0, 'max': 1.0, 'default': 0.1}, index = ['atn_min'])

def tdrva(state, n, env, sim_pars, purpose): # FIX
    '''
    Derived attention, but with trial by trial tracking of associative strength.
    Attention is a sigmoidal function of mean associative strength (tau).
    This is a bit like the model of Frey and Sears (1978), but is simpler.
    '''
    if purpose == 'initialize':
        new_state = {'atn': np.zeros(n['f']), # attention (i.e. learning rate for present features)
                     'tau': sim_pars['tau0']*np.ones(n['f'])} # tracks the mean of |w| across outcomes
        new_state_dims = {'atn': ['f_name'], 'tau': ['f_name']}
        new_state_sizes = {'atn': [n['f']], 'tau': [n['f']]}
        return new_state, new_state_dims, new_state_sizes
    
    elif purpose == 'compute':
        denom = (1/state['tau'] + 1)**sim_pars['power']
        state['atn'] = 0.5/denom + sim_pars['lrate_min']
        return state
    
    elif purpose == 'update':
        abs_w_mean = np.mean(abs(state['w']), axis = 1)
        f_present = state['f_x'] > 0
        state['tau'] += f_present*sim_pars['lrate_tau']*(abs_w_mean - state['tau'])
        return state        
tdrva.pars = pd.DataFrame([{'min': 0.0, 'max': 0.5, 'default': 0.1}, {'min': 0.0, 'max': 2.0, 'default': 0.5}, {'min': 0.01, 'max': 1.0, 'default': 0.5}, {'min': 0.0, 'max': 1.0, 'default': 0.2}], index = ['lrate_min', 'power', 'tau0', 'lrate_tau'])

def grad(state, n, env, sim_pars, purpose):
    '''
    Non-competitive attention learning from gradient descent (i.e. simple predictiveness/Model 2).
    Updates attention ('atn') by gradient descent on squared error (derived assuming 'fweight = 'fweight_direct').
    '''
    if purpose == 'initialize':
        new_state = {'atn': 0.5*np.ones(n['f'])}
        new_state_dims = {'atn': ['f_name']}
        new_state_sizes = {'atn': [n['f']]}
        return new_state, new_state_dims, new_state_sizes

    elif purpose == 'compute':
        return state
    
    elif purpose == 'update':
        w_psb = state['w'] @ np.diag(env['y_psb']) # select only columns corresponding to possible outcomes
        ngrad = state['delta'] @ w_psb.T @ np.diag(state['fbase']) # negative gradient
        new_atn = state['atn'] + sim_pars['lrate_atn'] * ngrad
        abv_min = new_atn >= 0.01
        blw_max = new_atn < 1
        new_atn = new_atn*abv_min*blw_max + 0.01*(1 - abv_min) + 1*(1 - blw_max)
        state['atn'] = new_atn
        return state 
grad.pars = pd.DataFrame({'min': 0.0, 'max': 2.0, 'default': 0.2}, index = ['lrate_atn'])

def grad_elem_bias(state, n, env, sim_pars, purpose):
    '''
    Non-competitive attention learning from gradient descent (i.e. simple predictiveness/Model 2).
    There is an initial bias toward (or possibly away from) elemental features.  Starting attention
    for configural features is specified with the parameter atn0, while elemental features start with
    an attention of 0.5.
    
    Notes
    -----
    This uses similar somewhat hacky code to gradcomp_elem_bias for identifying configural features.
    '''
    if purpose == 'initialize':
        new_state = {'atn': sim_pars['atn0']*np.ones(n['f'])}
        new_state['atn'][0, range(n['x'])] = 0.5 # the first n['x'] features are assumed to be the elemental ones and get higher initial attention
        new_state_dims = {'atn': ['f_name']}
        new_state_sizes = {'atn': [n['f']]}
        return new_state, new_state_dims, new_state_sizes

    elif purpose == 'compute':
        return state
    
    elif purpose == 'update':
        w_psb = state['w'] @ np.diag(env['y_psb']) # select only columns corresponding to possible outcomes
        ngrad = state['delta'] @ w_psb.T @ np.diag(state['fbase']) # negative gradient
        new_atn = state['atn'] + sim_pars['lrate_atn'] * ngrad
        abv_min = new_atn >= 0.01
        blw_max = new_atn < 1
        new_atn = new_atn*abv_min*blw_max + 0.01*(1 - abv_min) + 1*(1 - blw_max)
        state['atn'] = new_atn
        return state
grad_elem_bias.pars = pd.DataFrame([{'min': 0.0, 'max': 1.0, 'default': 0.5}, {'min': 0.0, 'max': 2.0, 'default': 0.2}], index = ['atn0', 'lrate_atn'])

def gradcomp(state, n, env, sim_pars, purpose):
    '''
    Competitive attention learning from gradient descent (i.e. CompAct's learning rule).
    Updates attention ('atn') by gradient descent on squared error (derived assuming 'fweight = 'fweight_norm').
    '''
    if purpose == 'initialize':
        new_state = {'atn': np.ones(n['f'])}
        new_state_dims = {'atn': ['f_name']}
        new_state_sizes = {'atn': [n['f']]}
        return new_state, new_state_dims, new_state_sizes

    elif purpose == 'compute':
        return state
    
    elif purpose == 'update':
        w_psb = state['w'] @ np.diag(env['y_psb']) # select only columns corresponding to possible outcomes
        y_hat_alone = w_psb.T @ np.diag(state['fbase'])
        comp_factor = state['fweight']**(sim_pars['metric'] - 1)
        y_hat_dif = y_hat_alone - np.outer(state['y_hat'], comp_factor)
        atn_gain = state['atn']*state['fbase']
        norm = sum(atn_gain**sim_pars['metric'])**(1/sim_pars['metric'])
        ngrad = state['delta'].reshape((1, n['y'])) @ y_hat_dif @ np.diag(state['fbase']/norm) # negative gradient
        state['atn'] = np.maximum(state['atn'] + sim_pars['lrate_atn']*ngrad, n['f']*[0.01]).squeeze()
        return state
gradcomp.pars = pd.DataFrame([{'min': 0.0, 'max': 2.0, 'default': 0.2}, {'min': 0.1, 'max': 10, 'default': 2}], index = ['lrate_atn', 'metric'])

def gradcomp_feature_counts(state, n, env, sim_pars, purpose):
    '''
    Competitive attention learning from gradient descent (i.e. CompAct's learning rule).
    Updates attention ('atn') by gradient descent on squared error (derived assuming 'fweight = 'fweight_norm').
    Also keeps track of feature counts.
    '''
    if purpose == 'initialize':
        new_state = {'f_counts': np.zeros(n['f']), 'atn': np.ones(n['f'])}
        new_state_dims = {'f_counts': ['f_name'], 'atn': ['f_name']}
        new_state_sizes = {'f_counts': [n['f']], 'atn': [n['f']]}
        return new_state, new_state_dims, new_state_sizes

    elif purpose == 'compute':
        state['f_counts'][state['f_x'] != 0] += 1
        return state
    
    elif purpose == 'update':
        w_psb = state['w'] @ np.diag(env['y_psb']) # select only columns corresponding to possible outcomes
        y_hat_alone = w_psb.T @ np.diag(state['fbase'])
        comp_factor = state['fweight']**(sim_pars['metric'] - 1)
        y_hat_dif = y_hat_alone - np.outer(state['y_hat'], comp_factor)
        atn_gain = state['atn']*state['fbase']
        norm = sum(atn_gain**sim_pars['metric'])**(1/sim_pars['metric'])
        ngrad = state['delta'].reshape((1, n['y'])) @ y_hat_dif @ np.diag(state['fbase']/norm) # negative gradient
        state['atn'] = np.maximum(state['atn'] + sim_pars['lrate_atn']*ngrad, n['f']*[0.01]).squeeze()
        return state
gradcomp_feature_counts.pars = pd.DataFrame([{'min': 0.0, 'max': 2.0, 'default': 0.2}, {'min': 0.1, 'max': 10, 'default': 2}], index = ['lrate_atn', 'metric'])

def gradcomp_elem_bias(state, n, env, sim_pars, purpose):
    '''
    Competitive attention learning from gradient descent (i.e. CompAct's learning rule).
    Elemental features get an initial attention value of 1, while other features (e.g.
    configural ones) get an initial attention value of eta0 (a free parameter).  Thus
    attention is biased towards elemental features.
    
    This object also keeps track of feature counts.
    
    Notes
    -----
    This uses a simple hack to decide which features are elemental, viz. the first
    n['x'] features are assumed to be the elemental ones.  This works with all of the
    built in fbase functions because these start with elemental features and then add
    configural features, intercept terms etc.  One should be careful when developing
    any new fbase function to copy this behavior, or else this aux object may not work
    as intended.
    '''
    if purpose == 'initialize':
        new_state = {'f_counts': np.zeros(n['f']), 'atn': sim_pars['eta0']*np.ones(n['f'])}
        new_state['atn'][0, range(n['x'])] = 1 # the first n['x'] features are assumed to be the elemental ones and get higher initial attention
        new_state_dims = {'f_counts': ['f_name'], 'atn': ['f_name']}
        new_state_sizes = {'f_counts': [n['f']], 'atn': [n['f']]}
        return new_state, new_state_dims, new_state_sizes

    elif purpose == 'compute':
        state['f_counts'][state['f_x'] != 0] += 1
        return state
    
    elif purpose == 'update':
        w_psb = state['w'] @ np.diag(env['y_psb']) # select only columns corresponding to possible outcomes
        y_hat_alone = w_psb.T @ np.diag(state['fbase'])
        comp_factor = state['fweight']**(sim_pars['metric'] - 1)
        y_hat_dif = y_hat_alone - np.outer(state['y_hat'], comp_factor)
        atn_gain = state['atn']*state['fbase']
        norm = sum(atn_gain**sim_pars['metric'])**(1/sim_pars['metric'])
        ngrad = state['delta'].reshape((1, n['y'])) @ y_hat_dif @ np.diag(state['fbase']/norm) # negative gradient
        state['atn'] = np.maximum(state['atn'] + sim_pars['lrate_atn']*ngrad, n['f']*[0.01]).squeeze()
        return state
gradcomp_elem_bias.pars = pd.DataFrame([{'min': 0.0, 'max': 2.0, 'default': 0.2}, {'min': 0.1, 'max': 10, 'default': 2}, {'min': 0.0, 'max': 10.0, 'default': 1}], index = ['lrate_atn', 'metric', 'eta0'])

def gradcomp_atn_decay(state, n, env, sim_pars, purpose):
    '''
    Competitive attention learning from gradient descent (i.e. CompAct's learning rule)
    with decay of attention for features that are present on each trial.
    This is intended to produce latent inhibition and similar phenomena.
    
    Notes
    -----
    Preliminary simulations suggest that this does NOT produce latent inhibition,
    but further investigation is needed.
    '''
    if purpose == 'initialize':
        new_state = {'atn': np.ones(n['f'])}
        new_state_dims = {'atn': ['f_name']}
        new_state_sizes = {'atn': [n['f']]}
        return new_state, new_state_dims, new_state_sizes

    elif purpose == 'compute':
        return state
    
    elif purpose == 'update':
        w_psb = state['w'] @ np.diag(env['y_psb']) # select only columns corresponding to possible outcomes
        y_hat_alone = w_psb.T @ np.diag(state['fbase'])
        comp_factor = state['fweight']**(sim_pars['metric'] - 1)
        y_hat_dif = y_hat_alone - np.outer(state['y_hat'], comp_factor)
        atn_gain = state['atn']*state['fbase']
        norm = sum(atn_gain**sim_pars['metric'])**(1/sim_pars['metric'])
        ngrad = state['delta'].reshape((1, n['y'])) @ y_hat_dif @ np.diag(state['fbase']/norm) # negative gradient
        decay_term = sim_pars['drate_atn']*state['atn']*state['fbase']
        state['atn'] = np.maximum(state['atn'] + sim_pars['lrate_atn']*ngrad - decay_term, n['f']*[0.01]).squeeze()
        return state
gradcomp_atn_decay.pars = pd.DataFrame([{'min': 0.0, 'max': 2.0, 'default': 0.2}, {'min': 0.1, 'max': 10, 'default': 2}, {'min': 0.0, 'max': 2.0, 'default': 0.2}], index = ['lrate_atn', 'metric', 'drate_atn'])

def gradcomp_Kruschke_idea(state, n, env, sim_pars, purpose): # FIX
    '''
    Competitive attention learning from gradient descent (i.e. CompAct's learning rule).
    Each feature is assumed to start with an association to a virtual 'outcome' that
    never occurs.  This should lead to latent inhibition.
    
    Notes
    -----
    This is based on an idea suggested by Kruschke (for his EXIT model, which is
    similar to CompAct) in the following paper:
    
    Kruschke, J. K. (2001).
    Toward a Unified Model of Attention in Associative Learning.
    Journal of Mathematical Psychology, 45(6), 812–863.

    '''
    def __init__(self, state, n, env, sim_pars):
        self.data = {'atn': np.zeros((n['t'] + 1, n['f'])),
                     'w_virtual': np.zeros((n['t'] + 1, n['f']))}
        self.data['atn'][0, :] = 1
        self.data['w_virtual'][0, :] = sim_pars['w_virtual0']

    def update(self, state, n, env, sim_pars):
        '''
        Update 'atn' by gradient descent on squared error (derived assuming 'fweight = 'fweight_norm')
        with decay of 'atn' for features that are present.  Also keep track of associations involving
        the virtual 'outcome' that is initially predicted but never occurs.
        '''
        # calculations based on the real outcomes
        w_psb = state['w'] @ np.diag(env['y_psb']) # select only columns corresponding to possible outcomes
        y_hat_alone = w_psb.T @ np.diag(state['fbase'])
        comp_factor = state['fweight']**(sim_pars['metric'] - 1)
        y_hat_dif = y_hat_alone - np.outer(state['y_hat'], comp_factor)
        atn_gain = state['atn']*state['fbase']
        norm = sum(atn_gain**sim_pars['metric'])**(1/sim_pars['metric'])
        ngrad = state['delta'].reshape((1, n['y'])) @ y_hat_dif @ np.diag(state['fbase']/norm) # negative gradient from real outcomes
        # calculations based on the virtual 'outcome'
        y_hat_virtual = np.sum(self.data['w_virtual'][t, :]*state['f_x'])
        delta_virtual = 0 - y_hat_virtual
        y_hat_alone_virtual = self.data['w_virtual'][t, :]*state['fbase']
        y_hat_dif_virtual = y_hat_alone_virtual - y_hat_virtual*comp_factor
        ngrad_virtual = delta_virtual*y_hat_dif_virtual*state['fbase']/norm # negative gradient from the virtual 'outcome'
        # update attention
        self.data['atn'][t + 1, :] = np.maximum(state['atn'] + sim_pars['lrate_atn']*(ngrad + ngrad_virtual), n['f']*[0.01])
        # update virtual 'outcome' associations
        self.data['w_virtual'][t + 1, :] = self.data['w_virtual'][t, :] + sim_pars['lrate']*state['f_x']*delta_virtual

    def add_data(self, ds):
        return ds.assign(atn = (['t', 'f_name'], self.data['atn'][range(n['t']), :]),
                         w_virtual = (['t', 'f_name'], self.data['w_virtual'][range(n['t']), :]))
gradcomp_Kruschke_idea.pars = pd.DataFrame([{'min': 0.0, 'max': 2.0, 'default': 0.2}, {'min': 0.1, 'max': 10, 'default': 2}, {'min': 0.0, 'max': 1.0, 'default': 0.5}], index = ['lrate_atn', 'metric', 'w_virtual0'])