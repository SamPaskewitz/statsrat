import numpy as np
import pandas as pd

########## GENERAL PURPOSE AUXILIARY LEARNING OBJECTS ##########
'''
Defines auxilliary learning (e.g. for selective attention) in the form of classes.
'''

class basic:
    '''Basic auxiliary learning (keeps track of feature counts).'''
    def __init__(self, sim_pars, n_t, n_x, n_f, n_y, f_names, x_dims):
        self.data = {'f_counts': np.zeros((n_t, n_f))}

    def update(self, sim_pars, n_y, n_f, t, fbase, fweight, f_x, y_psb, y_hat, delta, w):
        self.data['f_counts'][t, :] = np.apply_along_axis(np.sum, 0, fbase[0:(t+1), :] > 0)
        
    def add_data(self, ds):
        return ds.assign(f_counts = (['t', 'f_name'], self.data['f_counts']))
basic.par_names = []
        
class drva:
    '''Derived attention (Le Pelley et al, 2016).'''
    def __init__(self, sim_pars, n_t, n_x, n_f, n_y, f_names, x_dims):
        self.data = {'atn': np.zeros((n_t, n_f))}

    def update(self, sim_pars, n_y, n_f, t, fbase, fweight, f_x, y_psb, y_hat, delta, w):
        abs_w_sum = np.sum(abs(w[t, :]), axis = 1)
        abv_min = abs_w_sum >= sim_pars['atn_min']
        blw_max = abs_w_sum < 1
        self.data['atn'][t, :] = abs_w_sum*abv_min*blw_max + sim_pars['atn_min']*(1 - abv_min) + 1*(1 - blw_max)

    def add_data(self, ds):
        return ds.assign(atn = (['t', 'f_name'], self.data['atn'])) 
drva.par_names = ['atn_min']

class tdrva:
    '''
    Derived attention, but with trial by trial tracking of associative strength.
    Attention is a sigmoidal function of mean associative strength (tau).
    This is a bit like the model of Frey and Sears (1978), but is simpler.
    '''
    def __init__(self, sim_pars, n_t, n_x, n_f, n_y, f_names, x_dims):
        self.data = {'atn': np.zeros((n_t + 1, n_f)), # attention (i.e. learning rate for present features)
                     'tau': np.zeros((n_t + 1, n_f))} # tracks the mean of |w| across outcomes
        self.data['tau'][0, :] = sim_pars['tau0']
        denom = (1/sim_pars['tau0'] + 1)**sim_pars['power']
        self.data['atn'][0, :] = 0.5/denom + sim_pars['lrate_min']

    def update(self, sim_pars, n_y, n_f, t, fbase, fweight, f_x, y_psb, y_hat, delta, w):
        abs_w_mean = np.mean(abs(w[t, :]), axis = 1)
        f_present = f_x > 0
        self.data['tau'][t + 1, :] = self.data['tau'][t, :] + f_present*sim_pars['lrate_tau']*(abs_w_mean - self.data['tau'][t, :])
        denom = (1/self.data['tau'][t + 1, :] + 1)**sim_pars['power']
        self.data['atn'][t + 1, :] = 0.5/denom + sim_pars['lrate_min']

    def add_data(self, ds):
        n_t = ds['t'].values.shape[0]
        return ds.assign(atn = (['t', 'f_name'], self.data['atn'][range(n_t), :]), tau = (['t', 'f_name'], self.data['tau'][range(n_t), :]))
tdrva.par_names = ['lrate_min', 'power', 'tau0', 'lrate_tau']

class grad:
    '''Non-competitive attention learning from gradient descent (i.e. simple predictiveness/Model 2).'''
    def __init__(self, sim_pars, n_t, n_x, n_f, n_y, f_names, x_dims):
        self.n_t = n_t
        self.data = {'atn': np.zeros((n_t + 1, n_f))}
        self.data['atn'][0, :] = 0.5

    def update(self, sim_pars, n_y, n_f, t, fbase, fweight, f_x, y_psb, y_hat, delta, w):
        '''Update 'atn' by gradient descent on squared error (derived assuming 'fweight = 'fweight_direct').'''
        w_psb = w[t, :, :] @ np.diag(y_psb[t, :]) # select only columns corresponding to possible outcomes
        ngrad = delta[t, :] @ w_psb.T @ np.diag(fbase[t, :]) # negative gradient
        new_atn = self.data['atn'][t, :] + sim_pars['lrate_atn'] * ngrad
        abv_min = new_atn >= 0.01
        blw_max = new_atn < 1
        new_atn = new_atn*abv_min*blw_max + 0.01*(1 - abv_min) + 1*(1 - blw_max)
        self.data['atn'][t + 1, :] = new_atn
        
    def add_data(self, ds):
        return ds.assign(atn = (['t', 'f_name'], self.data['atn'][range(self.n_t), :]))  
grad.par_names = ['lrate_atn']

class gradcomp:
    '''Competitive attention learning from gradient descent (i.e. CompAct's learning rule).'''
    def __init__(self, sim_pars, n_t, n_x, n_f, n_y, f_names, x_dims):
        self.n_t = n_t
        self.data = {'atn': np.zeros((n_t + 1, n_f))}
        self.data['atn'][0, :] = 1

    def update(self, sim_pars, n_y, n_f, t, fbase, fweight, f_x, y_psb, y_hat, delta, w):
        '''Update 'atn' by gradient descent on squared error (derived assuming 'fweight = 'fweight_norm').'''
        w_psb = w[t, :, :] @ np.diag(y_psb[t, :]) # select only columns corresponding to possible outcomes
        y_hat_alone = w_psb.T @ np.diag(fbase[t, :])
        comp_factor = fweight[t, :]**(sim_pars['metric'] - 1)
        y_hat_dif = y_hat_alone - np.outer(y_hat[t, :], comp_factor)
        atn_gain = self.data['atn'][t, :]*fbase[t, :]
        norm = sum(atn_gain**sim_pars['metric'])**(1/sim_pars['metric'])
        ngrad = delta[t, :].reshape((1, n_y)) @ y_hat_dif @ np.diag(fbase[t, :]/norm) # negative gradient
        self.data['atn'][t + 1, :] = np.maximum(self.data['atn'][t, :] + sim_pars['lrate_atn']*ngrad, n_f*[0.01])

    def add_data(self, ds):
        return ds.assign(atn = (['t', 'f_name'], self.data['atn'][range(self.n_t), :]))   
gradcomp.par_names = ['lrate_atn', 'metric']

class gradcomp_elem_bias:
    '''
    Competitive attention learning from gradient descent (i.e. CompAct's learning rule).
    Elemental features get an initial attention value of 1, while other features (e.g.
    configural ones) get an initial attention value of 0.5.  Thus attention is biased
    towards elemental features.
    
    This object also keeps track of feature counts.
    
    Notes
    -----
    This uses a simple hack to decide which features are elemental, viz. the first
    n_x features are assumed to be the elemental ones.  This works with all of the
    built in fbase functions because these start with elemental features and then add
    configural features, intercept terms etc.  One should be careful when developing
    any new fbase function to copy this behavior, or else this aux object may not work
    as intended.
    '''
    def __init__(self, sim_pars, n_t, n_x, n_f, n_y, f_names, x_dims):
        self.n_t = n_t
        self.data = {'atn': np.zeros((n_t + 1, n_f)), 'f_counts': np.zeros((n_t, n_f))}
        self.data['atn'][0, :] = 0.5
        self.data['atn'][0, range(n_x)] = 1 # the first n_x features are assumed to be the elemental ones and get higher initial attention

    def update(self, sim_pars, n_y, n_f, t, fbase, fweight, f_x, y_psb, y_hat, delta, w):
        '''Update 'atn' by gradient descent on squared error (derived assuming 'fweight = 'fweight_norm').'''
        w_psb = w[t, :, :] @ np.diag(y_psb[t, :]) # select only columns corresponding to possible outcomes
        y_hat_alone = w_psb.T @ np.diag(fbase[t, :])
        comp_factor = fweight[t, :]**(sim_pars['metric'] - 1)
        y_hat_dif = y_hat_alone - np.outer(y_hat[t, :], comp_factor)
        atn_gain = self.data['atn'][t, :]*fbase[t, :]
        norm = sum(atn_gain**sim_pars['metric'])**(1/sim_pars['metric'])
        ngrad = delta[t, :].reshape((1, n_y)) @ y_hat_dif @ np.diag(fbase[t, :]/norm) # negative gradient
        self.data['atn'][t + 1, :] = np.maximum(self.data['atn'][t, :] + sim_pars['lrate_atn']*ngrad, n_f*[0.01])
        # Update feature counts.
        self.data['f_counts'][t, :] = np.apply_along_axis(np.sum, 0, fbase[0:(t+1), :] > 0)

    def add_data(self, ds):
        return ds.assign(atn = (['t', 'f_name'], self.data['atn'][range(self.n_t), :]),
                         f_counts = (['t', 'f_name'], self.data['f_counts']))   
gradcomp_elem_bias.par_names = ['lrate_atn', 'metric']

class gradcomp_feature_counts:
    '''
    Competitive attention learning from gradient descent (i.e. CompAct's learning rule),
    but also keeps track of feature counts.
    '''
    def __init__(self, sim_pars, n_t, n_x, n_f, n_y, f_names, x_dims):
        self.n_t = n_t
        self.data = {'atn': np.zeros((n_t + 1, n_f)), 'f_counts': np.zeros((n_t, n_f))}
        self.data['atn'][0, :] = 1

    def update(self, sim_pars, n_y, n_f, t, fbase, fweight, f_x, y_psb, y_hat, delta, w):
        # Update 'atn' by gradient descent on squared error (derived assuming 'fweight = 'fweight_norm').
        w_psb = w[t, :, :] @ np.diag(y_psb[t, :]) # select only columns corresponding to possible outcomes
        y_hat_alone = w_psb.T @ np.diag(fbase[t, :])
        comp_factor = fweight[t, :]**(sim_pars['metric'] - 1)
        y_hat_dif = y_hat_alone - np.outer(y_hat[t, :], comp_factor)
        atn_gain = self.data['atn'][t, :]*fbase[t, :]
        norm = sum(atn_gain**sim_pars['metric'])**(1/sim_pars['metric'])
        ngrad = delta[t, :].reshape((1, n_y)) @ y_hat_dif @ np.diag(fbase[t, :]/norm) # negative gradient
        self.data['atn'][t + 1, :] = np.maximum(self.data['atn'][t, :] + sim_pars['lrate_atn']*ngrad, n_f*[0.01])
        # Update feature counts.
        self.data['f_counts'][t, :] = np.apply_along_axis(np.sum, 0, fbase[0:(t+1), :] > 0)

    def add_data(self, ds):
        return ds.assign(atn = (['t', 'f_name'], self.data['atn'][range(self.n_t), :]),
                         f_counts = (['t', 'f_name'], self.data['f_counts']))   
gradcomp_feature_counts.par_names = ['lrate_atn', 'metric']

class gradcomp_atn_decay:
    '''
    Competitive attention learning from gradient descent (i.e. CompAct's learning rule)
    with decay of attention for features that are present on each trial.
    This is intended to produce latent inhibition and similar phenomena.
    
    Notes
    -----
    Preliminary simulations suggest that this does NOT produce latent inhibition,
    but further investigation is needed.
    '''
    def __init__(self, sim_pars, n_t, n_x, n_f, n_y, f_names, x_dims):
        self.n_t = n_t
        self.data = {'atn': np.zeros((n_t + 1, n_f))}
        self.data['atn'][0, :] = 1

    def update(self, sim_pars, n_y, n_f, t, fbase, fweight, f_x, y_psb, y_hat, delta, w):
        '''
        Update 'atn' by gradient descent on squared error (derived assuming 'fweight = 'fweight_norm')
        with decay of 'atn' for features that are present.
        '''
        w_psb = w[t, :, :] @ np.diag(y_psb[t, :]) # select only columns corresponding to possible outcomes
        y_hat_alone = w_psb.T @ np.diag(fbase[t, :])
        comp_factor = fweight[t, :]**(sim_pars['metric'] - 1)
        y_hat_dif = y_hat_alone - np.outer(y_hat[t, :], comp_factor)
        atn_gain = self.data['atn'][t, :]*fbase[t, :]
        norm = sum(atn_gain**sim_pars['metric'])**(1/sim_pars['metric'])
        ngrad = delta[t, :].reshape((1, n_y)) @ y_hat_dif @ np.diag(fbase[t, :]/norm) # negative gradient
        decay_term = sim_pars['drate_atn']*self.data['atn'][t, :]*fbase[t, :]
        self.data['atn'][t + 1, :] = np.maximum(self.data['atn'][t, :] + sim_pars['lrate_atn']*ngrad - decay_term, n_f*[0.01])

    def add_data(self, ds):
        return ds.assign(atn = (['t', 'f_name'], self.data['atn'][range(self.n_t), :]))   
gradcomp_atn_decay.par_names = ['lrate_atn', 'metric', 'drate_atn']

class Kalman:
    '''Kalman filter Rescorla-Wagner (Dayan & Kakade 2001, Gershman & Diedrichsen 2015).'''
    def __init__(self, sim_pars, n_t, n_x, n_f, n_y, f_names, x_dims):
        self.n_t = n_t
        self.data = {'gain' : np.zeros((n_t + 1, n_f, n_y)), 'Sigma' : np.zeros((n_t + 1, n_y, n_f, n_f))}
        for i in range(n_y):
            self.data['Sigma'][0, i, :, :] = sim_pars['w_var0']*np.identity(n_f) # initialize weight covariance matrix
    
    def update(self, sim_pars, n_y, n_f, t, fbase, fweight, f_x, y_psb, y_hat, delta, w):
        for i in range(n_y):
            f = fbase[t, :].reshape((n_f, 1))
            drift_mat = sim_pars['drift_var']*np.identity(n_f)
            numerator = ((self.data['Sigma'][t, i, :, :] + drift_mat)@f).squeeze()
            denominator = f.transpose()@numerator + sim_pars['y_var']
            self.data['gain'][t, :, i] = numerator/denominator
            self.data['Sigma'][t + 1, i, :, :] += drift_mat - self.data['gain'][t, :, i].reshape((n_f, 1))@f.transpose()@(self.data['Sigma'][t, i, :, :] + drift_mat) # update weight covariance matrix

    def add_data(self, ds):
        # possibly add Sigma and the diagonal of Sigma later
        return ds.assign(gain = (['t', 'f_name', 'y_name'], self.data['gain'][range(self.n_t), :, :])) 
Kalman.par_names = ['w_var0', 'y_var', 'drift_var']