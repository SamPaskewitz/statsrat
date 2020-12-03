import numpy as np

########## GENERAL PURPOSE AUXILIARY LEARNING OBJECTS ##########

class basic:
    '''Basic auxiliary learning (keeps track of feature counts).'''
    def __init__(self, sim_pars, n_t, n_f, n_u):
        self.data = {'f_counts': np.zeros((n_t, n_f))}

    def update(self, sim_pars, n_u, n_f, t, fbase, fweight, u_psb, u_hat, delta, w):
        self.data['f_counts'][t, :] = np.apply_along_axis(np.sum, 0, fbase[0:(t+1), :] > 0)
basic.par_names = []
        
class drva:
    '''Derived attention.'''
    def __init__(self, sim_pars, n_t, n_f, n_u):
        self.data = {'atn': np.zeros((n_t, n_f))}

    def update(self, sim_pars, n_u, n_f, t, fbase, fweight, u_psb, u_hat, delta, w):
        new_atn = np.sum(abs(w[t, :]), axis = 1)
        abv_min = new_atn >= sim_pars['atn_min']
        new_atn = new_atn*abv_min + sim_pars['atn_min']*(1 - abv_min)
        self.data['atn'][t, :] = new_atn
drva.par_names = ['atn_min']

class grad:
    '''Non-competitive attention learning from gradient descent (i.e. simple predictiveness/Model 2).'''
    def __init__(self, sim_pars, n_t, n_f, n_u):
        self.data = {'atn': np.zeros((n_t + 1, n_f))}
        self.data['atn'][0, :] = 0.5

    def update(self, sim_pars, n_u, n_f, t, fbase, fweight, u_psb, u_hat, delta, w):
        '''Update 'atn' by gradient descent on squared error (derived assuming 'fweight = 'fweight_direct').'''
        w_psb = w[t, :, :] @ np.diag(u_psb[t, :]) # select only columns corresponding to possible outcomes
        ngrad = delta[t, :] @ w_psb.T @ np.diag(fbase[t, :]) # negative gradient
        new_atn = self.data['atn'][t, :] + sim_pars['lrate_atn'] * ngrad
        abv_min = new_atn >= 0.01
        blw_max = new_atn < 1
        new_atn = new_atn*abv_min*blw_max + 0.01*(1 - abv_min) + 1*(1 - blw_max)
        self.data['atn'][t + 1, :] = new_atn
grad.par_names = ['lrate_atn']

class gradcomp:
    '''Competitive attention learning from gradient descent (i.e. CompAct's learning rule).'''
    def __init__(self, sim_pars, n_t, n_f, n_u):
        self.data = {'atn': np.zeros((n_t + 1, n_f))}
        self.data['atn'][0, :] = 1

    def update(self, sim_pars, n_u, n_f, t, fbase, fweight, u_psb, u_hat, delta, w):
        '''Update 'atn' by gradient descent on squared error (derived assuming 'fweight = 'fweight_norm').'''
        w_psb = w[t, :, :] @ np.diag(u_psb[t, :]) # select only columns corresponding to possible outcomes
        u_hat_alone = w_psb.T @ np.diag(fbase[t, :])
        comp_factor = fweight[t, :]**(sim_pars['metric'] - 1)
        u_hat_dif = u_hat_alone - np.outer(u_hat[t, :], comp_factor)
        atn_gain = self.data['atn'][t, :]*fbase[t, :]
        norm = sum(atn_gain**sim_pars['metric'])**(1/sim_pars['metric'])
        ngrad = delta[t, :].reshape((1, n_u)) @ u_hat_dif @ np.diag(fbase[t, :]/norm) # negative gradient
        self.data['atn'][t + 1, :] = self.data['atn'][t, :] + sim_pars['lrate_atn'] * ngrad
gradcomp.par_names = ['lrate_atn', 'metric']

class Kalman:
    '''Kalman filter Rescorla-Wagner (Dayan & Kakade 2001, Gershman & Diedrichsen 2015).'''
    def __init__(self, sim_pars, n_t, n_f, n_u):
        self.data = {'gain' : np.zeros((n_t + 1, n_f, n_u)), 'Sigma' : np.zeros((n_t + 1, n_u, n_f, n_f))}
        for i in range(n_u):
            self.data['Sigma'][0, i, :, :] = sim_pars['w_var0']*np.identity(n_f) # initialize weight covariance matrix
    def update(self, sim_pars, n_u, n_f, t, fbase, fweight, u_psb, u_hat, delta, w):
        for i in range(n_u):
            f = fbase[t, :].reshape((n_f, 1))
            drift_mat = sim_pars['drift_var']*np.identity(n_f)
            numerator = ((self.data['Sigma'][t, i, :, :] + drift_mat)@f).squeeze()
            denominator = f.transpose()@numerator + sim_pars['u_var']
            self.data['gain'][t, :, i] = numerator/denominator
            self.data['Sigma'][t + 1, i, :, :] += drift_mat - self.data['gain'][t, :, i].reshape((n_f, 1))@f.transpose()@(self.data['Sigma'][t, i, :, :] + drift_mat) # update weight covariance matrix
Kalman.par_names = ['w_var0', 'u_var', 'drift_var']
                   
##### FREE INITIAL ATTENTION PARAMETER #####
# For now, this assumes that there's an even number of features
# with the first half in one group (stimulus dimension) and the second half in another.
# My goal in creating this is to model the FAST (face task) and similar designs.
# Later I should probably make this more flexible.
# AS OF 8/5/2020 NEEDS TO BE UPDATED.

class grad_a0:
    '''Non-competitive attention learning with one free initial attention parameter.'''
    def __init__(self, sim_pars, n_t, n_f, n_u):
        self.data = {'atn': np.zeros((n_t + 1, n_f))}
        n_half = np.int_(n_f/2)
        self.data['atn'][0, range(n_half)] = 0.5
        self.data['atn'][0, range(n_half + 1, n_f)] = sim_pars['a0']

    def update(self, sim_pars, n_u, n_f, t, fbase, fweight, u_psb, u_hat, delta, w):
        '''Update 'atn' by gradient descent on squared error (derived assuming 'fweight = 'fweight_direct').'''
        w_psb = w[t, :, :] @ np.diag(u_psb[t, :]) # select only columns corresponding to possible outcomes
        ngrad = delta[t, :] @ w_psb.T @ np.diag(fbase[t, :]) # negative gradient
        new_atn = self.data['atn'][t, :] + sim_pars['lrate_atn'] * ngrad
        abv_min = new_atn >= 0.01
        blw_max = new_atn < 1
        new_atn = new_atn*abv_min*blw_max + 0.01*(1 - abv_min) + 1*(1 - blw_max)
        self.data['atn'][t + 1, :] = new_atn
grad_a0.par_names = ['lrate_atn']

class gradcomp_eta0:
    '''Competitive attention learning with one free initial attention parameter.'''
    def __init__(self, sim_pars, n_t, n_f, n_u):
        self.data = {'atn': np.zeros((n_t + 1, n_f))}
        n_half = np.int_(n_f/2)
        self.data['atn'][0, range(n_half)] = 1
        self.data['atn'][0, range(n_half + 1, n_f)] = sim_pars['eta0']

    def update(self, sim_pars, n_u, n_f, t, fbase, fweight, u_psb, u_hat, delta, w):
        '''Update 'atn' by gradient descent on squared error (derived assuming 'fweight = 'fweight_norm').'''
        w_psb = w[t, :, :] @ np.diag(u_psb[t, :]) # select only columns corresponding to possible outcomes
        u_hat_alone = w_psb.T @ np.diag(fbase[t, :])
        comp_factor = fweight[t, :]**(sim_pars['metric'] - 1)
        u_hat_dif = u_hat_alone - np.outer(u_hat[t, :], comp_factor)
        atn_gain = self.data['atn'][t, :]*fbase[t, :]
        norm = sum(atn_gain**sim_pars['metric'])**(1/sim_pars['metric'])
        ngrad = delta[t, :].reshape((1, n_u)) @ u_hat_dif @ np.diag(fbase[t, :]/norm) # negative gradient
        self.data['atn'][t + 1, :] = self.data['atn'][t, :] + sim_pars['lrate_atn'] * ngrad
gradcomp_eta0.par_names = ['lrate_atn', 'metric']
