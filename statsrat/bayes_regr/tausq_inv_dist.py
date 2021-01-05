import numpy as np
from scipy import stats

class constant:
    '''
    Prior precision (tausq_inv) is treated as constant, i.e.
    there is no attempt to change the initial hyperparameter values.
    '''
    def __init__(self, n_u, n_f, sim_pars):
        self.tausq_inv_array = np.array(n_u*n_f*[sim_pars['tausq_inv']]).reshape((n_f, n_u))
    
    def update(self, mean_wsq, u_psb):
        pass # do nothing, because tausq_inv is assumed to be known and constant
    
    def mean_tausq_inv(self):
        return self.tausq_inv_array
    
    def mean_tausq(self):
        return 1/self.tausq_inv_array

constant.par_names = ['tausq_inv']

class ard:
    '''
    Automatic relevance determination, i.e. the model tries
    to learn the distribution of tausq_inv via variational Bayes
    (assuming that tausq_inv has a gamma distribution).
    '''
    def __init__(self, n_u, n_f, sim_pars):
        self.n_u = n_u
        self.prior_hpar0 = sim_pars['prior_tausq_inv_hpar0']
        self.prior_hpar1 = sim_pars['prior_tausq_inv_hpar1']
        self.hpar0 = np.array(n_f*n_u*[sim_pars['prior_tausq_inv_hpar0']], dtype='float').reshape((n_f, n_u))
        self.hpar1 = sim_pars['prior_tausq_inv_hpar1']
        self.u_psb_so_far = np.zeros(n_u)
        
    def update(self, mean_wsq, u_psb):
        # keep track of which outcomes have been observed so far
        for j in range(self.n_u):
            if u_psb[j] == 1:
                self.u_psb_so_far[j] = 1
        # update hyperparameters
        for j in range(self.n_u):
            self.hpar0[:, j] = self.prior_hpar0 - 0.5*mean_wsq[:, j]
        self.hpar1 = self.prior_hpar1 + 0.5*self.u_psb_so_far.sum()
    
    def mean_tausq_inv(self):
        return (self.hpar1 + 1)/(-self.hpar0)
    
    def mean_tausq(self):
        return -self.hpar0/self.hpar1

ard.par_names = ['prior_tausq_inv_hpar0', 'prior_tausq_inv_hpar1']

class ard_drv_atn:
    '''
    Automatic relevance determination (based on the gamma distribution)
    with the assumption that all of the regression weights (w) associated
    with a feature share a common prior prevision (tausq_inv).  This ends
    up being a form of derived attention model.
    '''
    def __init__(self, n_u, n_f, sim_pars):
        self.n_u = n_u
        self.n_f = n_f
        self.prior_hpar0 = sim_pars['prior_tausq_inv_hpar0']
        self.prior_hpar1 = sim_pars['prior_tausq_inv_hpar1']
        self.hpar0 = np.array(n_f*[sim_pars['prior_tausq_inv_hpar0']], dtype='float')
        self.hpar1 = sim_pars['prior_tausq_inv_hpar1']
        self.u_psb_so_far = np.zeros(n_u)
        
    def update(self, mean_wsq, u_psb):
        # keep track of which outcomes have been observed so far
        for j in range(self.n_u):
            if u_psb[j] == 1:
                self.u_psb_so_far[j] = 1
        # update hyperparameters
        self.hpar0 = self.prior_hpar0 - 0.5*mean_wsq.sum(1)
        self.hpar1 = self.prior_hpar1 + 0.5*self.u_psb_so_far.sum()
    
    def mean_tausq_inv(self):
        mean_tausq_inv = np.zeros((self.n_f, self.n_u))
        for i in range(self.n_f):
            mean_tausq_inv[i, :] = (self.hpar1 + 1)/(-self.hpar0[i])
        return mean_tausq_inv
    
    def mean_tausq(self):
        mean_tausq = np.zeros((self.n_f, self.n_u))
        for i in range(self.n_f):
            mean_tausq[i, :] = -self.hpar0[i]/self.hpar1
        return mean_tausq
    
ard_drv_atn.par_names = ['prior_tausq_inv_hpar0', 'prior_tausq_inv_hpar1']