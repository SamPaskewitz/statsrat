import numpy as np

class discrete:
    '''
    Bernoulli likelihood with beta prior.
    
    Notes
    -----
    The argument 'obs' refers to either 'u[t, :]' or 'x[t, :]' (depending on 
    whether the current distribution object is used to model outcomes ('u')
    or stimulus attributes ('x').
    '''
    def __init__(self):
        self.n_tau = 2 # number of hyperparameters (including one for "sample size")
        
    def suf_stat(self, obs):
        return obs
    
    def expected_obs(self, tau):
        return (tau[0] + 1)/(tau[1] + 2)
        # FINISH
    
    def expected_eta(self, tau):
        # FINISH
        
    def expected_a_eta(self, tau):
        # FINISH
        
    
    # UPDATE/DELETE
    def lik(self, obs, use):
        lik = np.zeros(self.max_z)
        for i in range(self.max_z):
            p = self.alpha[i, :]/(self.alpha[i, :] + self.beta[i, :]) # FIX THIS
            lik_obs = br.pmf(x_t, p)**use # likelihood of observation (if 'use' is 1) or else 1 (if 'use' is 0)
            lik[i] = np.prod(lik_obs)
        return lik
    
    def predict_mean(self):
        return self.p # FIX THIS
    
    def update(self, obs, post_xu):
        suf_stat = obs
        self.hpar += np.outer(post_xu, suf_stat) # update natural hyperparameter
        self.sample_size += post_xu # update sample size
        self.mean_par =  # mean natural parameter (E_q(eta) in Blei and Jordan's notation)

discrete.par_names = ['prior_discrete']

class normal:
    '''
    Normal likelihood with normal-gamma prior.
    
    Notes
    -----
    The argument 'obs' refers to either 'u[t, :]' or 'x[t, :]' (depending on 
    whether the current distribution object is used to model outcomes ('u')
    or stimulus attributes ('x').
    '''
    def __init__(self):
        self.n_tau = 3 # number of hyperparameters (including for "sample size")
        
    def suf_stat(self, obs):
        return np.array([obs, obs**2])
    
    def expected_obs(self, obs):
        # FINISH
    
    def expected_eta(self, tau):
        # FINISH
        
    def expected_a_eta(self, tau):
        # FINISH

discrete.par_names = ['prior_normal']