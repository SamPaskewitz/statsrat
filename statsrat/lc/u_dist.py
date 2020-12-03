import numpy as np
from scipy.stats import bernoulli as br
from scipy.stats import t as tdist

class discrete_hard:
    '''
    Bernoulli likelihood updated with hard latent cause assignment (c.f. Anderson 1991).
    '''
    def __init__(self, sim_pars, n_t, n_u, max_z):
        self.alpha = np.array(max_z*n_u*[sim_pars['beta0']], dtype = 'float64').reshape((max_z, n_u))
        self.beta = np.array(max_z*n_u*[sim_pars['beta0']], dtype = 'float64').reshape((max_z, n_u))
        self.max_z = max_z
        self.n_u = n_u
    
    def predict(self, sim_pars, t, x_t, u_psb_t, post_x):
        p = np.zeros((self.max_z, self.n_u))
        for i in range(self.max_z):
            p[i, u_psb_t == 1] = self.alpha[i, :]/(self.alpha[i, :] + self.beta[i, :])
        u_hat = post_x@p
        return u_hat
    
    def lik(self, sim_pars, t, u_t, x_t, u_psb_t):
        prob = np.zeros(self.max_z)
        for i in range(self.max_z):
            p = np.ones(self.n_u)
            p[u_psb_t == 1] = self.alpha[i, :]/(self.alpha[i, :] + self.beta[i, :])
            prob[i] = np.prod(br.pmf(u_t, p))
        return prob
        
    def update(self, sim_pars, t, u_t, x_t, post_xu, z_counts, z_t):
        # ADD EFFECT OF u_lrn
        self.alpha[z_t == 1, :] += u_t
        self.beta[z_t == 1, :] += 1 - u_t

discrete_hard.par_names = ['beta0']
        
class normal_hard:
    '''
    Normal likelihood updated with hard latent cause assignment (c.f. Anderson 1991).
    The prior variance of each component of u is set at 0.125 (the square of 1/4 of each component's typical
    range, which is from 0 to 1).  The prior mean is set at 0.
    '''
    def __init__(self, sim_pars, n_t, n_u, max_z):
        self.lmb = np.array(max_z*n_u*[sim_pars['lmb0']], dtype = 'float64').reshape((max_z, n_u)) # sigma prior confidence
        self.a = np.array(max_z*n_u*[sim_pars['a0']], dtype = 'float64').reshape((max_z, n_u)) # mu prior confidence
        self.mu = np.zeros((max_z, n_u)) # prior mean (set to zero)
        self.sigmasq = np.array(max_z*n_u*[0.125]).reshape((max_z, n_u)) # prior variance (set to 0.125)
        self.u_bar = np.zeros((max_z, n_u)) # empirical means
        self.usq_bar = np.zeros((max_z, n_u)) # empirical means of squared observations
        self.ssq = np.zeros((max_z, n_u)) # empirical vars
        self.n = np.zeros(max_z)
        self.n_u = n_u
        self.max_z = max_z
    
    def predict(self, sim_pars, t, x_t, u_psb_t, post_x):
        u_hat_z = self.mu*u_psb_t
        u_hat = post_x@u_hat_z
        return u_hat
    
    def lik(self, sim_pars, t, u_t, x_t, u_psb_t):
        prob = np.zeros(self.max_z)
        for i in range(self.max_z):
            lik_var = self.sigmasq[i, :]*(1 + 1/self.lmb[i, :])
            indv_lik = tdist.pdf(u_t, df = self.a[i, :], loc = self.mu[i, :], scale = np.sqrt(lik_var))
            prob[i] = np.prod(indv_lik)
        return prob
        
    def update(self, sim_pars, t, u_t, x_t, post_xu, z_counts, z_t):
        # ADD EFFECT OF u_lrn
        index = z_t == 1
        self.n[index] += 1
        self.u_bar[index, :] += (1/self.n[index])*(u_t - self.u_bar[index, :]) # compute empirical mean recursively
        self.usq_bar[index, :] += (1/self.n[index])*(u_t**2 - self.usq_bar[index, :]) # compute empirical mean of squared observations recursively
        self.ssq[index, :] = self.n[index]*(self.usq_bar[index, :] - self.u_bar[index, :]**2) # empirical sum of squares
        self.lmb[index, :] += 1
        self.a[index, :] += 1
        numerator = self.n[index]*self.u_bar[index, :]
        denominator = sim_pars['lmb0'] + self.n[index]
        self.mu[index, :] = numerator/denominator
        ratio = sim_pars['lmb0']*self.n[index]/(sim_pars['lmb0'] + self.n[index])
        self.sigmasq[index, :] = (sim_pars['a0']*0.125 + self.ssq[index, :] + ratio*self.u_bar[index, :]**2)/self.a[index, :]
        
normal_hard.par_names = ['lmb0', 'a0']