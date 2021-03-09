import numpy as np
from scipy.stats import bernoulli as br
from scipy.stats import t as tdist

class discrete_hard:
    '''
    Bernoulli likelihood updated with hard latent cause assignment (c.f. Anderson 1991).
    '''
    def __init__(self, sim_pars, n_t, n_x, max_z):
        self.alpha = np.array(max_z*n_x*[sim_pars['beta0']], dtype = 'float64').reshape((max_z, n_x))
        self.beta = np.array(max_z*n_x*[sim_pars['beta0']], dtype = 'float64').reshape((max_z, n_x))
        self.max_z = max_z
        self.n_x = n_x
        
    def lik(self, sim_pars, t, x_t):
        prob = np.zeros(self.max_z)
        for i in range(self.max_z):
            p = self.alpha[i, :]/(self.alpha[i, :] + self.beta[i, :])
            prob[i] = np.prod(br.pmf(x_t, p))
        return prob
        
    def update(self, sim_pars, t, x_t, post_xu, z_counts, z_t):
        self.alpha[z_t == 1, :] += x_t
        self.beta[z_t == 1, :] += 1 - x_t

discrete_hard.par_names = ['beta0']
        
class normal_hard:
    '''
    Normal likelihood updated with hard latent cause assignment (c.f. Anderson 1991).
    The prior variance of each component of x is set at 0.125 (the square of 1/4 of each component's typical
    range, which is from 0 to 1).  The prior mean of each component of x is set at 0.
    '''
    def __init__(self, sim_pars, n_t, n_x, max_z):
        self.lmb = np.array(max_z*n_x*[sim_pars['lmb0']], dtype = 'float64').reshape((max_z, n_x)) # sigma prior confidence
        self.a = np.array(max_z*n_x*[sim_pars['a0']], dtype = 'float64').reshape((max_z, n_x)) # mu prior confidence
        self.mu = np.zeros((max_z, n_x)) # prior mean (set to 0)
        self.sigmasq = np.array(max_z*n_x*[0.125]).reshape((max_z, n_x)) # prior variance (set to 0.125)
        self.x_bar = np.zeros((max_z, n_x)) # empirical means
        self.xsq_bar = np.zeros((max_z, n_x)) # empirical means of squared observations
        self.ssq = np.zeros((max_z, n_x)) # empirical vars
        self.n = np.zeros(max_z)
        self.n_x = n_x
        self.max_z = max_z
    
    def lik(self, sim_pars, t, x_t):
        prob = np.zeros(self.max_z)
        for i in range(self.max_z):
            lik_var = self.sigmasq[i, :]*(1 + 1/self.lmb[i, :])
            indv_lik = tdist.pdf(x_t, df = self.a[i, :], loc = self.mu[i, :], scale = np.sqrt(lik_var))
            prob[i] = np.prod(indv_lik)
        return prob
        
    def update(self, sim_pars, t, x_t, post_xu, z_counts, z_t):
        index = z_t == 1
        self.n[index] += 1
        self.x_bar[index, :] += (1/self.n[index])*(x_t - self.x_bar[index, :]) # compute empirical mean recursively
        self.xsq_bar[index, :] += (1/self.n[index])*(x_t**2 - self.xsq_bar[index, :]) # compute empirical mean of squared observations recursively
        self.ssq[index, :] = self.n[index]*(self.xsq_bar[index, :] - self.x_bar[index, :]**2) # empirical sum of squares
        self.lmb[index, :] += 1
        self.a[index, :] += 1
        numerator = self.n[index]*self.x_bar[index, :]
        denominator = sim_pars['lmb0'] + self.n[index]
        self.mu[index, :] = numerator/denominator
        ratio = sim_pars['lmb0']*self.n[index]/(sim_pars['lmb0'] + self.n[index])
        self.sigmasq[index, :] = (sim_pars['a0']*0.125 + self.ssq[index, :] + ratio*self.x_bar[index, :]**2)/self.a[index, :]

normal_hard.par_names = ['lmb0', 'a0']