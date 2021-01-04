import numpy as np
from scipy import stats
from scipy.linalg import solve

class linear:
    def __init__(self, n_u, n_f):
        pass
        
    def u_hat(self, z_hat, u_psb, f_x, hpar1_w):
        return z_hat
    
    def mean_z(self, z_hat, u, u_psb):
        return u
linear.par_names = ['u_var']
    
class probit:
    def __init__(self, n_u, n_f):
        self.n_u = n_u
        self.id_matrix = np.diag(np.ones(n_f)) # identity matrix
        
    def u_hat(self, z_hat, u_psb, f_x, hpar1_w):
        u_hat = np.zeros(self.n_u)
        for j in range(self.n_u):
            Sigma = solve(hpar1_w[:, :, j], self.id_matrix, assume_a = 'pos') # variance matrix (inverse of hpar1_w)
            pred_sd = np.sqrt(1 + np.inner(f_x@Sigma, f_x)) # predictive SD for u
            u_hat[j] = u_psb[j]*stats.norm.cdf(z_hat[j]/pred_sd)
        return u_hat
    
    def mean_z(self, z_hat, u, u_psb):
        mean_z = np.zeros(self.n_u)
        for j in range(self.n_u):
            calculate = u_psb[j] == 1
            if calculate:
                phi = stats.norm.pdf(-z_hat[j])
                PHI = stats.norm.cdf(-z_hat[j])
                if u[j] == 1:
                    mean_z[j] = z_hat[j] + phi/(1 - PHI)
                else:
                    mean_z[j] = z_hat[j] - phi/PHI
        return mean_z
probit.par_names = []
                    
# Eventually I should add a censored, i.e. tobit link as well.