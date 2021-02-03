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

class multinomial_probit:
    def __init__(self, n_u, n_f):
        self.n_u = n_u
        
    def u_hat(self, z_hat, u_psb, f_x, hpar1_w):
        # This isn't the real predictive distribution for now, but whatever.
        u_hat = np.ones(self.n_u)
        for j in range(self.n_u):
            for k in range(self.n_u):
                if not k == j:
                    u_hat[j] *= stats.norm.cdf((z_hat[j] - z_hat[k])/np.sqrt(2))
        return u_hat
        
    def mean_z(self, z_hat, u, u_psb):
        # Run a little coordinate ascent iteration to get variational means.
        mean_z = u_psb*z_hat # initialize to the means of the non-truncated distributions (as a first guess)
        winner = np.argmax(u) # index of the outcome, i.e. of the z that must be the largest
        for i in range(3):
            # Update variational means of the other outcomes.
            for j in range(self.n_u):
                calculate = u_psb[j] == 1
                if (not j == winner) and calculate:
                    phi = stats.norm.pdf(mean_z[winner] - z_hat[j])
                    PHI = stats.norm.cdf(mean_z[winner] - z_hat[j])
                    mean_z[j] = z_hat[j] - phi/PHI
            # Update variational mean of the winner (i.e. observed outcome).
            z_star = np.delete(mean_z, winner).max() # biggest mean_z after the winner
            phi = stats.norm.pdf(z_star - z_hat[winner])
            PHI = stats.norm.cdf(z_star - z_hat[winner])
            mean_z[winner] = z_hat[winner] + phi/(1 - PHI)
        return mean_z
multinomial_probit.par_names = []                
        
                    
# Eventually I should add a censored, i.e. tobit link as well.