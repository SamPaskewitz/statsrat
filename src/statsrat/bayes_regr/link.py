import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import solve

'''
Specifies link functions (in the form of classes) for Bayesian regression learning models.

linear: Suitable for real valued outcomes.

probit: Suitable for binary outcomes.

multinomial_probit: Suitable for categorical outcomes.
'''

class linear:
    '''
    Suitable for real valued outcomes.
    '''
    def __init__(self, n_y, n_f):
        pass
        
    def y_hat(self, z_hat, y_psb, f_x, hpar1_w):
        return z_hat
    
    def mean_z(self, z_hat, y, y_psb):
        return y
linear.pars = pd.DataFrame({'min' : 0.0, 'max' : 10.0, 'default' : 0.1}, index = ['y_var']) # outcome variance
    
class probit:
    '''
    Suitable for binary outcomes.
    '''
    def __init__(self, n_y, n_f):
        self.n_y = n_y
        self.id_matrix = np.diag(np.ones(n_f)) # identity matrix
        
    def y_hat(self, z_hat, y_psb, f_x, hpar1_w):
        y_hat = np.zeros(self.n_y)
        for j in range(self.n_y):
            Sigma = solve(hpar1_w[:, :, j], self.id_matrix, assume_a = 'pos') # variance matrix (inverse of hpar1_w)
            pred_sd = np.sqrt(1 + np.inner(f_x@Sigma, f_x)) # predictive SD for u
            y_hat[j] = y_psb[j]*stats.norm.cdf(z_hat[j]/pred_sd)
        return y_hat
    
    def mean_z(self, z_hat, y, y_psb):
        mean_z = np.zeros(self.n_y)
        for j in range(self.n_y):
            calculate = y_psb[j] == 1
            if calculate:
                phi = stats.norm.pdf(-z_hat[j])
                PHI = stats.norm.cdf(-z_hat[j])
                if y[j] == 1:
                    mean_z[j] = z_hat[j] + phi/(1 - PHI)
                else:
                    mean_z[j] = z_hat[j] - phi/PHI
        return mean_z
probit.pars = None

class multinomial_probit:
    '''
    Suitable for categorical outcomes.
    '''
    def __init__(self, n_y, n_f):
        self.n_y = n_y
        
    def y_hat(self, z_hat, y_psb, f_x, hpar1_w):
        # This isn't the real predictive distribution for now, but whatever.
        y_hat = np.ones(self.n_y)
        for j in range(self.n_y):
            for k in range(self.n_y):
                if not k == j:
                    y_hat[j] *= stats.norm.cdf((z_hat[j] - z_hat[k])/np.sqrt(2))
        return y_hat
        
    def mean_z(self, z_hat, y, y_psb):
        # Run a little coordinate ascent iteration to get variational means.
        mean_z = y_psb*z_hat # initialize to the means of the non-truncated distributions (as a first guess)
        winner = np.argmax(y) # index of the outcome, i.e. of the z that must be the largest
        for i in range(3):
            # Update variational means of the other outcomes.
            for j in range(self.n_y):
                calculate = y_psb[j] == 1
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
multinomial_probit.pars = None                
        
                    
# Eventually I should add a censored, i.e. tobit link as well.