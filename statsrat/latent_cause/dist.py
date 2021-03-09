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
    
    
    def a(self, eta):
        
        
    def b(self, obs):
        
    def c(self, tau, n):
        
            
    def T(self, obs):
        '''Sufficient statistic(s) of the observation.'''
        return obs
    
    
    def E_eta(self, tau, n):
        '''Expected natural parameters.'''
        
        # FINISH
        
    def E_a_eta(self, tau, n):
        
        
    def E_pred_mean(self, tau, n):
        '''Mean of the posterior predictive distribution.'''
    