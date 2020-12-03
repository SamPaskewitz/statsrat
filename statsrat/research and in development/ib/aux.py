import numpy as np

class nothing:
    def __init__(self, sim_pars, n_t, n_x, n_u):
        self.atn = np.array(n_x*[1])
    
    def update(self, sim_pars, n_u, n_x, t, x_mem, u_mem, u_psb, u_hat):
        self.data = None