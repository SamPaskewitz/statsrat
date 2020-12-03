import numpy as np
from scipy.special import softmax

def choice(u_hat, u_psb, resp_scale):
    foo = softmax(resp_scale*u_hat)
    bar = u_psb*foo
    return bar/bar.sum()
      
def exct(u_hat, u_psb, resp_scale):
    return softmax(np.append(resp_scale*u_hat, 0))[0]
    
def supr(u_hat, u_psb, resp_scale):
    return softmax(np.append(-resp_scale*u_hat, 0))[0]