import numpy as np
from scipy.special import softmax

'''
Response functions that transform outcome predictions (u_hat)
into observed behavior.

choice: Response probabilities for discrete choices using a softmax function.

exct: Excitatory responses for Pavlovian conditioning (logistic function).

supr: Suppression of ongoing behavior as a Pavlovian response, i.e. inhibitory conditioning (logistic function).
'''

def choice(u_hat, u_psb, resp_scale):
    '''
    Response probabilities for discrete choices using a softmax function.
    
    Notes
    -----
    Response function:
    .. math:: \text{resp}_i = \frac{ e^{\phi \hat{u}_i} }{ \sum_j e^{\phi \hat{u}_j} }

    :math:`\phi` represents the 'resp_scale' parameter.
    '''
    foo = softmax(resp_scale*u_hat)
    bar = u_psb*foo
    return bar/bar.sum()
      
def exct(u_hat, u_psb, resp_scale):
    '''
    Excitatory responses for Pavlovian conditioning (logistic function).
    
    Notes
    -----
    Response function:
    .. math:: \text{resp} = \frac{ e^{\phi \hat{u}_i} }{ e^{\phi \hat{u}_i} + 1 }

    :math:`\phi` represents the 'resp_scale' parameter.
    '''
    return softmax(np.append(resp_scale*u_hat, 0))[0]
    
def supr(u_hat, u_psb, resp_scale):
    '''
    Suppression of ongoing behavior as a Pavlovian response, i.e. inhibitory conditioning (logistic function).
    
    Notes
    -----
    Response function:
    .. math:: \text{resp} = \frac{ e^{-\phi \hat{u}_i} }{ e^{-\phi \hat{u}_i} + 1 }

    :math:`\phi` represents the 'resp_scale' parameter.
    '''
    return softmax(np.append(-resp_scale*u_hat, 0))[0]