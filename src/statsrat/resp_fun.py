import numpy as np
from scipy.special import softmax
from scipy import stats

def choice(y_hat, y_psb, resp_scale):
    '''
    Response probabilities for discrete choices using a softmax function.
    
    Notes
    -----
    Response function:
    .. math:: \text{resp}_i = \frac{ e^{\phi \hat{u}_i} }{ \sum_j e^{\phi \hat{u}_j} }

    :math:`\phi` represents the 'resp_scale' parameter.
    '''
    foo = softmax(resp_scale*y_hat)
    bar = y_psb*foo
    return bar/bar.sum()
      
def exct(y_hat, y_psb, resp_scale):
    '''
    Excitatory responses for Pavlovian conditioning (logistic function).
    
    Notes
    -----
    Response function:
    .. math:: \text{resp} = \frac{ e^{\phi \hat{u}_i} }{ e^{\phi \hat{u}_i} + 1 }

    :math:`\phi` represents the 'resp_scale' parameter.
    '''
    return softmax(np.append(resp_scale*y_hat, 0))[0]
    
def supr(y_hat, y_psb, resp_scale):
    '''
    Suppression of ongoing behavior as a Pavlovian response, i.e. inhibitory conditioning (logistic function).
    
    Notes
    -----
    Response function:
    .. math:: \text{resp} = \frac{ e^{-\phi \hat{u}_i} }{ e^{-\phi \hat{u}_i} + 1 }

    :math:`\phi` represents the 'resp_scale' parameter.
    '''
    return softmax(np.append(-resp_scale*y_hat, 0))[0]

def generate_responses(b_hat, random_resp, resp_type):
    '''
    Generate simulated responses.
    '''
    n_t = b_hat.shape[0]
    n_y = b_hat.shape[1]
    if random_resp is False:
        b = b_hat
        b_index = None
    else: 
        if resp_type == 'choice':
            rng = np.random.default_rng()
            b = np.zeros((n_t, n_y))
            b_index = np.zeros(n_t, dtype = 'int')
            for t in range(n_t):
                b_index[t] = rng.choice(n_y, p = b_hat[t, :])
                b[t, b_index[t]] = 1
        else:
            b = b_hat + stats.norm.rvs(loc = 0, scale = 0.01, size = (n_t, n_y))
            b_index = None
    return (b, b_index)