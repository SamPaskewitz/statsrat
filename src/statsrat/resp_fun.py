import numpy as np
from scipy.special import softmax
from scipy import stats

class choice:
    '''
    Discrete choices using a softmax function.

    Notes
    -----
    Response function:
    .. math:: \text{resp}_i = \frac{ e^{\phi \hat{y}_i} }{ \sum_j e^{\phi \hat{y}_j} }
     *REVISE*

    :math:`\phi` represents the 'resp_scale' parameter.
    '''
    def mean(y_hat, y_psb, resp_scale):
        '''
        Compute mean responses.
        '''
        foo = softmax(resp_scale*y_hat)
        bar = y_psb*foo
        return bar/bar.sum()
        
    def log_dist(y, b_hat, resp_scale):
        '''
        Compute log of the response distribution.
        '''
        return 0 # *FIX*
    
    def random(b_hat, resp_scale):
        '''
        Generate responses.
        '''
        n_t = b_hat.shape[0]
        n_y = b_hat.shape[1]
        rng = np.random.default_rng()
        b = np.zeros((n_t, n_y))
        b_index = np.zeros(n_t, dtype = 'int')
        for t in range(n_t):
            b_index[t] = rng.choice(n_y, p = b_hat[t, :])
            b[t, b_index[t]] = 1
        return (b, b_index)

class normal:
    '''
    Responses are normally distributed with mean y_hat.

    Notes
    -----
    Response function:
    .. math:: \text{resp}_i \sim N(\hat{y}_i, phi)

    :math:`\phi` represents the 'resp_scale' parameter.
    '''
    def mean(y_hat, y_psb, resp_scale):
        '''
        Compute mean responses.
        '''
        return y_hat
        
    def log_dist(y, b_hat, resp_scale):
        '''
        Compute log of the response distribution.
        '''
        return np.sum(stats.normal.logpdf(y, b_hat, resp_scale))
    
    def random(b_hat, resp_scale):
        '''
        Generate responses.
        '''
        n_t = b_hat.shape[0]
        n_y = b_hat.shape[1]
        b = stats.norm.rvs(loc = b_hat, scale = resp_scale, size = (n_t, n_y))
        b_index = None
        (b, b_index)
    
class exct:
    '''
    Excitatory responses for Pavlovian conditioning (logistic function).
    
    Notes
    -----
    Response function:
    .. math:: \text{resp} = \frac{ e^{\phi \hat{y}_i} }{ e^{\phi \hat{y}_i} + 1 }
     *REVISE*

    :math:`\phi` represents the 'resp_scale' parameter.
    '''
    def mean(y_hat, y_psb, resp_scale):
        '''
        Compute mean responses.
        '''
        return softmax(np.append(resp_scale*y_hat, 0))[0]
        
    def log_dist(y, b_hat, resp_scale):
        '''
        Compute log of the response distribution.
        '''
        return 0 # *FIX*
    
    def random(b_hat, resp_scale):
        '''
        Generate responses.
        '''
        return 0 # *FIX*
    
class supr:
    '''
    Suppression of ongoing behavior as a Pavlovian response, i.e. inhibitory conditioning (logistic function).
    
    Notes
    -----
    Response function:
    .. math:: \text{resp} = \frac{ e^{-\phi \hat{y}_i} }{ e^{-\phi \hat{y}_i} + 1 }
     *REVISE*

    :math:`\phi` represents the 'resp_scale' parameter.
    '''
    def mean(y_hat, y_psb, resp_scale):
        '''
        Compute mean responses.
        '''
        return softmax(np.append(-resp_scale*y_hat, 0))[0]
        
    def log_dist(y, b_hat, resp_scale):
        '''
        Compute log of the response distribution.
        '''
        return 0 # *FIX*
    
    def random(b_hat, resp_scale):
        '''
        Generate responses.
        '''
        return 0 # *FIX*

class log_normal:
    '''
    Log-normally distributed response.
    
    Notes
    -----
    .. math:: \log(y) \sim N(\phi \hat{y}_i, 1)
    *FIGURE THIS OUT*

    :math:`\phi` represents the 'resp_scale' parameter.
    '''
    def mean(y_hat, y_psb, resp_scale):
        '''
        Compute mean responses.
        '''
        return np.log(resp_scale*y_hat[0])
        
    def log_dist(y, b_hat, resp_scale):
        '''
        Compute log of the response distribution.
        '''
        return 0 # *FIX*
    
    def random(b_hat, resp_scale):
        '''
        Generate responses.
        '''
        b = b_hat + stats.norm.rvs(loc = 0, scale = 1, size = (n_t, n_y)) # *FIX*
        b_index = None