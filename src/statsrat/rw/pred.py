import numpy as np

'''
Prediction functions.  Transforms sum_fw into y_hat.
Typically the identity function, but could be rectified positive.

identity: y_hat is simply equal to sum_fw.

rectified: y_hat equal to sum_fw if the latter is positive, or zero otherwise.
    This is suitable for conditioning experiments, in which negative values
    of the US are typically not possible.  Moreover, using this prediction function
    ensures that conditioned inhibitors do not suffer extinction, which is consistent
    with experimental results.

Notes
-----
sum_fw = y_psb[t, :]*(f_x[t, :]@w[t, :, :])
'''

def identity(sum_fw, sim_pars):
    '''
    y_hat is simply equal to sum_fw.
    '''
    return sum_fw
identity.par_names = []

def rectified(sum_fw, sim_pars):
    '''
    y_hat equal to sum_fw if the latter is positive, or zero otherwise.
    This is suitable for conditioning experiments, in which negative values
    of the US are typically not possible.  Moreover, using this prediction function
    ensures that conditioned inhibitors do not suffer extinction, which is consistent
    with experimental results.
    '''
    return np.clip(sum_fw, 0, 1)
rectified.par_names = []
    