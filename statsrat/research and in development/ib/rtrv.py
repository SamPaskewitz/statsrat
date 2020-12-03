import numpy as np
from scipy.spatial import distance

# https://docs.scipy.org/doc/scipy/reference/spatial.distance.html

# CHECK THIS.

def eucl(x, x_mem, aux, t, n_mem, n_t, sim_pars):
    '''Retrieval strength (similarity) is an exponential function of distance
    (where distance is based on the Euclidean metric).'''
    dist = np.apply_along_axis(func1d = lambda xm: distance.minkowski(u = xm, v = x, p = 2, w = aux.atn),
                               axis = 0,
                               arr = x_mem)
    rtrv = np.zeros(n_t)
    print(x_mem)
    print(dist)
    rtrv[0:n_mem] = np.exp(-sim_pars['c']*dist)
    return rtrv