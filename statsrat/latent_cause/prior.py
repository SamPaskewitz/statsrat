import numpy as np

def Chinese_rest(sim_pars, t, z, z_counts, max_z):
    '''
    Chinese restaurant process.
    See Gershman, Blei and Niv (2010).
    '''
    prob = np.zeros(max_z)
    if t > 0:
        n_z = (z_counts > 0).sum() # number of latent causes inferred so far
        prob[range(n_z)] = z_counts[range(n_z)]/(t + sim_pars['alpha']) # probability of old causes
        if n_z < max_z:
            prob[n_z] = sim_pars['alpha']/(t + sim_pars['alpha']) # probability of a new cause
    else:
        prob[0] = 1 # the initial observation is automatically assigned to the initial latent cause
    return prob

Chinese_rest.par_names = ['alpha']

def power_law(sim_pars, t, z, z_counts, max_z):
    '''
    Time-sensitive Chinese restaurant process with power law kernel.
    See Gershman, Monfils, Norman and Niv (2017), Equations 3 and 4.
    '''
    prob = np.zeros(max_z)
    if t > 0:
        n_z = (z_counts > 0).sum() # number of latent causes inferred so far
        numerator = np.zeros(max_z)
        adj_counts = np.zeros(max_z)
        for s in range(t):
            adj_counts[z[s] == 1] += 1/(t - s)
        numerator[range(n_z)] = adj_counts[range(n_z)] # probability of old causes
        if n_z < max_z:
            numerator[n_z] = sim_pars['alpha']
        prob = numerator/numerator.sum()
    else:
        prob[0] = 1 # the initial observation is automatically assigned to the initial latent cause
    return prob

power_law.par_names = ['alpha']