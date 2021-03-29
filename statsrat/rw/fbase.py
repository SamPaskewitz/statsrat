import numpy as np
from itertools import combinations

########## BASE FEATURE FUNCTIONS ##########

# Elemental features.
def elem(x, x_names):
    f_x = x
    f_names = x_names.copy()
    output = {'f_x': f_x, 'f_names': f_names}
    return output
elem.par_names = []

# Elemental features with intercept term.
def elem_intercept(x, x_names):
    f_x = x
    f_names = x_names.copy()
    
    # add intercept feature
    new_feature = np.ones((f_x.shape[0], 1))
    f_x = np.concatenate((f_x, new_feature), axis = 1)
    f_names += ['intercept']
    
    output = {'f_x': f_x, 'f_names': f_names}
    return output
elem_intercept.par_names = []

# Binary configural features.
def cfg2(x, x_names):
    f_x = x
    f_names = x_names.copy()
    n_t = f_x.shape[0]
    n_f = f_x.shape[1]

    # loop through attribute combinations and add configural features as needed
    combs = list(combinations(f_names, 2)) # combinations of stimulus attributes (elemental cues)
    i_combs = list(combinations(range(n_f), 2))
    f_elem = f_x.copy()
    for i in range(len(combs)):
        new_name = combs[i][0] + '.' + combs[i][1] # name of new column (configural feature)
        new_feature = np.array(f_elem[:, i_combs[i][0]] * f_elem[:, i_combs[i][1]])
        new_needed = abs(sum(new_feature)) > 0 # only include new configural feature if the relevant combination actually shows up in the experiment
        if new_needed:
            f_x = np.concatenate((f_x, new_feature.reshape(n_t, 1)), axis = 1)
            f_names += [new_name]

    output = {'f_x': f_x, 'f_names': f_names}
    return output
cfg2.par_names = []

# Binary configural features with intercept term.
def cfg2_intercept(x, x_names):
    f_x = x
    f_names = x_names.copy()
    n_t = f_x.shape[0]
    n_f = f_x.shape[1]

    # loop through attribute combinations and add configural features as needed
    combs = list(combinations(f_names, 2)) # combinations of stimulus attributes (elemental cues)
    i_combs = list(combinations(range(n_f), 2))
    f_elem = f_x.copy()
    for i in range(len(combs)):
        new_name = combs[i][0] + '.' + combs[i][1] # name of new column (configural feature)
        new_feature = np.array(f_elem[:, i_combs[i][0]] * f_elem[:, i_combs[i][1]])
        new_needed = abs(sum(new_feature)) > 0 # only include new configural feature if the relevant combination actually shows up in the experiment
        if new_needed:
            f_x = np.concatenate((f_x, new_feature.reshape(n_t, 1)), axis = 1)
            f_names += [new_name]

    # add intercept feature
    new_feature = np.ones((f_x.shape[0], 1))
    f_x = np.concatenate((f_x, new_feature), axis = 1)
    f_names += ['intercept']

    output = {'f_x': f_x, 'f_names': f_names}
    return output
cfg2_intercept.par_names = []