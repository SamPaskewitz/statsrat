import numpy as np
import pandas as pd
from itertools import combinations

def elem(x, x_names, sim_pars):
    '''
    Elemental features.
    '''
    f_x = x
    f_names = x_names.copy()
    output = {'f_x': f_x, 'f_names': f_names}
    return output
elem.pars = None

def elem_intercept(x, x_names, sim_pars):
    '''
    Elemental features with intercept term.
    '''
    f_x = x
    f_names = x_names.copy()
    
    # add intercept feature
    new_feature = np.ones((f_x.shape[0], 1))
    f_x = np.concatenate((f_x, new_feature), axis = 1)
    f_names += ['intercept']
    
    output = {'f_x': f_x, 'f_names': f_names}
    return output
elem_intercept.pars = None

def cfg2(x, x_names, sim_pars):
    '''
    Binary configural features.
    '''
    f_x = x
    f_names = x_names.copy()
    n_t = f_x.shape[0]
    n_f = f_x.shape[1]

    # loop through attribute combinations and add configural features as needed
    combs = list(combinations(f_names, 2)) # combinations of stimulus attributes (elemental cues)
    i_combs = list(combinations(range(n_f), 2))
    f_elem = f_x.copy()
    for i in range(len(combs)):
        new_feature = np.array(f_elem[:, i_combs[i][0]] * f_elem[:, i_combs[i][1]])
        new_needed = abs(sum(new_feature)) > 0 # only include new configural feature if the relevant combination actually shows up in the experiment
        if new_needed:
            f_x = np.concatenate((f_x, new_feature.reshape(n_t, 1)), axis = 1)
            f_names += [combs[i][0] + '.' + combs[i][1]] # name of new column (configural feature)

    output = {'f_x': f_x, 'f_names': f_names}
    return output
cfg2.pars = None

def cfg2_gradual(x, x_names, sim_pars):
    '''
    Binary configural features.  The strength of each configural feature starts at 0,
    and gradually increases every time the relevant cue pair is observed.
    
    Notes
    -----
    .. math:: f_{i, j}(x) = x_i x_j \frac{1}{1 + \exp(-(n_{i, j} - k))}
    where :math:`f_{i, j}(x)` is the configural feature for cues i and j,
    :math:`n_{i, j}` is the number of times features i and j have been observed together,
    and :math:`k` determines how long it takes for configural features to emerge (it is
    called 'cfg_emergence_par' in the code).
    '''
    f_x = x
    f_names = x_names.copy()
    n_t = f_x.shape[0]
    n_f = f_x.shape[1]

    # loop through attribute combinations and add configural features as needed
    combs = list(combinations(f_names, 2)) # combinations of stimulus attributes (elemental cues)
    i_combs = list(combinations(range(n_f), 2))
    f_elem = f_x.copy()
    for i in range(len(combs)):
        elem_product = np.array(f_elem[:, i_combs[i][0]] * f_elem[:, i_combs[i][1]])
        new_needed = abs(sum(elem_product)) > 0 # only include new configural feature if the relevant combination actually shows up in the experiment
        if new_needed:
            n = 0
            new_feature = np.zeros(n_t)
            for t in range(n_t):
                new_feature[t] = elem_product[t]/(1 + np.exp(-(n - sim_pars['cfg_emergence_par'])))
                n += elem_product[t]
            f_x = np.concatenate((f_x, new_feature.reshape(n_t, 1)), axis = 1)
            f_names += [combs[i][0] + '.' + combs[i][1]] # name of new column (configural feature)

    output = {'f_x': f_x, 'f_names': f_names}
    return output
cfg2_gradual.pars = pd.DataFrame({'min': 0.0, 'max': 20.0, 'default': 5.0}, index = ['cfg_emergence_par']) # controls how rapidly configural features emerge

def cfg2_intercept(x, x_names, sim_pars):
    '''
    Binary configural features with intercept term.
    '''
    f_x = x
    f_names = x_names.copy()
    n_t = f_x.shape[0]
    n_f = f_x.shape[1]

    # loop through attribute combinations and add configural features as needed
    combs = list(combinations(f_names, 2)) # combinations of stimulus attributes (elemental cues)
    i_combs = list(combinations(range(n_f), 2))
    f_elem = f_x.copy()
    for i in range(len(combs)):
        new_feature = np.array(f_elem[:, i_combs[i][0]] * f_elem[:, i_combs[i][1]])
        new_needed = abs(sum(new_feature)) > 0 # only include new configural feature if the relevant combination actually shows up in the experiment
        if new_needed:
            f_x = np.concatenate((f_x, new_feature.reshape(n_t, 1)), axis = 1)
            f_names += [combs[i][0] + '.' + combs[i][1]] # name of new column (configural feature)

    # add intercept feature
    new_feature = np.ones((f_x.shape[0], 1))
    f_x = np.concatenate((f_x, new_feature), axis = 1)
    f_names += ['intercept']

    output = {'f_x': f_x, 'f_names': f_names}
    return output
cfg2_intercept.pars = None

def cfg2_half(x, x_names, sim_pars):
    '''
    Binary configural features; all configural features get a value of 0.5 instead of 1.0.
    '''
    f_x = x
    f_names = x_names.copy()
    n_t = f_x.shape[0]
    n_f = f_x.shape[1]

    # loop through attribute combinations and add configural features as needed
    combs = list(combinations(f_names, 2)) # combinations of stimulus attributes (elemental cues)
    i_combs = list(combinations(range(n_f), 2))
    f_elem = f_x.copy()
    for i in range(len(combs)):
        new_feature = 0.5*np.array(f_elem[:, i_combs[i][0]]*f_elem[:, i_combs[i][1]])
        new_needed = abs(sum(new_feature)) > 0 # only include new configural feature if the relevant combination actually shows up in the experiment
        if new_needed:
            f_x = np.concatenate((f_x, new_feature.reshape(n_t, 1)), axis = 1)
            f_names += [combs[i][0] + '.' + combs[i][1]] # name of new column (configural feature)

    output = {'f_x': f_x, 'f_names': f_names}
    return output
cfg2_half.pars = None

def cfg2_distinct(x, x_names, sim_pars):
    '''
    Adds binary configural features only when they are distinct from the corresponding elemental features.
    
    Notes
    -----
    The following are some examples of when configural features would
    and would not be included.
    
    Example 1
    stimulus pairs: A, AB
    A elemental:   1, 1
    B elemental:   0, 1
    AB configural: 0, 1
    AB configural = B elemental -> don't include
    
    Example 2
    stimulus pairs: A, AB, B
    A elemental:   1, 1, 0
    B elemental:   0, 1, 1
    AB configural: 0, 1, 0
    AB configural is distinct -> include
    
    Example 3
    stimulus pairs: AB, B, AC
    A elemental:   1, 0, 1
    B elemental:   1, 1, 0
    C elemental:   0, 0, 1
    AB configural: 1, 0, 0
    AC configural: 0, 0, 1
    AB configural is distinct -> include
    AC configural = C elemental -> don't include
        
    We assume that the learner cannot foresee the what stimulus combinations it will see in the
    future, and this has an effect on the feature representation.  This is that a configural feature will have
    a non-zero value on a trial only if it is distinct from all elemental features that the learner has observed
    SO FAR.  In effect, the learner creates a configural feature as soon as it becomes clear that this will be 
    distinct from the elemental features but no sooner.  Further, the code will not include a configural feature
    if the relevant cue combination does not appear after the configural feature is discovered to be distinct;
    this avoids creating features that have the value 0 on all time steps.  The following examples illustrate
    this behavior.
    
    Example 4
    stage 1 stimulus pairs: B, AB
    stage 2 stimulus pairs: C, AC
    stage 3 stimulus pairs: B, AB
    The AB configural feature is absent (0) in stage 1 (not discovered to be distinct) but present in stage 3.
    The AC configural feature is present in stage 2.
    
    Example 5
    stage 1 stimulus pairs: B, AB
    stage 2 stimulus pairs: C, AC
    The AB configural feature is discovered to be distinct in stage 2 but the AB combination does not appear
    following that and thus the AB configural feature is not included.
    The AC configural feature is present in stage 2.
    
    Note: The code decides to create a new configural feature for two cues (let's call them cue 0 and cue 1)
    on the first time step that cue 0 has appeared without cue 1 and cue 1 has appeared without cue 0.  Prior
    to this, either cue 1 has only appeared with cue 0 or vice-versa (or both), which implies that the cue 0/cue 1
    configural feature will be a copy of one or both of the elemental features.
    
    Note: This type of feature representation is particularly appropriate for simluating schedule objects (usually
    Pavlovian conditioning) in which the context (i.e. set of background stimuli) is explicitly represented
    and inter-trial intervals are included.  This can produce a situation similar to the second example above, in
    which A corresponds to punctate/nominal cues such as tones and lights and B corresponds to the context.
    '''
    f_x = x
    f_names = x_names.copy()
    n_t = f_x.shape[0]
    n_f = f_x.shape[1]
    
    # loop through cue (stimulus attribute) combinations and add configural features as needed
    combs = list(combinations(f_names, 2)) # combinations of stimulus attributes (elemental cues)
    i_combs = list(combinations(range(n_f), 2))
    f_elem = f_x.copy()
    for i in range(len(combs)):
        cue0_name = combs[i][0] # name of cue 0
        cue1_name = combs[i][1] # name of cue 1
        cue0 = i_combs[i][0] # index of cue 0
        cue1 = i_combs[i][1] # index of cue 1
        new_name = cue0_name + '.' + cue1_name # name of new column (configural feature)
        new_feature = np.array(f_elem[:, cue0]*f_elem[:, cue1])
        shows_up = abs(sum(new_feature)) > 0 # only include configural feature if the relevant combination actually shows up
        if shows_up:
            t_first_distinct = None # time point when configural feature is first distinguishable from elemental ones
            cue0_without_cue1 = 0 # number of times so far cue 0 has shown up without cue 1
            cue1_without_cue0 = 0 # number of times so far cue 1 has shown up without cue 0
            for t in range(n_t):
                cue0_without_cue1 += x[t, cue0]*(1 - x[t, cue1]) 
                cue1_without_cue0 += x[t, cue1]*(1 - x[t, cue0])
                distinct = cue0_without_cue1 > 0 and cue1_without_cue0 > 0
                if distinct:
                    t_first_distinct = t
                    break
            new_feature[0:t_first_distinct] = 0 # the new configural feature has value 0 before it is discovered to be distinct from the elemental features
            shows_up_after_distinct = abs(sum(new_feature)) > 0 # only include if the relevant combination shows up after it is discovered to be distinct
            if (not t_first_distinct is None) and shows_up_after_distinct: # only include new configural feature if it's distinct from the elemental features and does not consist wholly of zeros
                f_x = np.concatenate((f_x, new_feature.reshape(n_t, 1)), axis = 1)
                f_names += [new_name]

    output = {'f_x': f_x, 'f_names': f_names}
    return output
cfg2_distinct.pars = None