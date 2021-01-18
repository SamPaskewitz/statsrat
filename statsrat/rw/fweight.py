import numpy as np

########## FEATURE WEIGHTING FUNCTIONS ##########

def none(aux, t, fbase, fweight, n_f, sim_pars):
    '''
    No special feature weights (vector of ones).
    '''
    new_fweight = np.ones(n_f)
    return new_fweight
none.par_names = []

def from_aux_feature(aux, t, fbase, fweight, n_f, sim_pars):
    '''
    Take feature weights directly from 'aux'.
    '''
    new_fweight = aux.data['atn'][t, :]
    return new_fweight
from_aux_feature.par_names = []

def from_aux_norm(aux, t, fbase, fweight, n_f, sim_pars):
    '''
    Produce weights that normalize features (e.g. CompAct).
    So long as the base features are 0 or 1, this is equivalent to EXIT-style feature weighting.
    '''
    atn_gain = aux.data['atn'][t, :]*fbase[t, :]
    atn_gain[atn_gain < 0.01] = 0.01 # Added this in to make this consistent with the R code.
    norm = sum(atn_gain**sim_pars['metric'])**(1/sim_pars['metric'])
    new_fweight = atn_gain/norm
    return new_fweight
from_aux_norm.par_names = []