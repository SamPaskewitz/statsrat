import numpy as np
import pandas as pd

def none(aux, t, fbase, n_f, sim_pars):
    '''
    No special feature weights (vector of ones).
    '''
    new_fweight = np.ones(n_f)
    return new_fweight
none.pars = None

def from_aux_feature(aux, t, fbase, n_f, sim_pars):
    '''
    Take feature weights directly from 'aux'.
    '''
    new_fweight = aux.data['atn'][t, :]
    return new_fweight
from_aux_feature.pars = None

def from_aux_norm2(aux, t, fbase, n_f, sim_pars):
    '''
    Produce weights that normalize features (e.g. CompAct) according to a Euclidean metric.
    This is a version of 'from_aux_norm' with the metric fixed at 2.
    So long as the base features are 0 or 1, this is equivalent to EXIT-style feature weighting.
    '''
    atn_gain = aux.data['atn'][t, :]*fbase[t, :]
    atn_gain[atn_gain < 0.01] = 0.01 # Added this in to make this consistent with the R code.
    norm = sum(atn_gain**2)**(1/2)
    new_fweight = atn_gain/norm
    return new_fweight
from_aux_norm2.pars = None

def from_aux_norm(aux, t, fbase, n_f, sim_pars):
    '''
    Produce weights that normalize features (e.g. CompAct).
    So long as the base features are 0 or 1, this is equivalent to EXIT-style feature weighting.
    '''
    atn_gain = aux.data['atn'][t, :]*fbase[t, :]
    atn_gain[atn_gain < 0.01] = 0.01 # Added this in to make this consistent with the R code.
    norm = sum(atn_gain**sim_pars['metric'])**(1/sim_pars['metric'])
    new_fweight = atn_gain/norm
    return new_fweight
from_aux_norm.pars = None