import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats
from statsrat import resp_fun

# This is based on 'Neural Representation of Subjective Value Under Risk and Ambiguity' by Levi et al (2010).
# It's designed for use with Louisa Smith's data in the fall of 2020 at CU Boulder.
# I might later expand this to be more flexible and general.

class model:

    def __init__(self):
        self.name = 'decision_model'
        # determine model's parameter space
        par_names = ['alpha', 'beta', 'resp_scale']
        self.pars = pars.loc[par_names]
 
    def simulate(self, trials, resp_type = 'choice', par_val = None, random_resp = False, ident = 'sim'):
        # use default parameters unless others are given
        if par_val is None:
            sim_pars = self.pars['default']
        else:
            # check that parameter values are within acceptable limits; if so assemble into a pandas series
            # for some reason, the optimization functions go slightly outside the specified bounds
            abv_min = par_val >= self.pars['min'] - 0.0001
            blw_max = par_val <= self.pars['max'] + 0.0001
            all_ok = np.prod(abv_min & blw_max)
            assert all_ok, 'par_val outside acceptable limits'
            sim_pars = pd.Series(par_val, self.pars.index) 
        
        # set stuff up
        choice_names = list(trials.choice.values)
        v = np.array(trials['v'], dtype = 'float64') # nominal choice values
        p = np.array(trials['p'], dtype = 'float64') # reward probabilities
        a = np.array(trials['a'], dtype = 'float64') # ambiguity levels
        n_t = v.shape[0] # number of time points
        n_c = v.shape[1] # number of choices
        s = np.zeros((n_t, n_c)) # subjective values
        b_hat = np.zeros((n_t, n_c)) # expected behavior

        # set up response function (depends on response type)
        resp_dict = {'choice': resp_fun.choice,
                     'exct': resp_fun.exct,
                     'supr': resp_fun.supr}
        sim_resp_fun = resp_dict[resp_type]

        # loop through time steps
        for t in range(n_t):
            prob_factor = (p[t, :] - sim_pars['beta']*a[t, :]/2)
            scaled_v = np.abs(v[t, :])**sim_pars['alpha']
            val_factor = np.sign(v[t, :])*scaled_v
            s[t, :] = prob_factor*val_factor
            b_hat[t, :] = resp_fun.choice(s[t, :], np.ones(n_c), sim_pars['resp_scale']) # response
                
        # generate simulated responses
        if random_resp is False:
            b = b_hat
        else:
            rng = np.random.default_rng()
            b = np.zeros((n_t, n_u))
            for t in range(n_t):
                choice = rng.choice(n_u, p = b_hat[t, :])
                b[t, choice] = 1
        
        # put all simulation data into a single xarray dataset
        ds = xr.Dataset(data_vars = {'v' : (['t', 'choice'], v),
                                     'p' : (['t', 'choice'], p),
                                     'a' : (['t', 'choice'], a),
                                     's' : (['t', 'choice'], s),
                                     'b_hat' : (['t', 'choice'], b_hat),
                                     'b' : (['t', 'choice'], b)},
                        coords = {'t' : range(n_t),
                                  'choice' : choice_names,
                                  'ident' : [ident]},
                        attrs = {'model': self.name,
                                 'model_class' : 'decision',
                                 'resp_type' : resp_type,
                                 'sim_pars' : sim_pars})
        return ds

########## PARAMETERS ##########

par_names = ['resp_scale']; par_list = [{'min': 0.0, 'max': 10.0, 'default': 1.0}] # scales the softmax response function; corresponds to 'gamma' in the Levi et al paper
par_names += ['alpha']; par_list += [{'min': 0.01, 'max': 2.0, 'default': 0.7}] # risk parameter (the largest alpha value fit in the Levi et al study was 1.2)
par_names += ['beta']; par_list += [{'min': 0.01, 'max': 1.0, 'default': 0.7}] # ambiguity parameter
pars = pd.DataFrame(par_list, index = par_names)
del par_names; del par_list