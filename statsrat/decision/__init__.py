import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats

class model:
    """
    Class for decision making models (used with lottery tasks).
    
    Attributes
    ----------
    name : str
        Model name.
    utility : function
        Utility function (maps nominal rewards to utility).
    dweight : function
        Decision weighting function (an increasing function
        of probability).
    par_names : list
        Names of the model's free parameters (strings).
        
    Methods
    -------
    simulate(trials, par_val = None)
        Simulate a trial sequence once with known model parameters.
    
    """
    def simulate(self, trials, par_val = None, random_resp = False, ident = 'sim'):
        """
        Parameters
        ----------
        trials : dataset (xarray)
            Time step level experimental data (cues, outcomes etc.).

        resp_type : str, optional
            Type of behavioral response: one of 'choice', 'exct' or 'supr'.
            Defaults to 'choice'.

        par_val : list, optional
            Learning model parameters (floats or ints).

        random_resp : str, optional
            Whether or not simulated responses should be random.  Defaults
            to false, in which case behavior (b) is identical to expected
            behavior (b_hat); this saves some computation time.  If true,
            then discrete responses are selected using b_hat as choice
            probabilities.

        ident : str, optional
            Individual participant identifier.  Defaults to 'sim'.

        Returns
        -------
        ds : dataset
            Simulation data.
        """
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
        probs = np.array(trials['probs'], dtype = 'float64')
        ambig = np.array(trials['ambig'], dtype = 'float64')
        rwds = np.array(trials['rwds'], dtype = 'float64')
        subj_value = np.zeros((n_t, n_p)) # subjective value
        b_hat = np.zeros((n_t, n_p)) # expected behavior
        
        # loop through time steps
        for t in range(n_t):
            # calculate the value of each prospect
            for p in range(n_p):
                outcome_utilities = self.dweight(sim_pars, probs[t, p, :], ambig[t, p, :])*self.utility(sim_pars, rwds[t, p, :])               
                subj_value[t, p] = outcome_utilities.sum()
            b_hat[t, :] = resp_fun.choice(subj_value[t, :], np.array(n_p*[1]), sim_pars['resp_scale']) # response
        # FINISH
        
        # generate simulated responses
        if random_resp is False:
            b = b_hat
        else:
            rng = np.random.default_rng()
            b = np.zeros((n_t, n_u))
            for t in range(n_t):
                choice = rng.choice(n_u, p = b_hat[t, :])
                b[t, choice] = 1

        # FIX THIS
        # put all simulation data into a single xarray dataset
        ds = xr.Dataset(data_vars = {'rwds' : (['t', 'p_name', 'o_name'], rwds),
                                     'probs' : (['t', 'p_name', 'o_name'], probs),
                                     'subj_value' : (['t', 'p_name'], subj_value),
                                     'b_hat' : (['t', 'p_name'], u)},
                        coords = {'t' : range(n_t),
                                  't_name' : ('t', trials.t_name),
                                  'p_name' : p_names,
                                  'o_name' : o_names,
                                  'ident' : [ident]},
                        attrs = {'model': self.name,
                                 'model_class' : 'decision',
                                 'schedule' : trials.attrs['schedule'],
                                 'sim_pars' : sim_pars})
        return ds

########## PARAMETERS ##########

par_names = ['resp_scale']; par_list = [{'min': 0.0, 'max': 10.0, 'default': 1.0}]
par_names += ['alpha']; par_list += [{'min': 0.0, 'max': 2.0, 'default': 1.0}]
pars = pd.DataFrame(par_list, index = par_names)
del par_names; del par_list