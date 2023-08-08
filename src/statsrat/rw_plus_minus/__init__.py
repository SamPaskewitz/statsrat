import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats
from statsrat import resp_fun
from time import time
from copy import deepcopy

class model:
    '''
    Class for Rescorla-Wagner family learning models with
    association weights separated into positive and negative 
    parts and separate decay rates for each.

    Attributes
    ----------
    name: str
        Model name.
    pred: function
        Prediction function.
    fbase: function
        Base mapping between cues (x) and features (f_x).
    fweight: function
        Attentional weights for features.
    lrate: function
        Determines learning rates.
    aux: object
        Auxilliary learning, e.g. attention or weight covariance.
    par_names: list
        Names of the model's free parameters (strings).
    pars: dict
        Information about model parameters.

    Methods
    -------
    simulate(trials, par_val = None, random_resp = False, ident = 'sim')
        Simulate a trial sequence once with known model parameters.
        
    Notes
    -----
    This is a variant of the Rescorla-Wagner model family in which association weights
    (w) are divided into positive (w_plus) and negative (w_minus) parts:
    w = w_plus + w_minus, with w_plus >=0 and w_minus <= 0
    
    The learning rules for w_plus and w_minus are arranged such that the change in net
    association weights (w) is IDENTICAL to the ordinary Rescorla-Wagner update rule,
    while keeping w_plus >= 0 and w_minus <= 0.
    
    The parameter gamma determines the balance between changes in w_plus and w_minus: when gamma
    is large, w_plus mainly changes, while when gamma is small w_minus largely changes.
    We keep 0 < gamma < 1.  See the code below for the exact learning rules (THIS SHOULD BE ADDED TO THE
    HELP TEXT ITSELF IN LATER VERSIONS).
    '''

    def __init__(self, name, pred, fbase, fweight, lrate, aux):
        """
        Parameters
        ----------
        name: str
            Model name.
        pred: function
            Prediction function.
        fbase: function
            Base mapping between cues (x) and features (f_x).
        fweight: function
            Attentional weights for features.
        lrate: function
            Determines learning rates.
        aux: object
            Auxilliary learning, e.g. attention or weight covariance.
        """
        # add attributes to object ('self')
        self.name = name
        self.pred = pred
        self.fbase = fbase
        self.fweight = fweight
        self.lrate = lrate
        self.aux = aux
        # determine model's parameter space
        par_list = [elm for elm in [pred.pars, fbase.pars, fweight.pars, lrate.pars, aux.pars, pd.DataFrame([{'min': 0.0, 'max': 1.0, 'default': 0.25}, {'min': 0.0, 'max': 1.0, 'default': 0.25}, {'min': 0.0, 'max': 1.0, 'default': 0.5}, {'min': 0.0, 'max': 1.0, 'default': 0.5}, {'min': 0.0, 'max': 10.0, 'default': 1.0}], index = ['drate_plus', 'drate_minus', 'gamma_pos', 'gamma_neg', 'resp_scale'])] if elm is not None] # create list of par dataframes, excluding None
        self.pars = pd.concat(par_list)
        self.pars = self.pars.loc[~self.pars.index.duplicated()].sort_index()
        self.par_names = self.pars.index.values
 
    def simulate(self, trials, par_val = None, rich_output = False, random_resp = False, ident = 'sim'):
        """
        Simulate a trial sequence once with known model parameters.
        
        Parameters
        ----------
        trials: dataset (xarray)
            Time step level experimental data (cues, outcomes etc.).

        par_val: list, optional
            Learning model parameters (floats or ints).
            
        rich_output: Boolean, optional
            Whether to output full simulation data (True) or just
            responses, i.e. b/b_hat (False).  Defaults to True.

        random_resp: str, optional
            Whether or not simulated responses should be random.  Defaults
            to false, in which case behavior (b) is identical to expected
            behavior (b_hat); this saves some computation time.  If true
            and resp_type is 'choice', then discrete responses are selected
            using b_hat as choice probabilities.  If true and resp_type is
            'exct' or 'supr' then a small amount of normally distributed
            noise (sd = 0.01) is added to b_hat.

        ident: str, optional
            Individual participant identifier.  Defaults to 'sim'.

        Returns
        -------
        
        ds: dataset
            Simulation data.
            
        Explanation of variables in ds
        ------------------------------
        f_name: feature names
        y_psb: indicator vector for outcomes (y) that are possible on the trial (from the learner's perspective)
        y_lrn: indicator vector for outcomes (y) for which there is feedback and hence learning will occur
        fbase: base feature vectors (before weighting)
        fweight: feature weights
        f_x: feature vectors
        y_hat: outcome predictions
        b_hat: expected value of behavioral response
        b: vector representing actual behavioral response (identical to b_hat unless the random_resp argument is set to True)
        w: association weights
        delta: prediction error
        lrate: learning rates
        b_index: index of behavioral response (only present if response type is 'choice' and random_resp is True)
        b_name: name of behavioral response (only present if response type is 'choice' and random_resp is True)

        Notes
        -----
        The response type is determined by the 'resp_type' attribute of the 'trials' object.
        
        The response type 'choice' is used for discrete response options.  This
        produces response probabilities using a softmax function:
        .. math:: \text{resp}_i = \frac{ e^{\phi \hat{u}_i} }{ \sum_j e^{\phi \hat{u}_j} }

        The response type 'exct' is used for excitatory Pavlovian
        conditioning:
        .. math:: \text{resp} = \frac{ e^{\phi \hat{u}_i} }{ e^{\phi \hat{u}_i} + 1 }

        The response type 'supr' (suppression) is used for inhibitory
        Pavlovian conditioning:
        .. math:: \text{resp} = \frac{ e^{-\phi \hat{u}_i} }{ e^{-\phi \hat{u}_i} + 1 }

        Here :math:`\phi` represents the 'resp_scale' parameter.
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
        time = np.array(trials['time'])
        x = np.array(trials['x'], dtype = 'float64')
        y = np.array(trials['y'], dtype = 'float64')
        y_psb = np.array(trials['y_psb'], dtype = 'float64')
        y_lrn = np.array(trials['y_lrn'], dtype = 'float64')
        x_names = list(trials.x_name.values)
        y_names = list(trials.y_name.values)
        (fbase, f_names) = self.fbase(x, x_names, sim_pars).values() # features and feature names
        fbase = fbase.squeeze()
        # count things
        n = {'t': x.shape[0], # number of time points
             'x': x.shape[1], # number of cues
             'y': y.shape[1], # number of outcomes/response options
             'f': fbase.shape[1]} # number of features
        # set up array for mean response (b_hat)
        b_hat = np.zeros((n['t'], n['y']))
        # initialize state
        state_dims = {'fbase': ['f_name'], 'fweight': ['f_name'], 'f_x': ['f_name'], 'y_hat': ['y_name'], 'delta': ['y_name'], 'w_plus': ['f_name', 'y_name'], 'w_minus': ['f_name', 'y_name'], 'w': ['f_name', 'y_name'], 'lrate': ['f_name', 'y_name']}
        state_sizes = {'fbase': [n['f']], 'fweight': [n['f']], 'f_x': [n['f']], 'y_hat': [n['y']], 'delta': [n['y']], 'w_plus': [n['f'], n['y']], 'w_minus': [n['f'], n['y']], 'w': [n['f'], n['y']], 'lrate': [n['f'], n['y']]}
        state = {}
        for var in state_sizes:
            state[var] = np.zeros(state_sizes[var])
        new_state, new_state_dims, new_state_sizes = self.aux(state, n, {}, sim_pars, 'initialize') # add auxilliary variables (e.g. attention) to initial state
        state.update(new_state); state_dims.update(new_state_dims); state_sizes.update(new_state_sizes)
        state_history = []
        # figure out x_dims
        has_x_dims = 'x_dims' in list(trials.attrs.keys())
        if has_x_dims:
            x_dims = trials.attrs['x_dims']
        else:
            x_dims = None
        # set up response function (depends on response type)
        resp_dict = {'choice': resp_fun.choice,
                     'exct': resp_fun.exct,
                     'supr': resp_fun.supr,
                     'normal': resp_fun.normal,
                     'log_normal': resp_fun.log_normal}
        response = resp_dict[trials.resp_type]

        # set up response function (depends on response type)
        resp_dict = {'choice': resp_fun.choice,
                     'exct': resp_fun.exct,
                     'supr': resp_fun.supr,
                     'normal': resp_fun.normal,
                     'log_normal': resp_fun.log_normal}
        response = resp_dict[trials.resp_type]

        # loop through time steps
        for t in range(n['t']):
            # check if there has been a timeskip greater than (e.g. representing a delay of a day); if so then process weight decay
            timeskip = time[t] - time[t-1]
            if timeskip > 1:
                state['w_plus'] *= (1 - sim_pars['drate_plus'])**timeskip
                state['w_minus'] *= (1 - sim_pars['drate_minus'])**timeskip
            
            # prediction, behavior, and prediction error calculation
            env = {'x': x[t, :], 'y': y[t, :], 'y_psb': y_psb[t, :], 'y_lrn': y_lrn[t, :]}
            state['fbase'] = fbase[t, :]
            state['fweight'] = self.fweight(state, n, env, sim_pars)
            state['f_x'] = state['fbase']*state['fweight'] # weight base features
            state['w'] = state['w_plus'] + state['w_minus'] # association weights
            state['y_hat'] = self.pred(env['y_psb']*(state['f_x']@state['w']), sim_pars) # prediction
            b_hat[t, :] = response.mean(state['y_hat'], env['y_psb'], sim_pars['resp_scale']) # response
            state['delta'] = env['y_lrn']*(env['y'] - state['y_hat']) # prediction error
            state = self.aux(state, n, env, sim_pars, 'compute') # compute auxiliary data for current time step
            state['lrate'] = self.lrate(state, n, env, sim_pars) # learning rates for this time step
                        
            # make a copy of the current state before learning occurs
            if rich_output:
                state_history += [deepcopy(state)]     
            
            # association learning
            for j in range(n['y']):
                if env['y_lrn'][j] == 1:
                    if state['delta'][j] < 0:
                        for i in range(n['f']):
                            if sim_pars['gamma_neg']*state['lrate'][i, j]*state['delta'][j] >= -state['w_plus'][i, j]:
                                state['w_plus'][i, j] += sim_pars['gamma_neg']*state['lrate'][i, j]*state['delta'][j]
                                state['w_minus'][i, j] += (1 - sim_pars['gamma_neg'])*state['lrate'][i, j]*state['delta'][j]
                            else:
                                state['w_plus'][i, j] = 0.0
                                state['w_minus'][i, j] += state['lrate'][i, j]*state['delta'][j] + state['w_plus'][i, j]
                    else:
                        for i in range(n['f']):
                            if (1 - sim_pars['gamma_pos'])*state['lrate'][i, j]*state['delta'][j] <= -state['w_minus'][i, j]:
                                state['w_plus'][i, j] += sim_pars['gamma_pos']*state['lrate'][i, j]*state['delta'][j]
                                state['w_minus'][i, j] += (1 - sim_pars['gamma_pos'])*state['lrate'][i, j]*state['delta'][j]
                            else:
                                state['w_plus'][i, j] += state['lrate'][i, j]*state['delta'][j] + state['w_minus'][i, j]
                                state['w_minus'][i, j] = 0.0
                    
            # weight decay
            state['w_plus'] -= sim_pars['drate_plus']*state['w_plus']
            state['w_minus'] -= sim_pars['drate_minus']*state['w_minus']
            
            # update aux
            state = self.aux(state, n, env, sim_pars, 'update') # update auxiliary data for current time step
        
        # generate simulated responses
        if random_resp:
            (b, b_index) = response.random(b_hat, sim_pars['resp_scale'])
        else:
            b = b_hat
            b_index = None

        # put data in an xarray dataset
        ds = trials.copy(deep = True)
        ds = ds.assign_coords({'f_name' : f_names, 'ident' : [ident]})
        ds = ds.assign({'b_hat' : (['t', 'y_name'], b_hat),
                        'b' : (['t', 'y_name'], b)})
        ds = ds.assign_attrs({'model': self.name,
                              'model_class' : 'rw_plus_minus',
                              'sim_pars' : sim_pars.values})
        if rich_output:             
            # fill out the xarray dataset from state_history
            for var in state:
                ds = ds.assign({var: (['t'] + state_dims[var], np.zeros([n['t']] + state_sizes[var]))})
                for t in range(n['t']):
                    ds[var].loc[{'t': t}] = state_history[t][var]
                    
        return ds