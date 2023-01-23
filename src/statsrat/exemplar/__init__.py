import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats
from statsrat import resp_fun
from copy import deepcopy
from . import sim, rtrv, atn_update, y_ex_update

class model:
    '''
    Class for exemplar models.

    Attributes
    ----------
    name: str
        Model name.
    sim: function
        Similarity function.
    rtrv: function
        Determines retrieval strength based on similarity.
    atn_update: function
        Determines how attention is updated.
    y_ex_update: function
        Determines how exemplar associations are updated.
    par_names: list
        Names of the model's free parameters (strings).
    pars: dict
        Information about model parameters (min, max, default).

    Methods
    -------
    simulate(trials, par_val = None, init_atn = None, random_resp = False, ident = 'sim')
        Simulate a trial sequence once with known model parameters.
        
    Notes
    -----
    Initial attention weights are not treated as a free model parameter in OAT or model fitting
    functions (in those cases they are fixed at 1), but can be changed in other simulations by using
    the 'init_atn' parameter of the 'simulate' method.  In the future I hope to make it possible to
    include initial attention weights as part of a model's parameter space for purposes of OATs and
    model fitting (the programming is a bit tricky to figure out).
    
    Relevant Papers
    ---------------
    Ghirlanda, S. (2015).
    On elemental and configural models of associative learning.
    Journal of Mathematical Psychology, 64–65, 8–16.
    
    Kruschke, J. K. (1992).
    ALCOVE: An exemplar-based connectionist model of category learning.
    Psychological Review, 99(1), 22–44.
    
    Medin, D. L., & Schaffer, M. M. (1978).
    Context theory of classification learning.
    Psychological Review, 85(3), 207.
    
    Nosofsky, R. M. (1986).
    Attention, Similarity, and the Identification-Categorization Relationship.
    Journal of Experimental Psychology: General, 115(1), 39–57.

    '''
    
    def __init__(self, name, sim, rtrv, atn_update, y_ex_update):
        """
        Parameters
        ----------
        name: str
            Model name.
        sim: function
            Similarity function.
        rtrv: function
            Determines retrieval strength based on similarity.
        atn_update: function
            Determines how attention is updated.
        y_ex_update: function
            Determines how exemplar associations are updated.
        """
        # add attributes to object ('self')
        self.name = name
        self.sim = sim
        self.rtrv = rtrv
        self.atn_update = atn_update
        self.y_ex_update = y_ex_update
        # determine model's parameter space
        par_list = [elm for elm in [sim.pars, rtrv.pars, atn_update.pars, y_ex_update.pars, pd.DataFrame({'min': 0.0, 'max': 10.0, 'default': 1.0}, index = ['resp_scale'])] if elm is not None] # create list of par dataframes, excluding None
        self.pars = pd.concat(par_list)
        self.pars = self.pars.loc[~self.pars.index.duplicated()].sort_index()
        self.par_names = self.pars.index.values
        
    def simulate(self, trials, par_val = None, rich_output = True, init_atn = 1.0, random_resp = False, ident = 'sim'):
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
            responses, i.e. b/b_hat (False).  Defaults to False.
            
        init_atn: float or array, optional
            Initial attention values.  If a scalar (float), then all
            attention values (across cues and exemplars) start with this
            value.  If a 1-D array, then this is intepreted as initial attention
            across cues.  If a 2-D array, then this is interpreted as initial attention
            across exemplars (dimension 0) and cues (dimension 1).  Defaults to 1.0.

        random_resp: str, optional
            Whether or not simulated responses should be random.  Defaults
            to false, in which case behavior (b) is identical to expected
            behavior (b_hat); this saves some computation time.

        ident: str, optional
            Individual participant identifier.  Defaults to 'sim'.

        Returns
        -------
        ds: dataset
            Simulation data.

        Explanation of variables in ds (if rich_output = True)
        ------------------------------------------------------
        y_psb: indicator vector for outcomes (y) that are possible on the trial (from the learner's perspective)
        y_lrn: indicator vector for outcomes (y) for which there is feedback and hence learning will occur
        y_hat: outcome predictions
        b_hat: expected value of behavioral response
        b: vector representing actual behavioral response (identical to b_hat unless the random_resp argument is set to True)
        y_ex: outcome values (y) associated with each exemplar
        atn: attention weights for each exemplar
        sim: similarity between exemplars and the current stimulus (x)
        rtrv: retrieval strength for exemplars
        b_index: index of behavioral response (only present if response type is 'choice' and random_resp is True)
        b_name: name of behavioral response (only present if response type is 'choice' and random_resp is True)

        Notes
        -----
        The response type is determined by the 'resp_type' attribute of the 'trials' object.
        UPDATE AND/OR MOVE THIS.
        
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
        x = np.array(trials['x'], dtype = 'float64')
        y = np.array(trials['y'], dtype = 'float64')
        y_psb = np.array(trials['y_psb'], dtype = 'float64')
        y_lrn = np.array(trials['y_lrn'], dtype = 'float64')
        ex = trials['ex'].values
        ex_names = list(trials['ex_name'].values) # exemplar names
        # count things
        n = {'t': x.shape[0], # number of time points
             'x': x.shape[1], # number of cues
             'y': y.shape[1], # number of outcomes/response options
             'ex': len(ex_names)} # number of exemplars
        # set up array for mean response (b_hat)
        b_hat = np.zeros((n['t'], n['y']))
        # initialize state
        state = {}
        state['x_ex'] = trials['x_ex'].values # exemplar locations
        state['sim'] = np.zeros(n['ex']) # similarity to exemplars
        state['rtrv'] = np.zeros(n['ex']) # exemplar retrieval strength
        state['y_ex'] = np.zeros((n['ex'], n['y'])) # outcomes (y) associated with each exemplar
        state['atn'] = np.zeros((n['ex'], n['x'])) # attention (can be different for each exemplar, but doesn't need to be)
        state['atn'][:, :] = init_atn
        state['y_hat'] = np.zeros(n['y']) # outcome predictions
        state['b_hat'] = np.zeros(n['y']) # expected behavior
        state['ex_seen_yet'] = pd.Series(n['ex']*[0], index = ex_names) # keeps track of which exemplars have been observed yet
        state['ex_counts'] = pd.Series(n['ex']*[0], index = ex_names) # number of times each exemplar has been observed
        state_history = []

        # set up response function (depends on response type)
        resp_dict = {'choice': resp_fun.choice,
                     'exct': resp_fun.exct,
                     'supr': resp_fun.supr,
                     'normal': resp_fun.normal,
                     'log_normal': resp_fun.log_normal}
        response = resp_dict[trials.resp_type]

        # loop through time steps
        for t in range(n['t']):
            env = {'x': x[t, :], 'y': y[t, :], 'y_psb': y_psb[t, :], 'y_lrn': y_lrn[t, :], 'ex': ex[t]}
            state['ex_seen_yet'][env['ex']] = 1 # note that current exemplar has been seen
            state['ex_counts'][env['ex']] += 1            
            state['sim'] = state['ex_seen_yet']*self.sim(state, n, env, sim_pars) # similarity
            state['rtrv'] = self.rtrv(state, n, env, sim_pars) # retrieval strength
            state['y_hat'] = state['rtrv']@(env['y_psb']*state['y_ex']) # prediction
            b_hat[t, :] = response.mean(state['y_hat'], env['y_psb'], sim_pars['resp_scale']) # response
            state_history += [state]
            #state_history += [deepcopy(state)] # record a copy of the current state before learning occurs
            state['y_ex'] += self.y_ex_update(state, n, env, sim_pars) # update y_ex
            state['atn'] += self.atn_update(state, n, env, sim_pars) # update attention
        
        # generate simulated responses
        if random_resp:
            (b, b_index) = response.random(b_hat, sim_pars['resp_scale'])
        else:
            b = b_hat
            b_index = None
        
        if rich_output:
            # put all simulation data into a single xarray dataset
            ds = trials.copy(deep = True)
            ds = ds.assign_coords({'ex_name' : ex_names, 'ident' : [ident]})
            ds = ds.assign({'y_psb' : (['t', 'y_name'], y_psb),
                            'y_lrn' : (['t', 'y_name'], y_lrn),
                            'b_hat' : (['t', 'y_name'], b_hat),
                            'b' : (['t', 'y_name'], b),
                            'y_ex' : (['t', 'ex_name', 'y_name'], np.zeros((n['t'], n['ex'], n['y']))),
                            'atn': (['t', 'ex_name', 'x_name'], np.zeros((n['t'], n['ex'], n['x']))),
                            'sim': (['t', 'ex_name'], np.zeros((n['t'], n['ex']))),
                            'rtrv': (['t', 'ex_name'], np.zeros((n['t'], n['ex']))),
                            'y_hat': (['t', 'y_name'], np.zeros((n['t'], n['y'])))})
            for t in range(n['t']): # fill out the xarray dataset from state_history
                ds['y_ex'].loc[{'t': t}] = state_history[t]['y_ex']
                ds['atn'].loc[{'t': t}] = state_history[t]['atn']
                ds['sim'].loc[{'t': t}] = state_history[t]['sim']
                ds['rtrv'].loc[{'t': t}] = state_history[t]['rtrv']
                ds['y_hat'].loc[{'t': t}] = state_history[t]['y_hat']
            ds = ds.assign_attrs({'model': self.name,
                                  'model_class' : 'exemplar',
                                  'sim_pars' : sim_pars.values})
        else:
            # FOR NOW (until I revise how log-likelihood calculations work) just put b and b_hat in a dataset
            ds = trials.copy(deep = True)
            ds = ds.assign({'b_hat' : (['t', 'y_name'], b_hat),
                            'b' : (['t', 'y_name'], b)})
        return ds