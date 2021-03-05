import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats
from statsrat import resp_fun
from . import sim, atn_update, lrate

class model:
    '''
    Class for exemplar models.

    Attributes
    ----------
    name: str
        Model name.
    sim: function
        Similarity function.
    atn_update: function
        Determines how attention is updated.
    lrate: function
        Determines learning rates for exemplars.
    par_names: list
        Names of the model's free parameters (strings).
    pars: dict
        Information about model parameters.

    Methods
    -------
    simulate(trials, resp_type = 'choice', par_val = None, init_atn = None, random_resp = False, ident = 'sim')
        Simulate a trial sequence once with known model parameters.
        
    Notes
    -----
    Initial attention weights are not treated as a free model parameter in OAT or model fitting
    functions (in those cases they are fixed at 1), but can be changed in other simulations by using
    the 'init_atn' parameter of the 'simulate' method.  In the future I hope to make it possible to
    include initial attention weights as part of a model's parameter space for purposes of OATs and
    model fitting (the programming is a bit tricky to figure out).
    '''
    
    def __init__(self, name, sim, atn_update, lrate):
        """
        Parameters
        ----------
        name: str
            Model name.
        sim: function
            Similarity function.
        atn_update: function
            Determines how attention is updated.
        lrate: function
            Determines learning rates for exemplars.
        """
        # add attributes to object ('self')
        self.name = name
        self.sim = sim
        self.atn_update = atn_update
        self.lrate = lrate
        # determine model's parameter space
        self.par_names = list(np.unique(sim.par_names + atn_update.par_names + lrate.par_names))
        self.pars = pars.loc[self.par_names + ['resp_scale']]
        
    def simulate(self, trials, resp_type = 'choice', par_val = None, init_atn = 1.0, random_resp = False, ident = 'sim'):
        """
        Simulate a trial sequence once with known model parameters.
        
        Parameters
        ----------
        trials: dataset (xarray)
            Time step level experimental data (cues, outcomes etc.).

        resp_type: str, optional
            Type of behavioral response: one of 'choice', 'exct' or 'supr'.
            Defaults to 'choice'.

        par_val: list, optional
            Learning model parameters (floats or ints).
            
        init_atn: float or array, optional
            Initial attention values.  If a scalar (float), then all
            attention values (across cues and exemplars) start with this
            value.  If a 1-D array, then this is intepreted as initial attention
            across cues.  If a 2-D array, then this is interpreted as initial attention
            across exemplars (dimension 0) and cues (dimension 1).  Defaults to 1.0.

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

        Notes
        -----
        Use the response type 'choice' for discrete response options.  This
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
        u = np.array(trials['u'], dtype = 'float64')
        u_psb = np.array(trials['u_psb'], dtype = 'float64')
        u_lrn = np.array(trials['u_lrn'], dtype = 'float64')
        ex = trials['ex_name'].values
        x_ex = trials.ex # exemplar locations
        x_names = list(trials.x_name.values) # cue names
        u_names = list(trials.u_name.values) # outcome names
        ex_names = list(x_ex.index.values) # exemplar names
        n_t = x.shape[0] # number of time points
        n_x = x.shape[1] # number of features
        n_u = u.shape[1] # number of outcomes/response options
        n_ex = len(ex_names) # number of exemplars
        sim = np.zeros((n_t, n_ex)) # similarity to exemplars
        rtrv = np.zeros((n_t, n_ex)) # retrieval strength, i.e. normalized similarity
        u_ex = np.zeros((n_t + 1, n_ex, n_u)) # outcomes (u) associated with each exemplar
        lrate = np.zeros((n_t, n_ex)) # learning rates for exemplars
        atn = np.zeros((n_t + 1, n_ex, n_x)) # attention (can be different for each exemplar, but doesn't need to be)
        atn[0, :, :] = init_atn
        u_hat = np.zeros((n_t, n_u)) # outcome predictions
        b_hat = np.zeros((n_t, n_u)) # expected behavior
        delta = np.zeros((n_t, n_u)) # prediction error
        has_x_dims = 'x_dims' in list(trials.attrs.keys())
        if has_x_dims:
            x_dims = trials.attrs['x_dims']
        else:
            x_dims = None
        ex_seen_yet = pd.Series(n_ex*[0], index = ex_names) # FIX THIS keeps track of which exemplars have been observed yet

        # set up response function (depends on response type)
        resp_dict = {'choice': resp_fun.choice,
                     'exct': resp_fun.exct,
                     'supr': resp_fun.supr}
        sim_resp_fun = resp_dict[resp_type]

        # loop through time steps
        for t in range(n_t):
            ex_seen_yet[ex[t]] = 1 # note that current exemplar has been seen
            sim[t, :] = ex_seen_yet*self.sim(x[t, :], x_ex, atn[t, :, :], sim_pars) # similarity of current stimulus to exemplars
            rtrv[t, :] = sim[t, :]/sim[t, :].sum() # retrieval strength (normalized similarity)
            u_hat[t, :] = rtrv[t, :]@(u_psb[t, :]*u_ex[t, :, :]) # prediction
            b_hat[t, :] = sim_resp_fun(u_hat[t, :], u_psb[t, :], sim_pars['resp_scale']) # response
            delta[t, :] = u[t, :] - u_hat[t, :] # prediction error
            lrate[t, :] = self.lrate(rtrv[t, :], n_ex, sim_pars) # learning rates for exemplars
            u_ex[t + 1, :, :] = u_ex[t, :, :] + np.outer(lrate[t, :], u_lrn[t, :]*delta[t, :]) # update
            atn[t + 1, :, :] = atn[t, :] + self.atn_update(rtrv[t, :], delta[t, :], n_x, n_u, n_ex, sim_pars) # update attention
            
        # generate simulated responses
        if random_resp is False:
            b = b_hat
        else:
            rng = np.random.default_rng()
            if resp_type == 'choice':
                b = np.zeros((n_t, n_u))
                for t in range(n_t):
                    choice = rng.choice(n_u, p = b_hat[t, :])
                    b[t, choice] = 1
            else:
                b = b_hat + stats.norm.rvs(loc = 0, scale = 0.01, size = (n_t, n_u))
        
        # put all simulation data into a single xarray dataset
        ds = xr.Dataset(data_vars = {'x' : (['t', 'x_name'], x),
                                     'u' : (['t', 'u_name'], u),
                                     'u_psb' : (['t', 'u_name'], u_psb),
                                     'u_lrn' : (['t', 'u_name'], u_lrn),
                                     'u_hat' : (['t', 'u_name'], u_hat),
                                     'b_hat' : (['t', 'u_name'], b_hat),
                                     'b' : (['t', 'u_name'], b),
                                     'u_ex' : (['t', 'ex_name', 'u_name'], u_ex[range(n_t), :, :]), # remove unnecessary last row
                                     'delta' : (['t', 'u_name'], delta),
                                     'lrate': (['t', 'ex_name'], lrate),
                                     'atn': (['t', 'ex_name', 'x_name'], atn[range(n_t), :, :]),
                                     'sim': (['t', 'ex_name'], sim),
                                     'rtrv': (['t', 'ex_name'], rtrv)},
                        coords = {'t' : range(n_t),
                                  't_name' : ('t', trials.t_name),
                                  'trial' : ('t', trials.trial),
                                  'trial_name' : ('t', trials.trial_name),
                                  'stage' : ('t', trials.stage),
                                  'stage_name' : ('t', trials.stage_name),
                                  'ex' : ('t', ex),
                                  'x_name' : x_names,
                                  'ex_name' : ex_names,
                                  'u_name' : u_names,
                                  'ident' : [ident]},
                        attrs = {'model': self.name,
                                 'model_class' : 'rw',
                                 'schedule' : trials.attrs['schedule'],
                                 'resp_type' : resp_type,
                                 'sim_pars' : sim_pars})
        
        return ds

########## PARAMETERS ##########
par_names = ['resp_scale']; par_list = [{'min': 0.0, 'max': 10.0, 'default': 1.0}]
par_names += ['lrate_par']; par_list += [{'min': 0.0, 'max': 1.0, 'default': 0.5}]
par_names += ['decay_rate']; par_list += [{'min': 0.0, 'max': 5.0, 'default': 0.5}]
pars = pd.DataFrame(par_list, index = par_names)
del par_names; del par_list