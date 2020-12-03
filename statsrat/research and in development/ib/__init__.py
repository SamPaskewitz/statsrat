import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats
from statsrat import resp_fun
from . import rtrv, act, encd, aux

# 9/24/2020: FINISH THIS.

class model:
    '''
    Class for instance-based learning models.

    Attributes
    ----------
    name : str
        Model name.
    rtrv : function
        Retrieval strength.
    act : function
        Activation strength.
    encd : function
        Determines encoding strength for memories.
    aux : object
        Auxilliary learning, e.g. for selective attention.
    par_names : list
        Names of the model's free parameters (strings).

    Methods
    -------
    simulate(trials, resp_type = 'choice', par_val = None)
        Simulate a trial sequence once with known model parameters.
    multi_sim(trials_list, resp_type, par_val)
        Simulate one or more trial sequences with known parameters.
    log_lik(trials, eresp, par_val)
        Compute log-likelihood of individual time step data.
    '''

    def __init__(self, name, rtrv, act, encd, aux, par_names):
        """
        Parameters
        ----------
        name : str
            Model name.
        rtrv : function
            Retrieval strength: similarity or an increasing function of similarity.
            Retrieval strength for each memory trace only depends on that
            trace's recorded stimulus, its encoding strength, and the
            current stimulus.
        act : function
            Activation function, which transforms retrieval strength into
            weights that determine how much each memory trace impacts prediction.
            Examples include normalizing retrieval strengths and selecting the 
            k traces with the largest retrieval strengths (for k nearest neighbors).
        encd : function
            Determines encoding strength for memories, i.e. the part of retrieval
            strength that does not depend on similarity.
        aux : object
            Auxilliary learning, e.g. for selective attention.
        par_names : list
            Names of the model's free parameters (strings).
        """
        # add data to object ('self')
        self.name = name
        self.rtrv = rtrv
        self.act = act
        self.encd = encd
        self.aux = aux
        self.pars = pars.loc[par_names + ['resp_scale']]
 
    def simulate(self, trials, resp_type = 'choice', par_val = None, random_resp = False, ident = 'sim'):
        """
        Simulate a trial sequence once with known model parameters.
        
        Parameters
        ----------
        trials : data frame
            Time step level experimental data (cues, outcomes etc.).

        resp_type : str, optional
            Type of behavioral response: one of 'choice', 'exct' or 'supr'.
            Defaults to 'choice'.

        par_val : list, optional
            Learning model parameters (floats or ints).

        random_resp : str, optional
            Whether or not simulated responses should be random.  Defaults
            to false, in which case behavior (b) is identical to expected
            behavior (b_hat); this saves some computation time.  If true
            and resp_type is 'choice', then discrete responses are selected
            using b_hat as choice probabilities.  If true and resp_type is
            'exct' or 'supr' then a small amount of normally distributed
            noise (sd = 0.01) is added to b_hat.

        ident : str, optional
            Individual participant identifier.  Defaults to 'sim'.

        Returns
        -------
        ds : dataset

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
        x_names = list(trials.x_name.values)
        u_names = list(trials.u_name.values)
        n_t = x.shape[0] # number of time points
        n_x = x.shape[1] # number of stimulus attributes
        n_u = u.shape[1] # number of outcomes/response options
        x_mem = np.zeros((n_t, n_x)) # memory for stimuli (x)
        u_mem = np.zeros((n_t, n_u)) # memory for outcomes (u)
        rtrv = np.zeros((n_t, n_t)) # retrieval strength
        act = np.zeros((n_t, n_t)) # memory activation
        encd = np.zeros(n_t) # record of whether each item was recorded or not
        u_hat = np.zeros((n_t, n_u)) # outcome predictions
        b_hat = np.zeros((n_t, n_u)) # expected behavior
        delta = np.zeros((n_t, n_u))
        aux = self.aux(sim_pars, n_t, n_x, n_u)

        # set up response function (depends on response type)
        resp_dict = {'choice': resp_fun.choice,
                     'exct': resp_fun.exct,
                     'supr': resp_fun.supr}
        sim_resp_fun = resp_dict[resp_type]
        
        # time step 0
        u_hat[0, :] = n_u*[0] # prediction
        b_hat[0, :] = n_u*[0] # response
        delta[0, :] = u[0, :] # prediction error
        encd[0] = 1 # encoding
        x_mem[0, :] = x[0, :] # record initial stimulus (x)
        u_mem[0, :] = u_lrn[0, :]*u[0, :] # record initial outcome vector (u)
        n_mem = 1 # number of items in memory
        
        # loop through remaining time steps
        for t in range(1, n_t):
            rtrv[t, :] = self.rtrv(x[t, :], x_mem[0:n_mem, :], aux, t, n_mem, n_t, sim_pars) # retrieval
            act[t, :] = self.act(rtrv, self.pars) # activation
            u_hat[t, :] = u_psb[t, :]*np.sum(act, self.pars) # prediction
            b_hat[t, :] = sim_resp_fun(u_hat[t, :], u_psb[t, :], sim_pars['resp_scale']) # response
            delta[t, :] = u[t, :] - u_hat[t, :] # prediction error
            encd[t] = self.encd(x[t, :], x_mem[0:n_mem, :], delta, aux, t, sim_pars) # encoding
            if encd[t] == 1:
                n_mem += 1
                x_mem[n_mem, :] = x[t, :]
                u_mem[n_mem, :] = u_lrn[t, :]*u[t, :] 
            aux.update(sim_pars, n_u, n_x, t, x_mem, u_mem, u_psb, u_hat) # update auxiliary data
            
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
                                     'rtrv' : (['t', 'mem'], rtrv[:, range(n_mem)]),
                                     'act' : (['t', 'mem'], act[:, range(n_mem)]),
                                     'encd' : (['t'], encd),
                                     'x_mem' : (['mem', 'x_name'], x_mem[range(n_mem), :]),
                                     'u_mem' : (['mem', 'u_name'], u_mem[range(n_mem), :]),
                                     'u_hat' : (['t', 'u_name'], u_hat),
                                     'b_hat' : (['t', 'u_name'], b_hat),
                                     'b' : (['t', 'u_name'], b),
                                     'delta' : (['t', 'u_name'], delta)},
                        coords = {'t' : range(n_t),
                                  't_name' : ('t', trials.t_name),
                                  'trial' : ('t', trials.trial),
                                  'trial_name' : ('t', trials.trial_name),
                                  'stage' : ('t', trials.stage),
                                  'stage_name' : ('t', trials.stage_name),
                                  'mem' : range(n_mem),
                                  'x_name' : x_names,
                                  'u_name' : u_names},
                        attrs = {'model' : self.name,
                                 'model_class' : 'ib',
                                 'schedule' : trials.attrs['schedule'],
                                 'resp_type' : resp_type,
                                 'sim_pars' : sim_pars})
        ds = ds.expand_dims(ident = [ident], schedule = [trials.schedule.values])
        
        return ds
    
########## PARAMETERS ##########

par_names = ['resp_scale']; par_list = [{'min': 0.0, 'max': 10.0, 'default': 1.0}]
par_names += ['c']; par_list += [{'min': 0.0, 'max': 5.0, 'default': 1.0}]
par_names += ['k']; par_list += [{'min': 1, 'max': 10, 'default': 2}]
par_names += ['threshold']; par_list += [{'min': 0, 'max': 10, 'default': 1}]
pars = pd.DataFrame(par_list, index = par_names)
del par_names; del par_list