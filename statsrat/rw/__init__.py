import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats
from statsrat import resp_fun
from . import fbase, fweight, aux, lrate

class model:
    '''
    Class for Rescorla-Wagner family learning models.

    Attributes
    ----------
    name: str
        Model name.
    fbase : function
        Base mapping between cues (x) and features (f_x).
    fweight: function
        Attentional weights for features.
    lrate: function
        Determines learning rates.
    aux: object
        Auxilliary learning, e.g. attention or weight covariance.
    par_names: list
        Names of the model's free parameters (strings).

    Methods
    -------
    simulate(trials, resp_type = 'choice', par_val = None, random_resp = False, ident = 'sim')
        Simulate a trial sequence once with known model parameters.
    '''

    def __init__(self, name, fbase, fweight, lrate, aux):
        """
        Parameters
        ----------
        name: str
            Model name.
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
        self.fbase = fbase
        self.fweight = fweight
        self.lrate = lrate
        self.aux = aux
        # determine model's parameter space
        par_names = list(np.unique(fbase.par_names + fweight.par_names + lrate.par_names + aux.par_names))
        self.pars = pars.loc[par_names + ['resp_scale']]
 
    def simulate(self, trials, resp_type = 'choice', par_val = None, random_resp = False, ident = 'sim'):
        """
        Simulate a trial sequence once with known model
        parameters.
        
        Parameters
        ----------
        trials: dataset (xarray)
            Time step level experimental data (cues, outcomes etc.).

        resp_type: str, optional
            Type of behavioral response: one of 'choice', 'exct' or 'supr'.
            Defaults to 'choice'.

        par_val: list, optional
            Learning model parameters (floats or ints).

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
        x_names = list(trials.x_name.values)
        u_names = list(trials.u_name.values)
        (fbase, f_names) = self.fbase(x, x_names).values() # features and feature names
        n_t = fbase.shape[0] # number of time points
        n_f = fbase.shape[1] # number of features
        n_u = u.shape[1] # number of outcomes/response options
        fweight = np.zeros((n_t, n_f))
        f_x = np.zeros((n_t, n_f))
        u_hat = np.zeros((n_t, n_u)) # outcome predictions
        b_hat = np.zeros((n_t, n_u)) # expected behavior
        delta = np.zeros((n_t, n_u)) # prediction error
        w = np.zeros((n_t + 1, n_f, n_u))
        lrate = np.zeros((n_t, n_f, n_u))
        aux = self.aux(sim_pars, n_t, n_f, n_u)

        # set up response function (depends on response type)
        resp_dict = {'choice': resp_fun.choice,
                     'exct': resp_fun.exct,
                     'supr': resp_fun.supr}
        sim_resp_fun = resp_dict[resp_type]

        # loop through time steps
        for t in range(n_t):
            fweight[t, :] = self.fweight(aux, t, fbase, fweight, n_f, sim_pars)
            f_x[t, :] = fbase[t, :] * fweight[t, :] # weight base features
            u_hat[t, :] = u_psb[t, :] * (f_x[t, :] @ w[t, :, :]) # prediction
            b_hat[t, :] = sim_resp_fun(u_hat[t, :], u_psb[t, :], sim_pars['resp_scale']) # response
            delta[t, :] = u[t, :] - u_hat[t, :] # prediction error
            aux.update(sim_pars, n_u, n_f, t, fbase, fweight, u_psb, u_hat, delta, w) # update auxiliary data (e.g. attention weights, or Kalman filter covariance matrix)
            lrate[t, :, :] = self.lrate(aux, t, fbase, fweight, n_f, n_u, sim_pars) # learning rates for this time step
            w[t+1, :, :] = w[t, :, :] + u_lrn[t, :] * lrate[t, :, :] * delta[t, :].reshape((1, n_u)) # association learning

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
                                     'fbase' : (['t', 'f_name'], fbase),
                                     'f_x' : (['t', 'f_name'], f_x),
                                     'u_hat' : (['t', 'u_name'], u_hat),
                                     'b_hat' : (['t', 'u_name'], b_hat),
                                     'b' : (['t', 'u_name'], b),
                                     'w' : (['t', 'f_name', 'u_name'], w[range(n_t), :, :]), # remove unnecessary last row from w
                                     'delta' : (['t', 'u_name'], delta),
                                     'lrate' : (['t', 'f_name', 'u_name'], lrate)},
                        coords = {'t' : range(n_t),
                                  't_name' : ('t', trials.t_name),
                                  'trial' : ('t', trials.trial),
                                  'trial_name' : ('t', trials.trial_name),
                                  'stage' : ('t', trials.stage),
                                  'stage_name' : ('t', trials.stage_name),
                                  'x_name' : x_names,
                                  'f_name' : f_names,
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
par_names += ['lrate']; par_list += [{'min': 0.0, 'max': 1.0, 'default': 0.2}]
par_names += ['lrate_atn']; par_list += [{'min': 0.0, 'max': 2.0, 'default': 0.2}] # max was previously 1
par_names += ['extra_counts']; par_list += [{'min': 1.0, 'max': 10.0, 'default': 5.0}]
par_names += ['metric']; par_list += [{'min': 0.1, 'max': 10, 'default': 2}] # min was previously 1
par_names += ['atn_min']; par_list += [{'min': 0.0, 'max': 1.0, 'default': 0.1}]
par_names += ['a0']; par_list += [{'min': 0.0, 'max': 1.0, 'default': 0.5}]
par_names += ['eta0']; par_list += [{'min': 0.0, 'max': 10.0, 'default': 1}] # max is 10 in the R version used for the spring 2020 FAST analysis
par_names += ['w_var0']; par_list += [{'min' : 0.0, 'max' : 10.0, 'default' : 1.0}] # initial weight variance for Kalman filter
par_names += ['u_var']; par_list += [{'min' : 0.0, 'max' : 5.0, 'default' : 0.1}] # outcome variance for Kalman filter
par_names += ['drift_var']; par_list += [{'min' : 0.0, 'max' : 2.0, 'default' : 0.01}] # drift variance for Kalman filter
pars = pd.DataFrame(par_list, index = par_names)
del par_names; del par_list