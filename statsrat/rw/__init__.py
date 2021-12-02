import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats
from statsrat import resp_fun
from . import pred, fbase, fweight, aux, lrate, drate

class model:
    '''
    Class for Rescorla-Wagner family learning models.

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
    drate: function
        Determines decay rates.
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
        
    Relevant Papers
    ---------------
    Dayan, P., & Kakade, S. (2001).
    Explaining Away in Weight Space.
    Advances in Neural Information Processing Systems, 451–457.
    
    Gluck, M. A., & Bower, G. H. (1988).
    Evaluating an adaptive network model of human learning.
    Journal of Memory and Language, 27(2), 166–195.
    
    Kruschke, J. K. (2001).
    Toward a Unified Model of Attention in Associative Learning.
    Journal of Mathematical Psychology, 45(6), 812–863.
    
    Paskewitz, S., & Jones, M. (2020).
    Dissecting EXIT.
    Journal of Mathematical Psychology, 97, 102371.
    
    Rescorla, R. A., & Wagner, A. R. (1972).
    A Theory of Pavlovian Conditioning: Variations in the
    Effectiveness of Reinforcement and Nonreinforcement.
    Classical Conditioning II: Current Research and Theory, 2, 64–99.
    '''

    def __init__(self, name, pred, fbase, fweight, lrate, drate, aux):
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
        drate: function
            Determines decay rates.
        aux: object
            Auxilliary learning, e.g. attention or weight covariance.
        """
        # add attributes to object ('self')
        self.name = name
        self.pred = pred
        self.fbase = fbase
        self.fweight = fweight
        self.lrate = lrate
        self.drate = drate
        self.aux = aux
        # determine model's parameter space
        self.par_names = list(np.unique(pred.par_names + fbase.par_names + fweight.par_names + lrate.par_names + drate.par_names + aux.par_names))
        self.pars = pars.loc[self.par_names + ['resp_scale']]
 
    def simulate(self, trials, par_val = None, random_resp = False, ident = 'sim'):
        """
        Simulate a trial sequence once with known model parameters.
        
        Parameters
        ----------
        trials: dataset (xarray)
            Time step level experimental data (cues, outcomes etc.).

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
        drate: decay rates
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
        x = np.array(trials['x'], dtype = 'float64')
        y = np.array(trials['y'], dtype = 'float64')
        y_psb = np.array(trials['y_psb'], dtype = 'float64')
        y_lrn = np.array(trials['y_lrn'], dtype = 'float64')
        x_names = list(trials.x_name.values)
        y_names = list(trials.y_name.values)
        (fbase, f_names) = self.fbase(x, x_names).values() # features and feature names
        n_t = fbase.shape[0] # number of time points
        n_x = x.shape[1] # number of cues/stimulus elements
        n_f = fbase.shape[1] # number of features
        n_y = y.shape[1] # number of outcomes/response options
        fweight = np.zeros((n_t, n_f))
        f_x = np.zeros((n_t, n_f))
        y_hat = np.zeros((n_t, n_y)) # outcome predictions
        b_hat = np.zeros((n_t, n_y)) # expected behavior
        delta = np.zeros((n_t, n_y)) # prediction error
        w = np.zeros((n_t + 1, n_f, n_y))
        lrate = np.zeros((n_t, n_f, n_y))
        drate = np.zeros((n_t, n_f, n_y))
        has_x_dims = 'x_dims' in list(trials.attrs.keys())
        if has_x_dims:
            x_dims = trials.attrs['x_dims']
        else:
            x_dims = None
        aux = self.aux(sim_pars, n_t, n_x, n_f, n_y, f_names, x_dims)

        # set up response function (depends on response type)
        resp_dict = {'choice': resp_fun.choice,
                     'exct': resp_fun.exct,
                     'supr': resp_fun.supr}
        sim_resp_fun = resp_dict[trials.resp_type]

        # loop through time steps
        for t in range(n_t):
            fweight[t, :] = self.fweight(aux, t, fbase, fweight, n_f, sim_pars)
            f_x[t, :] = fbase[t, :]*fweight[t, :] # weight base features
            y_hat[t, :] = self.pred(y_psb[t, :]*(f_x[t, :]@w[t, :, :]), sim_pars) # prediction
            b_hat[t, :] = sim_resp_fun(y_hat[t, :], y_psb[t, :], sim_pars['resp_scale']) # response
            delta[t, :] = y_lrn[t, :]*(y[t, :] - y_hat[t, :]) # prediction error
            aux.update(sim_pars, n_y, n_f, t, fbase, fweight, f_x[t, :], y_psb, y_hat, delta, w) # update auxiliary data (e.g. attention weights, or Kalman filter covariance matrix)
            lrate[t, :, :] = self.lrate(aux, t, fbase, fweight, n_f, n_y, sim_pars) # learning rates for this time step
            drate[t, :, :] = self.drate(aux, t, w, n_f, n_y, sim_pars) # decay rates for this time step
            w[t+1, :, :] = w[t, :, :] + y_lrn[t, :]*lrate[t, :, :]*delta[t, :].reshape((1, n_y)) - drate[t, :, :]*w[t, :, :] # association learning

        # generate simulated responses
        (b, b_index) = resp_fun.generate_responses(b_hat, random_resp, trials.resp_type)
        
        # put all simulation data into a single xarray dataset
        ds = trials.copy(deep = True)
        ds = ds.assign_coords({'f_name' : f_names, 'ident' : [ident]})
        ds = ds.assign({'y_psb' : (['t', 'y_name'], y_psb),
                        'y_lrn' : (['t', 'y_name'], y_lrn),
                        'fbase' : (['t', 'f_name'], fbase),
                        'fweight' : (['t', 'f_name'], fweight),
                        'f_x' : (['t', 'f_name'], f_x),
                        'y_hat' : (['t', 'y_name'], y_hat),
                        'b_hat' : (['t', 'y_name'], b_hat),
                        'b' : (['t', 'y_name'], b),
                        'w' : (['t', 'f_name', 'y_name'], w[range(n_t), :, :]), # remove unnecessary last row from w
                        'delta' : (['t', 'y_name'], delta),
                        'lrate' : (['t', 'f_name', 'y_name'], lrate),
                        'drate' : (['t', 'f_name', 'y_name'], drate)})
        ds = ds.assign_attrs({'model': self.name,
                              'model_class' : 'rw',
                              'sim_pars' : sim_pars})
        ds = aux.add_data(ds) # add extra data from aux
        if random_resp and trials.resp_type == 'choice':
            ds = ds.assign({'b_index': (['t'], b_index),
                            'b_name': (['t'], np.array(y_names)[b_index])})
        
        return ds

########## PARAMETERS ##########
par_names = []; par_list = []
par_names += ['feature_count_window']; par_list += [{'min': 0.0, 'max': 100, 'default': 20}]
par_names += ['lrate']; par_list += [{'min': 0.0, 'max': 1.0, 'default': 0.2}]
par_names += ['lrate_min']; par_list += [{'min': 0.0, 'max': 0.5, 'default': 0.1}]
par_names += ['drate']; par_list += [{'min': 0.0, 'max': 0.5, 'default': 0.25}]
par_names += ['lrate_atn']; par_list += [{'min': 0.0, 'max': 2.0, 'default': 0.2}]
par_names += ['lrate_atn0']; par_list += [{'min': 0.0, 'max': 2.0, 'default': 0.2}]
par_names += ['lrate_atn1']; par_list += [{'min': 0.0, 'max': 2.0, 'default': 0.2}]
par_names += ['drate_atn']; par_list += [{'min': 0.0, 'max': 2.0, 'default': 0.2}] # decay rate for attention
par_names += ['lrate_tau']; par_list += [{'min': 0.0, 'max': 1.0, 'default': 0.2}] # for tdrva
par_names += ['tau0']; par_list += [{'min': 0.01, 'max': 1.0, 'default': 0.5}] # for tdrva
par_names += ['power']; par_list += [{'min': 0.0, 'max': 2.0, 'default': 0.5}]
par_names += ['metric']; par_list += [{'min': 0.1, 'max': 10, 'default': 2}] # min is 0.1 in the R version, but this doesn't work here
par_names += ['atn_min']; par_list += [{'min': 0.0, 'max': 1.0, 'default': 0.1}]
par_names += ['atn0']; par_list += [{'min': 0.0, 'max': 1.0, 'default': 0.5}]
par_names += ['eta0']; par_list += [{'min': 0.0, 'max': 10.0, 'default': 1}] # max is 10 in the R version used for the spring 2020 FAST analysis
par_names += ['w_var0']; par_list += [{'min' : 0.0, 'max' : 10.0, 'default' : 1.0}] # initial weight variance for Kalman filter
par_names += ['y_var']; par_list += [{'min' : 0.0, 'max' : 5.0, 'default' : 0.1}] # outcome variance for Kalman filter
par_names += ['drift_var']; par_list += [{'min' : 0.0, 'max' : 2.0, 'default' : 0.01}] # drift variance for Kalman filter
par_names += ['resp_scale']; par_list += [{'min': 0.0, 'max': 10.0, 'default': 1.0}]
pars = pd.DataFrame(par_list, index = par_names)
del par_names; del par_list