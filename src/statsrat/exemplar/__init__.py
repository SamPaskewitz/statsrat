import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats
from statsrat import resp_fun
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
        
    par_logit_transform(phi)
        Perform logit (inverse logistic) transformation on model parameters.
        
    par_logistic_transform(theta)        
        Perform logistic transform on numbers in (-infty, infty) to bring
        them back to the model's specified parameter range.
        
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
        self.par_names = list(np.unique(sim.par_names + rtrv.par_names + atn_update.par_names + y_ex_update.par_names))
        self.pars = pars.loc[self.par_names + ['resp_scale']]
        
    def simulate(self, trials, par_val = None, init_atn = 1.0, random_resp = False, ident = 'sim'):
        """
        Simulate a trial sequence once with known model parameters.
        
        Parameters
        ----------
        trials: dataset (xarray)
            Time step level experimental data (cues, outcomes etc.).

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

        Explanation of variables in ds
        ------------------------------
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
        x_ex = trials.x_ex # exemplar locations
        x_names = list(trials.x_name.values) # cue names
        y_names = list(trials.y_name.values) # outcome names
        ex_names = list(trials.ex_names) # exemplar names
        n_t = x.shape[0] # number of time points
        n_x = x.shape[1] # number of features
        n_y = y.shape[1] # number of outcomes/response options
        n_ex = len(ex_names) # number of exemplars
        sim = np.zeros((n_t, n_ex)) # similarity to exemplars
        rtrv = np.zeros((n_t, n_ex)) # exemplar retrieval strength
        y_ex = np.zeros((n_t + 1, n_ex, n_y)) # outcomes (u) associated with each exemplar
        atn = np.zeros((n_t + 1, n_ex, n_x)) # attention (can be different for each exemplar, but doesn't need to be)
        atn[0, :, :] = init_atn
        y_hat = np.zeros((n_t, n_y)) # outcome predictions
        b_hat = np.zeros((n_t, n_y)) # expected behavior
        has_x_dims = 'x_dims' in list(trials.attrs.keys())
        if has_x_dims:
            x_dims = trials.attrs['x_dims']
        else:
            x_dims = None
        ex_seen_yet = pd.Series(n_ex*[0], index = ex_names) # keeps track of which exemplars have been observed yet
        ex_counts = pd.Series(n_ex*[0], index = ex_names) # number of times each exemplar has been observed

        # set up response function (depends on response type)
        resp_dict = {'choice': resp_fun.choice,
                     'exct': resp_fun.exct,
                     'supr': resp_fun.supr}
        sim_resp_fun = resp_dict[trials.resp_type]

        # loop through time steps
        for t in range(n_t):
            ex_seen_yet[ex[t]] = 1 # note that current exemplar has been seen
            ex_counts[ex[t]] += 1            
            sim[t, :] = ex_seen_yet*self.sim(x[t, :], x_ex, atn[t, :, :], sim_pars) # similarity
            rtrv[t, :] = self.rtrv(sim[t, :], ex_counts, ex_seen_yet, sim_pars) # retrieval strength
            y_hat[t, :] = rtrv[t, :]@(y_psb[t, :]*y_ex[t, :, :]) # prediction
            b_hat[t, :] = sim_resp_fun(y_hat[t, :], y_psb[t, :], sim_pars['resp_scale']) # response
            y_ex[t + 1, :, :] = y_ex[t, :, :] + self.y_ex_update(sim[t, :], rtrv[t, :], y[t, :], y_hat[t, :], y_lrn[t, :], y_ex[t, :], ex_counts, n_ex, n_y, sim_pars) # update y_ex
            atn[t + 1, :, :] = atn[t, :] + self.atn_update(sim[t, :], x[t, :], y[t, :], y_psb[t, :], rtrv[t, :], y_hat[t, :], y_lrn[t, :], x_ex.values, y_ex[t, :, :], n_x, n_y, ex_seen_yet, ex_counts, n_ex, sim_pars) # update attention
            
        # generate simulated responses
        (b, b_index) = resp_fun.generate_responses(b_hat, random_resp, trials.resp_type)
        
        # put all simulation data into a single xarray dataset
        ds = trials.copy(deep = True)
        ds = ds.assign_coords({'ex_name' : ex_names, 'ident' : [ident]})
        ds = ds.assign({'y_psb' : (['t', 'y_name'], y_psb),
                        'y_lrn' : (['t', 'y_name'], y_lrn),
                        'y_hat' : (['t', 'y_name'], y_hat),
                        'b_hat' : (['t', 'y_name'], b_hat),
                        'b' : (['t', 'y_name'], b),
                        'y_ex' : (['t', 'ex_name', 'y_name'], y_ex[range(n_t), :, :]), # remove unnecessary last row
                        'atn': (['t', 'ex_name', 'x_name'], atn[range(n_t), :, :]),
                        'sim': (['t', 'ex_name'], sim),
                        'rtrv': (['t', 'ex_name'], rtrv)})
        ds = ds.assign_attrs({'model': self.name,
                              'model_class' : 'exemplar',
                              'sim_pars' : sim_pars})
        
        return ds
    
    def par_logit_transform(self, phi):
        '''
        Arguments
        ---------
        phi: array-like of floats
            Numbers to be transformed.
            
        Perform logit (inverse logistic) transformation on model parameters.
        
        Notes
        -----
        The methods par_logit_transform and par_logistic_transform are designed
        for use with hierarchical model fitting methods, which in this package use
        logit-normal priors.  That is, model parameters (denoted phi) are logit
        transformed into the space -infty, infty (the logit-transformed parameters
        are denoted theta) where they are given some variation on normal distribution
        prior.  Inference is done on the logit-transformed parameters (theta), and
        the logistic transformation can be used to transform them back to their
        original range (theta -> phi).
        '''
        theta = np.log(phi - self.pars['min']) - np.log(self.pars['max'] - phi)
        return theta
    
    def par_logistic_transform(self, theta):
        '''
        Arguments
        ---------
        theta: array-like of floats
            Numbers to be transformed.
        
        Perform logistic transform on numbers in (-infty, infty) to bring
        them back to the model's specified parameter range.
        
        Notes
        -----
        The methods par_logit_transform and par_logistic_transform are designed
        for use with hierarchical model fitting methods, which in this package use
        logit-normal priors.  That is, model parameters (denoted phi) are logit
        transformed into the space -infty, infty (the logit-transformed parameters
        are denoted theta) where they are given some variation on normal distribution
        prior.  Inference is done on the logit-transformed parameters (theta), and
        the logistic transformation can be used to transform them back to their
        original range (theta -> phi).
        '''
        phi = self.pars['min'] + (self.pars['max'] - self.pars['min'])/(1 + np.exp(-theta))
        return phi

########## PARAMETERS ##########
par_names = []; par_list = []   
par_names += ['lrate_par']; par_list += [{'min': 0.0, 'max': 1.0, 'default': 0.5}]
par_names += ['atn_lrate_par']; par_list += [{'min': 0.0, 'max': 1.0, 'default': 0.5}] # learning rate for attention updates
par_names += ['decay_rate']; par_list += [{'min': 0.0, 'max': 10.0, 'default': 0.5}]
par_names += ['nu']; par_list += [{'min': 0.0, 'max': 10.0, 'default': 0.0}] # extra counts for ex_mean
par_names += ['resp_scale']; par_list += [{'min': 0.0, 'max': 10.0, 'default': 1.0}]
pars = pd.DataFrame(par_list, index = par_names)
del par_names; del par_list