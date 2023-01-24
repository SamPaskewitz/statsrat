import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats
from statsrat.rw import fbase
from statsrat import resp_fun
from numpy.linalg import cond
from scipy.linalg import solve
from copy import deepcopy
from . import link, tausq_inv_dist

class model:
    '''
    Class for Bayesian generalized linear regression models.

    Attributes
    ----------
    name: str
        Model name.
    fbase: function
        Base mapping between cues (x) and features (f_x).
        These functions are borrow from the rw (Rescorla-Wagner) submodule.
    link: object
        Specifies distribution of u (observed outcomes) as a function of
        z (normally distributed latent variable).
    tausq_inv_dist: function
            Determines distribution of tausq_inv (prior precision for regression weights).
    par_names: list
        Names of the model's free parameters (strings).
    pars: dict
        Information about model parameters.

    Methods
    -------
    simulate(trials, par_val = None, random_resp = False, ident = 'sim)
        Simulate a trial sequence once with known model parameters.
        
    Notes
    -----
    The observed variable (y) has a distribution that depends on
    a latent variable (z).  z is normally distributed with mean x^T w,
    just as in linear regression.
    .. math::
        z &\sim \matcha{N}(x^T w, \sigma^2) \\
        y &\sim \text{y\_dist}(z)
    
    If y = z, then we have straightforward linear regression.
    However, we can also have probit and censored regression schemes
    for different choices of 'link'.
    
    Learning proceeds via a streaming variational Bayes algorithm
    with a batch size of 1 (i.e. hyperparameters are updated after each
    observation).  We use the canonical (i.e. natural) form of each
    distribution, so updating hyperparameters is merely a matter of
    adding sufficient statistics.
    
    Regression weights (w) have a normal prior with mean 0 and precision
    tausq_inv (hence variance tausq).  Using variational Bayes, we can
    estimate tausq for each feature, i.e. perform automatic relevance
    determination.  Whether or not we do this and what assumptions we
    make are determined by the 'tausq_inv_dist' attribute.
    
    Relevant Papers
    ---------------
    Broderick, T., Boyd, N., Wibisono, A., Wilson, A. C., & Jordan, M. I. (2013).
    Streaming variational Bayes.
    ArXiv Preprint ArXiv:1307.6769.
    
    Dayan, P., & Kakade, S. (2001).
    Explaining Away in Weight Space.
    Advances in Neural Information Processing Systems, 451–457.
    '''

    def __init__(self, name, fbase, link, tausq_inv_dist):
        """
        Parameters
        ----------
        name : str
            Model name.
        fbase : function
            Base mapping between cues (x) and features (f_x).
            These functions are borrowed from the rw (Rescorla-Wagner) submodule.
        link: object
            Specifies distribution of y (observed outcomes) as a function of
            z (normally distributed latent variable).
        tausq_inv_dist: function
            Determines distribution of tausq_inv (prior precision for regression weights).
        """
        # add attributes to object ('self')
        self.name = name
        self.fbase = fbase
        self.link = link
        self.tausq_inv_dist = tausq_inv_dist
        # determine model's parameter space
        par_list = [elm for elm in [fbase.pars, link.pars, tausq_inv_dist.pars, pd.DataFrame({'min': 0.0, 'max': 10.0, 'default': 1.0}, index = ['resp_scale'])] if elm is not None] # create list of par dataframes, excluding None
        self.pars = pd.concat(par_list)
        self.pars = self.pars.loc[~self.pars.index.duplicated()].sort_index()
        self.par_names = self.pars.index.values
 
    def simulate(self, trials, par_val = None, rich_output = True, random_resp = False, ident = 'sim'):
        """
        Simulate a trial sequence once with known model
        parameters.
        
        Parameters
        ----------
        trials : data frame
            Time step level experimental data (cues, outcomes etc.).
            
        par_val : list, optional
            Learning model parameters (floats or ints).
            
        rich_output: Boolean, optional
            Whether to output full simulation data (True) or just
            responses, i.e. b/b_hat (False).  Defaults to True.

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
        
        Explanation of variables in ds
        ------------------------------
        f_name: feature names
        y_psb: indicator vector for outcomes (y) that are possible on the trial (from the learner's perspective)
        y_lrn: indicator vector for outcomes (y) for which there is feedback and hence learning will occur
        f_x: feature vectors
        z_hat: predicted latent variable (z) before observing outcome (y)
        y_hat: outcome predictions
        b_hat: expected value of behavioral response
        b: vector representing actual behavioral response (identical to b_hat unless the random_resp argument is set to True)
        shrink_cond: condition number of the shrinkage matrix
        mean_tausq_inv: prior/posterior mean of the precision matrix
        mean_tausq: prior/posterior mean of the variance matrix
        mean_w: prior/posterior mean of weights (w)
        var_w: prior/posterior variance of weights (w)
        mean_wsq: prior/posterior mean of squared weights (w**2)
        b_index: index of behavioral response (only present if response type is 'choice' and random_resp is True)
        b_name: name of behavioral response (only present if response type is 'choice' and random_resp is True)

        Notes
        -----
        The response type is determined by the 'resp_type' attribute of the 'trials' object.
        
        The response type 'choice' is used for discrete response options.  This
        produces response probabilities using a softmax function:
        .. math:: \text{resp}_i = \frac{ e^{\phi \hat{y}_i} }{ \sum_j e^{\phi \hat{y}_j} }

        The response type 'exct' is used for excitatory Pavlovian
        conditioning:
        .. math:: \text{resp} = \frac{ e^{\phi \hat{y}_i} }{ e^{\phi \hat{y}_i} + 1 }

        The response type 'supr' (suppression) is used for inhibitory
        Pavlovian conditioning:
        .. math:: \text{resp} = \frac{ e^{-\phi \hat{y}_i} }{ e^{-\phi \hat{y}_i} + 1 }

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
            
        # various arrays etc.
        x = np.array(trials['x'], dtype = 'float64')
        y = np.array(trials['y'], dtype = 'float64')
        y_psb = np.array(trials['y_psb'], dtype = 'float64')
        y_lrn = np.array(trials['y_lrn'], dtype = 'float64')
        x_names = list(trials.x_name.values)
        y_names = list(trials.y_name.values)
        (f_x, f_names) = self.fbase(x, x_names, sim_pars).values() # features and feature names
        # count things
        n = {'t': x.shape[0], # number of time points
             'y': y.shape[1], # number of outcomes/response options
             'f': f_x.shape[1]} # number of features
        
        # set up array for mean response (b_hat)
        b_hat = np.zeros((n['t'], n['y']))
        # initialize state
        state = {'y_hat': np.zeros(n['y']), # outcome predictions
                 'shrink_cond': np.zeros(n['y']), # condition number of a matrix relevant to shrinkage
                 'hpar0_w': np.zeros((n['f'], n['y'])), # hyperparameter for the variational distribution of w (precision * mean)
                 'hpar1_w': np.zeros((n['f'], n['f'], n['y'])), # hyperparameter for the variational distribution of w (precision)
                 'sufstat0_w': np.zeros((n['f'], n['y'])), # sufficient statistic for w
                 'sufstat1_w': np.zeros((n['f'], n['f'], n['y'])), # other sufficient statistic for w
                 'mean_tausq_inv': np.zeros((n['f'], n['y'])), # variational mean of 1/tau^2
                 'mean_tausq': np.zeros((n['f'], n['y'])), # variational mean of tau^2 (solely for analysis purposes)
                 'mean_w': np.zeros((n['f'], n['y'])), # variational mean of w (a.k.a. mu)
                 'var_w': np.zeros((n['f'], n['y'])), # variational variance of w (i.e. the diagonal elements of the covariance matrix)
                 'mean_wsq': np.zeros((n['f'], n['y'])), # variational mean of w^2
                 'z_hat': np.zeros(n['y']), # predicted latent variable (z) before observing outcome (y)
                 'mean_z': np.zeros(n['y']), # variational mean of the latent variable z (after observing y)
                 'y_psb_so_far': np.zeros(n['y']) # keeps track of which outcomes have been encountered as possible so far
                }
        state_history = []
        # initialize tausq_inv_dist and link function
        tausq_inv_dist = self.tausq_inv_dist(n['y'], n['f'], sim_pars)
        link = self.link(n['y'], n['f'])
        # determine value of z_var (variance of the latent variable z)
        if 'y_var' in self.pars.index:
            # In this case, y = z so y_var = z_var.
            z_var = sim_pars['y_var']
        else:
            # Used for probit regression.
            z_var = 1

        # set up response function (depends on response type)
        resp_dict = {'choice': resp_fun.choice,
                     'exct': resp_fun.exct,
                     'supr': resp_fun.supr,
                     'normal': resp_fun.normal,
                     'log_normal': resp_fun.log_normal}
        response = resp_dict[trials.resp_type]
        
        # loop through time steps
        for t in range(n['t']):
            env = {'x': x[t, :], 'y': y[t, :], 'y_psb': y_psb[t, :], 'y_lrn': y_lrn[t, :]}
            # compute means of tausq and tausq_inv
            state['mean_tausq_inv'] = tausq_inv_dist.mean_tausq_inv()
            state['mean_tausq'] = tausq_inv_dist.mean_tausq()
            # compute hpar_w and related quantities using mean_tausq_inv
            for j in range(n['y']):
                # keep track of which outcomes have been observed so far
                if env['y_psb'][j] == 1:
                    state['y_psb_so_far'][j] = 1
                    
                calculate = env['y_psb'][j] == 1
                if calculate:
                    # mean prior precision matrix
                    T = np.diag(state['mean_tausq_inv'][:, j])
                    # compute hpar_w
                    state['hpar0_w'][:, j] = state['sufstat0_w'][:, j]
                    state['hpar1_w'][:, :, j] = T + state['sufstat1_w'][:, :, j] # precision matrix
                    # compute mean_w
                    P_inv = np.diag(1/np.diag(state['hpar1_w'][:, :, j])) # Jacobi preconditioning matrix
                    state['shrink_cond'][j] = cond(P_inv@state['hpar1_w'][:, :, j]) # condition number
                    state['mean_w'][:, j] = solve(P_inv@state['hpar1_w'][:, :, j], P_inv@state['hpar0_w'][:, j])
                    # compute var_w and mean_wsq
                    state['var_w'][:, j] = 1/np.diag(state['hpar1_w'][:, :, j])
                    state['mean_wsq'][:, j] = state['var_w'][:, j] + state['mean_w'][:, j]**2
            # update tausq_inv_dist using mean_wsq
            tausq_inv_dist.update(state['mean_wsq'], state['y_psb_so_far'])
            # predict y (outcome) and compute b (behavior)
            state['z_hat'] = env['y_psb']*(f_x[t, :]@state['mean_w']) # predicted value of latent variable (z)
            state['y_hat'] = link.y_hat(state['z_hat'], env['y_psb'], f_x[t, :], state['hpar1_w']) # predicted value of outcome (y)
            b_hat[t, :] = response.mean(state['y_hat'], env['y_psb'], sim_pars['resp_scale']) # response
            state['mean_z'] = link.mean_z(state['z_hat'], env['y'], env['y_psb']) # mean of z after observing y
            state_history += [deepcopy(state)] # record a copy of the current state before learning occurs
            # update sufficient statistics of x and y for estimating w
            f = f_x[t, :].squeeze() # for convenience
            for j in range(n['y']):
                update = env['y_lrn'][j] == 1
                if update:
                    state['sufstat0_w'][:, j] += (f*state['mean_z'][j])/z_var
                    state['sufstat1_w'][:, :, j] += np.outer(f, f)/z_var

        # generate simulated responses
        if random_resp:
            (b, b_index) = response.random(b_hat, sim_pars['resp_scale'])
        else:
            b = b_hat
            b_index = None
        
        if rich_output:
            # put all simulation data into a single xarray dataset
            ds = trials.copy(deep = True)
            ds = ds.assign_coords({'f_name' : f_names, 'ident' : [ident]})
            ds = ds.assign({'b_hat' : (['t', 'y_name'], b_hat),
                            'b' : (['t', 'y_name'], b),
                            'f_x' : (['t', 'f_name'], f_x),
                            'z_hat' : (['t', 'y_name'], np.zeros((n['t'], n['y']))),
                            'y_hat' : (['t', 'y_name'], np.zeros((n['t'], n['y']))),
                            'shrink_cond' : (['t', 'y_name'], np.zeros((n['t'], n['y']))),
                            'mean_tausq_inv' : (['t', 'f_name', 'y_name'], np.zeros((n['t'], n['f'], n['y']))),
                            'mean_tausq' : (['t', 'f_name', 'y_name'], np.zeros((n['t'], n['f'], n['y']))),
                            'mean_w' : (['t', 'f_name', 'y_name'], np.zeros((n['t'], n['f'], n['y']))),
                            'var_w' : (['t', 'f_name', 'y_name'], np.zeros((n['t'], n['f'], n['y']))),
                            'mean_z' : (['t', 'y_name'], np.zeros((n['t'], n['y']))),
                            'mean_wsq' : (['t', 'f_name', 'y_name'], np.zeros((n['t'], n['f'], n['y'])))})
            for t in range(n['t']): # fill out the xarray dataset from state_history
                for var in ['z_hat', 'y_hat', 'shrink_cond', 'mean_tausq_inv', 'mean_tausq', 'mean_w', 'var_w', 'mean_z', 'mean_wsq']:
                    ds[var].loc[{'t': t}] = state_history[t][var]
            ds = ds.assign_attrs({'model': self.name,
                                  'model_class': 'bayes_regr',
                                  'sim_pars': sim_pars.values})
        else:
            # FOR NOW (until I revise how log-likelihood calculations work) just put b and b_hat in a dataset
            ds = trials.copy(deep = True)
            ds = ds.assign({'b_hat' : (['t', 'y_name'], b_hat),
                            'b' : (['t', 'y_name'], b)})
        return ds