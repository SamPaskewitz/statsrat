import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats
from statsrat.rw import fbase
from statsrat import resp_fun
from numpy.linalg import cond
from scipy.linalg import solve
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
        self.pars = pd.concat(par_list).drop_duplicates().sort_index()
        self.par_names = self.pars.index.values
 
    def simulate(self, trials, par_val = None, random_resp = False, ident = 'sim'):
        """
        Simulate a trial sequence once with known model
        parameters.
        
        Parameters
        ----------
        trials : data frame
            Time step level experimental data (cues, outcomes etc.).
            
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
        n_t = f_x.shape[0] # number of time points
        n_f = f_x.shape[1] # number of features
        n_y = y.shape[1] # number of outcomes/response options
        y_hat = np.zeros((n_t, n_y)) # outcome predictions
        b_hat = np.zeros((n_t, n_y)) # expected behavior
        shrink_cond = np.zeros((n_t, n_y)) # condition number of a matrix relevant to shrinkage
        # arrays for variational distribution hyperparameters and objects
        tausq_inv_dist = self.tausq_inv_dist(n_y, n_f, sim_pars)
        link = self.link(n_y, n_f)
        hpar0_w = np.zeros((n_t + 1, n_f, n_y)) # hyperparameter for the variational distribution of w (precision * mean)
        hpar1_w = np.zeros((n_t + 1, n_f, n_f, n_y)) # hyperparameter for the variational distribution of w (precision)
        # array for sufficient statistics need to estimate w
        sufstat0_w = np.zeros((n_t + 1, n_f, n_y)) # sufficient statistic for w
        sufstat1_w = np.zeros((n_t + 1, n_f, n_f, n_y)) # sufficient statistic for w
        # arrays for variational means and variances
        mean_tausq_inv = np.zeros((n_t, n_f, n_y)) # variational mean of 1/tau^2
        mean_tausq = np.zeros((n_t, n_f, n_y)) # variational mean of tau^2 (solely for analysis purposes)
        mean_w = np.zeros((n_t, n_f, n_y)) # variational mean of w (a.k.a. mu)
        var_w = np.zeros((n_t, n_f, n_y)) # variational variance of w (i.e. the diagonal elements of the covariance matrix)
        mean_wsq = np.zeros((n_t, n_f, n_y)) # variational mean of w^2
        z_hat = np.zeros((n_t, n_y)) # predicted latent variable (z) before observing outcome (y)
        mean_z = np.zeros((n_t, n_y)) # variational mean of the latent variable z (after observing y)
        y_psb_so_far = np.zeros(n_y) # keeps track of which outcomes have been encountered as possible so far
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
                     'supr': resp_fun.supr}
        sim_resp_fun = resp_dict[trials.resp_type]
        
        # loop through time steps
        for t in range(n_t):
            # compute means of tausq and tausq_inv
            mean_tausq_inv[t, :, :] = tausq_inv_dist.mean_tausq_inv()
            mean_tausq[t, :, :] = tausq_inv_dist.mean_tausq()
            # compute hpar_w and related quantities using mean_tausq_inv
            for j in range(n_y):
                # keep track of which outcomes have been observed so far
                if y_psb[t, j] == 1:
                    y_psb_so_far[j] = 1
                    
                calculate = y_psb[t, j] == 1
                if calculate:
                    # mean prior precision matrix
                    T = np.diag(mean_tausq_inv[t, :, j])
                    # compute hpar_w
                    hpar0_w[t:n_t, :, j] = sufstat0_w[t, :, j]
                    hpar1_w[t:n_t, :, :, j] = T + sufstat1_w[t, :, :, j] # precision matrix
                    # compute mean_w
                    P_inv = np.diag(1/np.diag(hpar1_w[t, :, :, j])) # Jacobi preconditioning matrix
                    shrink_cond[t, j] = cond(P_inv@hpar1_w[t, :, :, j]) # condition number
                    mean_w[t:n_t, :, j] = solve(P_inv@hpar1_w[t, :, :, j], P_inv@hpar0_w[t, :, j])
                    # compute var_w and mean_wsq
                    var_w[t:n_t, :, j] = 1/np.diag(hpar1_w[t, :, :, j])
                    mean_wsq[t:n_t, :, j] = var_w[t, :, j] + mean_w[t, :, j]**2
            # update tausq_inv_dist using mean_wsq
            tausq_inv_dist.update(mean_wsq[t, :, :], y_psb_so_far)
            # predict y (outcome) and compute b (behavior)
            z_hat[t, :] = y_psb[t, :]*(f_x[t, :]@mean_w[t, :, :]) # predicted value of latent variable (z)
            y_hat[t, :] = link.y_hat(z_hat[t, :], y_psb[t, :], f_x[t, :], hpar1_w[t, :, :, :]) # predicted value of outcome (y)
            b_hat[t, :] = sim_resp_fun(y_hat[t, :], y_psb[t, :], sim_pars['resp_scale']) # response
            mean_z[t, :] = link.mean_z(z_hat[t, :], y[t, :], y_psb[t, :]) # mean of z after observing y
            # update sufficient statistics of x and y for estimating w
            f = f_x[t, :].squeeze() # for convenience
            for j in range(n_y):
                update = y_lrn[t, j] == 1
                if update:
                    sufstat0_w[(t+1):n_t, :, j] = sufstat0_w[t, :, j] + (f*mean_z[t, j])/z_var
                    sufstat1_w[(t+1):n_t, :, :, j] = sufstat1_w[t, :, :, j] + np.outer(f, f)/z_var

        # generate simulated responses
        (b, b_index) = resp_fun.generate_responses(b_hat, random_resp, trials.resp_type)
        
        # put all simulation data into a single xarray dataset
        ds = trials.copy(deep = True)
        ds = ds.assign_coords({'f_name' : f_names, 'ident' : [ident]})
        ds = ds.assign({'y_psb' : (['t', 'y_name'], y_psb),
                        'y_lrn' : (['t', 'y_name'], y_lrn),
                        'f_x' : (['t', 'f_name'], f_x),
                        'z_hat' : (['t', 'y_name'], z_hat),
                        'y_hat' : (['t', 'y_name'], y_hat),
                        'b_hat' : (['t', 'y_name'], b_hat),
                        'b' : (['t', 'y_name'], b),
                        'shrink_cond' : (['t', 'y_name'], shrink_cond),
                        'mean_tausq_inv' : (['t', 'f_name', 'y_name'], mean_tausq_inv),
                        'mean_tausq' : (['t', 'f_name', 'y_name'], mean_tausq),
                        'mean_w' : (['t', 'f_name', 'y_name'], mean_w),
                        'var_w' : (['t', 'f_name', 'y_name'], var_w),
                        'mean_z' : (['t', 'y_name'], mean_z),
                        'mean_wsq' : (['t', 'f_name', 'y_name'], mean_wsq)})
        ds = ds.assign_attrs({'model': self.name,
                              'model_class' : 'bayes_regr',
                              'sim_pars' : sim_pars})

        return ds