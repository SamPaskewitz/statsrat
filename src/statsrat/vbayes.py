import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats
from plotnine import *
import nlopt
import multiprocessing
import numdifftools as nd
from time import perf_counter
from numpy.random import default_rng

# FINISH UPDATING EVERYTHING

def _compute_Hessian_diag_(model, ds, theta):
    """
    For interal use.  Computes the Hessian of p(theta_i | D_i, rho, beta) across individuals
    with a given value of theta for each person.
    """
    
    

def fit_vbayes(model, ds, fixed_pars = None, X = None, max_vb_iter = 5, global_maxeval = 200, local_maxeval = 1000, local_tolerance = 0.05, algorithm = nlopt.GD_STOGO, use_multiproc = True):
    """
    Estimates psychological parameters in a hierarchical Bayesian manner
    using a variational Bayesian inference.
    
    Parameters
    ----------
    model: object
        Learning model.
        
    ds: dataset (xarray)
        Dataset of time step level experimental data (cues, outcomes etc.)
        for each participant.
        
    fixed_pars: dict or None, optional
        Dictionary with names of parameters held fixed (keys) and fixed values.
        Defaults to None.
        
    X: data frame or None, optional
        Data frame of regressors for any psychological parameters in the list
        'reg_par_names', or None.  The index (row names) should be the same 
        participant IDs as in ds.  If None, then the mean and precision
        of each (logit transformed) psych parameter are estimated and no
        regression is performed.  Defaults to None.

    max_vb_iter: int, optional
        Maximum number of variational Bayes algorithm iterations.
        Defaults to 5.
        
    global_maxeval: int, optional
        Maximum number of function evaluations per individual for global optimization.
        Defaults to 200.
        
    local_maxeval: int, optional
        Maximum number of function evaluations per individual for local optimization.
        Defaults to 1000.
    
    local_tolerance: float, optional
        Specifies tolerance for relative change in parameter values (xtol_rel)
        as a condition for ending the local optimization.  Defaults to 0.05.
        
    algorithm: object, optional
            The algorithm used for global optimization.  Defaults to nlopt.GD_STOGO.
            
    use_multiproc: Boolean, optional
        Whether or not to use multiprocessing.  Defaults to True.

    Returns
    -------
    dict
    
    Notes
    -----
    X should NOT contain an intercept term: one is added automatically.
    
    The predictors in X are automatically standardized before being used in
    regression.
    
    'theta' indicates logit-transformed psychological parameters, while 'par'
    indicates these parameters in their original space.
    
    theta (logit-transformed psychological parameters) are assumed to have independent
    normal group level priors, which in turn are given normal-gamma hyperpriors.
    
    See the documentation for a full description of the statistical model and
    variational Bayes inference algorithm.
    """
    # count things, set up parameter space boundaries etc.
    idents = ds['ident'].values
    n = len(idents) # number of individuals
    all_par_names = list(model.pars.index)
    if fixed_pars is None:
        free_par_names = all_par_names.copy()        
    else:
        fixed_par_names = list(fixed_pars.keys())
        fixed_par_values = list(fixed_pars.values())
        free_par_names = list(set(all_par_names).difference(fixed_par_names))
    par_max = model.pars.loc[all_par_names, 'max'].values
    par_min = model.pars.loc[all_par_names, 'min'].values
    par_range = par_max - par_min
    free_par_names = sorted(free_par_names)
    n_p = len(free_par_names) # number of free parameters
    if not X is None:
        ds = ds.loc[{'ident': X.index}] # make sure that ds['ident'] is sorted the same way as X.index
        X = (X - X.mean())/X.std() # standardize X
        X.insert(0, 'intercept', 1.0) # insert a column for the intercept term
        x_names = X.columns.values # names of regressors
        n_x = X.shape[1] # number of regressors
    
    # logit function to transform parameters from their original space to -infty, infty (phi -> theta)
    def logit_transform(phi):
        theta = np.log(phi - par_min) - np.log(par_max - phi)
        return theta
    
    # logistic function to transform parameters from -infty, infty to their original space (theta -> phi)
    def logistic_transform(theta):
        phi = par_min + par_range/(1 + np.exp(-theta))
        return phi
    
    # keep track of relative change in theta
    rel_change = np.zeros(max_vb_iter)
    
    # define multivariate normal prior on beta (mean is zero) and gamma prior on rho
    prior_precision = (1/10)*np.identity(n_x) # prior precision matrix for beta (prior variance is 10)
    prior_a = 5
    prior_b = 5

    # compute posterior hyperparameters and other variables that do not change across iterations
    a = prior_a + n/2
    Xt = X.values.transpose()
    XtX = Xt@X.values
    
    # initialize E_rho and E_beta
    E_rho = np.ones(n_p)
    E_beta = np.zeros((n_p, n_x))

    # loop through variational Bayes updates
    for i in range(max_vb_iter):
        print('\n variational Bayes iteration ' + str(i + 1))
        
        # ** compute approximate posterior of theta | rho, beta (Laplace approximation) **
        global logit_normal_log_prior # needs this for multiprocessing to work
        def logit_normal_log_prior(phi): # define log-prior function (logit-normal)
            theta = logit_transform(phi)
            return np.sum(stats.norm.logpdf(theta, loc = ***, scale = 1/np.sqrt(E_rho)))
        map_fit = fit_indv(model = model,
                           ds = ds,
                           fixed_pars = fixed_pars,
                           phi0 = logistic_transform(theta_star),
                           log_prior = logit_normal_log_prior, # FIGURE THAT OUT
                           global_maxeval = global_maxeval,
                           local_maxeval = local_maxeval,
                           local_tolerance = local_tolerance,
                           algorithm = algorithm,
                           use_multiproc = use_multiproc)
        phi_star = result.loc[:, free_par_names] # MAP estimate of phi
        E_theta = logit_transform(phi_star) # MAP estimate of theta is aprx. posterior mean
        E_theta2 = pd.DataFrame(0.0, columns = free_par_names, index = idents) # aprx. posterior mean of theta^2
        for i in idents: # loop through subjects to compute E_theta2
            ds_i = ds.loc[{'ident': i}].squeeze() # data from subject i
            if 'valid_resp' in list(ds_i.keys()): # exclude time steps (trials) without valid responses
                ds_i = ds_i.loc[{'t': ds_i['valid_resp']}]
            H = nd.Hessian(lambda phi: logit_normal_prior(phi) + sr.log_lik(model, ds_i, phi_star[i].values)) # -Hessian is aprx. post. precision matrix ***** SHOULD BE DEFINED IN TERMS OF THETA *****
            V_theta = np.diag(np.linalg.inv(-H)) # aprx. posterior variance of theta
            E_theta2[i] = V_theta + E_theta[i]**2
        
        # ** compute approximate posterior of beta | theta, rho **
        Sigma = np.linalg.inv(prior_precision + E_rho*XtX)
        mu = (E_rho*Sigma@Xt@E_theta).squeeze()
        E_beta = mu
        E_betak_betaj = ***
        
        # ** compute approximate posterior of rho | beta, rho **
        b = ***
        
        # FINISH UPDATING        
        
        # relative change (to assess convergence)
        rel_change[i] = np.sum(abs(new_est_psych_par - est_psych_par))/np.sum(abs(est_psych_par))
        print('relative change: ' + '{:.8}'.format(rel_change[i]))
        # update est_psych_par
        est_psych_par = new_est_psych_par
        # exit loop if have achieved tolerance
        if rel_change[i] < 0.0001:
            break
    
    # output
    return result