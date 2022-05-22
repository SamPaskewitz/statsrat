import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats
from plotnine import *
import nlopt
import multiprocessing
from time import perf_counter
from numpy.random import default_rng

# FINISH UPDATING EVERYTHING

def fit_em_multitask(model, ds, fixed_pars = None, x0 = None, max_em_iter = 5, global_maxeval = 200, local_maxeval = 1000, local_tolerance = 0.05, algorithm = nlopt.GD_STOGO, use_multiproc = True):
    """
    Fit the model to time step data using the expectation-maximization (EM) algorithm.
    
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
        
    x0: array-like of floats or None, optional
        Start points for each individual in the dataset.  If None (the default),
        then initial estimates are obtained using maximum likelihood estimation.
        If x0 is provided, then typically this is from previous maximum likelihood
        estimation using the fit_indv function.

    max_em_iter: int, optional
        Maximum number of EM algorithm iterations.
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
    This assumes that all (psychological) model parameters have what might be called a logit-normal
    distribution, as described below.
    
    Let theta be defined as any model parameter.  Then (theta - theta_min)/(theta_max - theta_min) is
    in the interval [0, 1].  We know re-parameterize in terms of a new variable called y which is in
    (-inf, inf), assuming that (theta - theta_min)/(theta_max - theta_min) is a logistic function of y:
    
    (theta - theta_min)/(theta_max - theta_min) = (1 + e^{-y})^{-1}
    
    It follows that:
    
    y = log(theta - theta_min) - log(theta_max - theta)
    
    This is the logit function, which is the inverse of the logistic function.  We have now transformed theta
    (which is confined to a finite interval) to y (which is on the whole real line).  Finally, we assume that
    y has a normal distribution:

    y ~ N(mu, 1/rho) 
    
    with mean mu and precision (inverse variance) rho.  The corresponding natural parameters are
    tau0 = mu*rho and tau1 = -0.5*rho.  This logit-normal distribution can take the same types of
    shapes as the beta distribution, including left skewed, right skewed, bell shaped, and valley shaped.     
    
    We perform the EM algorithm to estimate y (treating tau0 and tau1 as our latent variables)
    where y' is the current estimate and x is the behavioral data:
    
    Q(y | y') = E[log p(y | x, tau0, tau1)]
    = log p(x | y) + E[tau0]*y + E[tau1]*y^2 + constant term with respect to y
    
    This is obtained by using Bayes' theorem along with the canonical exponential form of the
    log-normal prior.  Thus the E step consists of computing E[tau0] and E[tau1], where the
    expectation is taken according to the posterior distribution of tau0 and tau1 (i.e. of mu and rho)
    given x and y'.  Recognizing that this posterior is normal-gamma allows us to make the neccesary
    calculations (details not provided here).
    """
    
    # count things, set up parameter space boundaries etc.
    idents = ds['ident'].values
    n = len(idents) # number of individuals
    all_par_names = list(model.pars.index)
    free_par_names = all_par_names.copy()
    if not fixed_pars is None:
        fixed_par_names = list(fixed_pars.keys())
        for fxpn in fixed_par_names:
            free_par_names.remove(fxpn)
    par_max = model.pars.loc[free_par_names, 'max'].values
    par_min = model.pars.loc[free_par_names, 'min'].values
    n_p = len(free_par_names) # number of free parameters

    # keep track of relative change in est_psych_par
    rel_change = np.zeros(max_em_iter)
    
    if x0 is None:
        # initialize (using MLE, i.e. uniform priors)
        print('\n initial estimation with uniform priors')
        result = fit_indv(model = model, ds = ds, fixed_pars = fixed_pars, tau = None, x0 = None, global_maxeval = global_maxeval, local_maxeval = local_maxeval, local_tolerance = local_tolerance, algorithm = algorithm)
        est_psych_par = result.loc[:, free_par_names].values
    else:
        est_psych_par = x0
    
    # See the following:
    # https://en.wikipedia.org/wiki/Conjugate_prior#When_likelihood_function_is_a_continuous_distribution
    mu0 = 0
    nu = 1 # 1 "virtual observation" in prior (i.e. weak prior)
    alpha = 5
    beta = 5

    # loop through EM algorithm
    for i in range(max_em_iter):
        print('\n EM iteration ' + str(i + 1))
        # E step (posterior means of hyperparameters given current estimates of psych_par)
        E_tau0 = np.zeros(n_p)
        E_tau1 = np.zeros(n_p)
        for j in range(n_p):
            y = np.log(est_psych_par[:, j] - par_min[j]) - np.log(par_max[j] - est_psych_par[:, j])
            y_bar = y.mean()
            # posterior hyperparameters for tau0 and tau1
            mu0_prime = (nu*mu0 + n*y_bar)/(nu + n)
            nu_prime = nu + n
            alpha_prime = alpha + n/2 
            beta_prime = beta + 0.5*np.sum((y - y_bar)**2) + 0.5*(n*nu/(n + nu))*(y_bar - mu0)**2
            # expectations of natural hyperparameters (https://en.wikipedia.org/wiki/Normal-gamma_distribution)
            E_tau0[j] = mu0_prime*(alpha_prime/beta_prime) # see "Moments of the natural statistics" on the above page
            E_tau1[j] = -0.5*(alpha_prime/beta_prime)
        # M step (MAP estimates of psych_par given results of E step)
        result = fit_indv(model = model, ds = ds, fixed_pars = fixed_pars, x0 = est_psych_par, tau = [E_tau0, E_tau1], global_maxeval = global_maxeval, local_maxeval = local_maxeval, local_tolerance = local_tolerance, algorithm = algorithm, use_multiproc = use_multiproc)
        new_est_psych_par = result.loc[:, free_par_names].values
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