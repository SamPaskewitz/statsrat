import numpy as np
import pandas as pd
import xarray as xr
import statsrat as sr
from scipy import stats
import nlopt
import numdifftools as nd

def fit(model, ds, fixed_pars = None, X = None, phi0 = None, max_vb_iter = 10, global_maxeval = 200, local_maxeval = 1000, local_tolerance = 0.05, algorithm = nlopt.GD_STOGO, use_multiproc = True):
    """
    Estimates psychological parameters in a hierarchical Bayesian manner
    using variational Bayesian inference.
    
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
        
    phi0: array-like of floats or None, optional
        Start points for each individual in the dataset.
        If None, then parameter search starts at the midpoint
        of each parameter's allowed interval.  Defaults to None.

    max_vb_iter: int, optional
        Maximum number of variational Bayes algorithm iterations.
        Defaults to 10.
        
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
    **FINISH THIS**
    
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
    # DEVELOPMENT NOTES:
    # The solutions obtained and time required for optimizing in terms of phi and theta are comparable.
    # Using numerical differentiation to supply gradients for gradient based optimization doesn't work that well.
    # Instead of using fit_indv and then separately computing the Hessian, it might make more sense to do both in the same step.
    
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
        # FIX THIS
        #ds = ds.loc[{'ident': X.index.values}] # make sure that ds['ident'] is sorted the same way as X.index
        X = (X - X.mean())/X.std() # standardize X
        X.insert(0, 'intercept', 1.0) # insert a column for the intercept term
        x_names = X.columns.values # names of regressors
        n_x = X.shape[1] # number of regressors
    else:
        X = pd.DataFrame({'intercept': 1.0}, index = idents)
        x_names = ['intercept']
        n_x = 1
    
    # define multivariate normal prior on beta (mean is zero) and gamma prior on rho
    prior_precision = (1/10)*np.identity(n_x) # prior precision matrix for beta (prior variance is 10)
    prior_a = 5
    prior_b = 5

    # compute posterior hyperparameters and other variables that do not change across iterations
    a = prior_a + n/2
    Xt = X.values.transpose()
    XtX = Xt@X.values
    
    # initialize E_rho and E_beta
    E_rho = 10*np.ones(n_p)
    E_beta = np.zeros((n_x, n_p))

    # set up arrays for other variables
    Sigma = np.zeros((n_x, n_x, n_p))
    mu = np.zeros((n_x, n_p))
    a = 0.5*n*np.ones(n_p) # this does not change across iterations
    b = np.zeros(n_p)
    E_beta_beta = np.zeros((n_x, n_x, n_p))
    E_theta = np.zeros((n, n_p))
    V_theta = np.zeros((n, n_p))
    E_theta2 = np.zeros((n, n_p))
    E_resid2 = np.zeros((n, n_p))
    
    # loop through variational Bayes updates
    for z in range(max_vb_iter):
        print('\n variational Bayes iteration ' + str(z + 1))
        
        # ** compute approximate posterior of theta | rho, beta (Laplace approximation) **    
        global logit_normal_log_prior # needs this for multiprocessing to work
        def logit_normal_log_prior(phi, X): # define log-prior function (logit-normal)
            theta = sr.par_logit_transform(model, phi, free_par_names)
            theta_hat = (X.values.reshape((1, n_x))@E_beta).squeeze()
            return np.sum(stats.norm.logpdf(theta, loc = theta_hat, scale = 1/np.sqrt(E_rho)))
        map_fit = sr.fit_indv(model = model,
                              ds = ds,
                              fixed_pars = fixed_pars,
                              X = X,
                              phi0 = phi0,
                              log_prior = logit_normal_log_prior,
                              global_maxeval = global_maxeval,
                              local_maxeval = local_maxeval,
                              local_tolerance = local_tolerance,
                              algorithm = algorithm,
                              use_multiproc = use_multiproc)
        phi_star = map_fit.loc[:, free_par_names] # MAP estimate of phi
        old_E_theta = E_theta.copy() # use this later to assess convergence (via size of relative change)
        E_theta = sr.par_logit_transform(model, phi_star, free_par_names).values # MAP estimate of theta is aprx. posterior mean
        for i in range(n): # loop through subjects to compute V_theta and E_theta2
            ds_i = ds.loc[{'ident': idents[i]}].squeeze() # data from subject i
            if 'valid_resp' in list(ds_i.keys()): # exclude time steps (trials) without valid responses
                ds_i = ds_i.loc[{'t': ds_i['valid_resp']}]
            theta_hat = (X.values[i, :].reshape((1, n_x))@E_beta).squeeze()
            def f(theta):
                if fixed_pars is None:
                    phi = sr.par_logistic_transform(model, theta)
                else:
                    phi = pd.Series(0.0, index = all_par_names)
                    phi[free_par_names] = sr.par_logistic_transform(model, theta, free_par_names)
                    phi[fixed_par_names] = fixed_par_values
                return np.sum(stats.norm.logpdf(theta, loc = theta_hat, scale = 1/np.sqrt(E_rho))) + sr.log_lik(model, ds_i, phi)
            H = nd.Hessian(f, step = 0.05)(E_theta[i, :].squeeze()) # -Hessian is aprx. post. precision matrix
            V_theta[i, :] = np.diag(np.linalg.inv(-H)) # aprx. posterior variance of theta
            E_theta2[i, :] = V_theta[i, :] + E_theta[i, :]**2
        
        # ** compute approximate posterior of beta | theta, rho **
        old_E_beta = E_beta.copy()
        for p in range(n_p):
            Sigma[:, :, p] = np.linalg.inv(prior_precision + E_rho[p]*XtX)
            mu[:, p] = (E_rho[p]*Sigma[:, :, p]@Xt@E_theta[:, p]).squeeze()
            E_beta[:, p] = mu[:, p]
            E_beta_beta[:, :, p] = Sigma[:, :, p] + np.outer(mu[:, p], mu[:, p])
        
        # ** compute approximate posterior of rho | theta, beta **
        old_E_rho = E_rho.copy()
        for p in range(n_p):
            for i in range(n):
                E_resid2[i, p] = E_theta2[i, p] - 2*E_theta[i, p]*X.values[i, :]@E_beta[:, p] + X.values[i, :]@E_beta_beta[:, :, p]@Xt[:, i]
            b[p] = prior_b + 0.5*E_resid2[:, p].sum()
            E_rho[p] = a[p]/b[p]
        
        # relative change in E_theta (to assess convergence)
        if z > 0:
            rel_change_theta = np.sum(abs(E_theta - old_E_theta))/np.sum(abs(old_E_theta))
            print('relative change in E_theta: ' + '{:.8}'.format(rel_change_theta))
            rel_change_beta = np.sum(abs(E_beta - old_E_beta))/np.sum(abs(old_E_beta))
            print('relative change in E_beta: ' + '{:.8}'.format(rel_change_beta))
            rel_change_rho = np.sum(abs(E_rho - old_E_rho))/np.sum(abs(old_E_rho))
            print('relative change in E_rho: ' + '{:.8}'.format(rel_change_rho))
            # exit loop if change in E_theta is small enough
            if rel_change_theta < 0.01:
                break
            
    # package results into an Xarray dataset
    result = xr.Dataset({'E_theta': (['ident', 'par_name'], E_theta),
                         'V_theta': (['ident', 'par_name'], V_theta),
                         'E_beta': (['x_name', 'par_name'], E_beta),
                         'Sigma': (['x_name', 'x_name1', 'par_name'], Sigma),
                         'E_rho': (['par_name'], E_rho),
                         'a': (['par_name'], a),
                         'b': (['par_name'], b)},
                        coords = {'par_name': free_par_names,
                                  'x_name': x_names,
                                  'x_name1': x_names,
                                  'ident': idents})
    
    # output
    return result