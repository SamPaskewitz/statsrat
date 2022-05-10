import numpy as np
import pandas as pd
import xarray as xr
import statsrat as sr
from scipy import stats
from random import choices

def Gelman_Rubin_stat(samples):
    '''
    Computes the Gelman-Rubin statistic (R_hat) for a single variable from MCMC chains.
    
    Parameters
    ----------
    samples: Numpy array (n_chains x n_samples)
        Array of samples from the different chains (row = chain, column = sample).
    '''
    n_s = samples.shape[1] # number of samples
    chain_means = samples.mean(axis = 1)
    chain_vars = samples.var(axis = 1)
    B = n_s*chain_means.var()
    W = chain_vars.mean()
    V_hat = (1/n_s)*((n_s - 1)*W + B)
    return np.sqrt(V_hat/W)

def fit_mcmc(model, ds, fixed_pars = None, X = None, n_samples = 2000, proposal_width_factor = 0.1):
    '''
    Estimates psychological parameters in a hierarchical Bayesian manner
    using a Markov chain Monte Carlo (MCMC) method.
    
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
        
    n_samples: int, optional
        Number of samples to take.  Defaults to 2000.
        
    X: data frame or None, optional
        Data frame of regressors for any psychological parameters in the list
        'reg_par_names', or None.  The index (row names) should be the same 
        participant IDs as in ds.  If None, then the mean and precision
        of each (logit transformed) psych parameter are estimated and no
        regression is performed.  Defaults to None.
        
    proposal_width_factor: float, optional
        Initial value for the proposal width factor, which controls how wide
        the Metropolis-Hastings random walk proposal distribution for theta
        is.  Defaults to 0.1
        
    Returns
    -------
    An xarray dataset of samples.
    
    Notes
    -----
    X should NOT contain an intercept term: one is added automatically.
    
    The predictors in X are automatically standardized before being used in
    regression.
    
    'theta' indicates logit-transformed psychological parameters, while 'par'
    indicates these parameter in their original space.
    
    theta (logit-transformed psychological parameters) are assumed to have independent
    normal group level priors, which in turn are given normal-gamma hyperpriors.
    
    Currently the function only produces a single MCMC chain.
    
    This runs a Gibbs sampler that uses a single Metropolis-Hastings step to
    sample theta.
    '''
    
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
    n_p = len(free_par_names) # number of free parameters
    if not X is None:
        ds = ds.loc[{'ident': X.index}] # make sure that ds['ident'] is sorted the same way as X.index
        X = (X - X.mean())/X.std() # standardize X
        X.insert(0, 'intercept', 1.0) # insert a column for the intercept term
        x_names = X.columns.values # names of regressors
        n_x = X.shape[1] # number of regressors
    
    # logit function to transform parameters from their original space to -infty, infty
    def logit_transform(par):
        theta = np.log(par - par_min) - np.log(par_max - par)
        return theta
    
    # logistic function to transform parameters from -infty, infty to their original space
    def logistic_transform(theta):
        par = par_min + par_range/(1 + np.exp(-theta))
        return par
    
    if X is None:
        # set up xarray dataset for storing MCMC samples
        samples = xr.Dataset(data_vars = {'theta': (['sample', 'ident', 'par_name'], np.zeros((n_samples + 1, n, n_p))), # logit-transformed psych pars
                                          'par': (['sample', 'ident', 'par_name'], np.zeros((n_samples + 1, n, n_p))), # psych pars
                                          'mu': (['sample', 'par_name'], np.zeros((n_samples + 1, n_p))), # mean of theta
                                          'rho': (['sample', 'par_name'], np.zeros((n_samples + 1, n_p))), # precision of theta
                                          'sigma': (['sample', 'par_name'], np.zeros((n_samples + 1, n_p))), # standard deviation of theta
                                          'accept_prob': (['sample', 'ident'], np.zeros((n_samples + 1, n)))}, # probability of accepting the proposed new theta vector
                             coords = {'sample': range(-1, n_samples),
                                       'ident': ds['ident'].values,
                                       'par_name': free_par_names})
        
        # define normal-gamma prior on mu and rho
        mu0 = 0
        nu = 1 # 1 "virtual observation" in prior (i.e. weak prior)
        a = 5
        b = 5
        
        # compute normal-gamma posterior hyperparameters that do not change across iterations
        nu_prime = np.array(n_p*[nu + n])
        a_prime = np.array(n_p*[a + n/2])
    else:
        # set up xarray dataset for storing MCMC samples
        samples = xr.Dataset(data_vars = {'theta': (['sample', 'ident', 'par_name'], np.zeros((n_samples + 1, n, n_p))),
                                          'theta_hat': (['sample', 'ident', 'par_name'], np.zeros((n_samples + 1, n, n_p))),
                                          'par': (['sample', 'ident', 'par_name'], np.zeros((n_samples + 1, n, n_p))),
                                          'beta': (['sample', 'par_name', 'x_name'], np.zeros((n_samples + 1, n_p, n_x))), # regression weights for theta
                                          'rho': (['sample', 'par_name'], np.zeros((n_samples + 1, n_p))),
                                          'sigma': (['sample', 'par_name'], np.zeros((n_samples + 1, n_p))),
                                          'accept_prob': (['sample', 'ident'], np.zeros((n_samples + 1, n)))},
                             coords = {'sample': range(-1, n_samples),
                                       'ident': ds['ident'].values,
                                       'par_name': free_par_names,
                                       'x_name': x_names})
        
        # define multivariate normal prior on beta (mean is zero) and gamma prior on rho
        Lambda = (1/10)*np.identity(n_x) # prior precision matrix for beta (prior variance is 10)
        a = 5
        b = 5
        
        # compute posterior hyperparameters and other variables that do not change across iterations
        a_prime = a + n/2
        Xt = X.values.transpose()
        XtX = Xt@X.values
    
    # initialize the chain (sample "-1" is not included in the actual output)
    samples['theta'].loc[{'sample': -1}] = stats.norm.rvs(size = n*n_p, scale = 4).reshape((n, n_p)) # NOTE: this previously had a SD of 1 instead of 4
    if not X is None:
        samples['rho'].loc[{'sample': - 1}] = stats.gamma.rvs(size = n_p, a = 1, scale = 1)
    
    # define other required variables
    old_log_lik = pd.Series(0.0, index = ds['ident'])
    old_log_prior = pd.Series(0.0, index = ds['ident'])
    adjustment_factor = np.sqrt(1.1) # affects the standard deviation of the proposal distribution (adjusted to keep acceptance rate between 0.1 and 0.4)
    last_adjustment = 0 # last time step that proposal_width_factor was changed
    
    # advance through the chain
    for s in range(0, n_samples):
        if X is None:
            # SAMPLE MU (MEAN) AND RHO (PRECISION) OF THETA
            
            # compute normal-gamma posterior hyperparameters (given theta)
            mu0_prime = pd.Series(0.0, index = non_reg_par_names)
            b_prime = pd.Series(0.0, index = non_reg_par_names)
            for par_name in free_par_names:
                theta = samples['theta'].loc[{'sample': s - 1, 'par_name': par_name}]
                theta_bar = theta.mean()
                mu0_prime[par_name] = (nu*mu0 + n*theta_bar)/(nu + n)
                b_prime[par_name] = b + 0.5*np.sum((theta - theta_bar)**2) + 0.5*(n*nu/(n + nu))*(theta_bar - mu0)**2

            # sample mu and rho
            rho = stats.gamma.rvs(a = a_prime, scale = 1/b_prime)
            mu = stats.norm.rvs(loc = mu0_prime, scale = np.sqrt(1/(nu_prime*rho)))
            samples['rho'].loc[{'sample': s}] = rho
            samples['sigma'].loc[{'sample': s}] = np.sqrt(1/rho) # the standard deviation corresponding to precision rho
            samples['mu'].loc[{'sample': s}] = mu
            expected_theta = mu
        
        else:
            # SAMPLE BETA (REGRESSION WEIGHTS) AND RHO (PRECISION OF RESIDUALS) OF LINEAR MODEL FOR THETA
            for par_name in free_par_names:
                # compute multivariate normal posterior hyperparameters for beta (given rho and theta)
                rho = samples['rho'].loc[{'sample': s - 1, 'par_name': par_name}].values
                theta = samples['theta'].loc[{'sample': s - 1, 'par_name': par_name}].values.reshape((n, 1))
                Lambda_prime = Lambda + rho*XtX
                Sigma_prime = np.linalg.inv(Lambda_prime)
                mu0_prime = (rho*Sigma_prime@Xt@theta).squeeze()
                
                # sample beta
                beta = stats.multivariate_normal.rvs(mean = mu0_prime, cov = Sigma_prime)
                samples['beta'].loc[{'sample': s, 'par_name': par_name}] = beta
                
                # compute theta_hat (predicted theta values) and residuals
                theta_hat = (X@beta.reshape((n_x, 1))).squeeze()
                samples['theta_hat'].loc[{'sample': s, 'par_name': par_name}] = theta_hat
                resid = theta.squeeze() - theta_hat

                # compute gamma posterior hyperparameters for rho (given beta and theta) and sample it
                b_prime = b + 0.5*np.sum(resid**2) # recall that the sum, and hence mean, of residuals is always zero
                rho = stats.gamma.rvs(a = a_prime, scale = 1/b_prime)
                samples['rho'].loc[{'sample': s, 'par_name': par_name}] = rho
                samples['sigma'].loc[{'sample': s, 'par_name': par_name}] = np.sqrt(1/rho)
                 
        # SAMPLE THETA (LOGIT-TRANSFORMED PSYCH PARS) FOR EACH INDIVIDUAL
        for i in samples['ident'].values:
            # perform a single Metropolis-Hastings step
            ds_i = ds.loc[{'ident' : i}].squeeze() # this may be an inefficient way to select person i's data
            proposed_theta = pd.Series(0.0, index = all_par_names)
            if not fixed_pars is None:
                proposed_theta[fixed_par_names] = fixed_par_values
            proposed_theta[free_par_names] = stats.norm.rvs(loc = samples['theta'].loc[{'sample': s - 1, 'ident': i}],
                                                            scale = proposal_width_factor*samples['sigma'].loc[{'sample': s}])
            u = stats.uniform.rvs()
            if X is None:
                proposed_log_prior = stats.norm.logpdf(proposed_theta[free_par_names],
                                                       loc = samples['mu'].loc[{'sample': s}],
                                                       scale = samples['sigma'].loc[{'sample': s}]).sum()
            else:
                proposed_log_prior = stats.norm.logpdf(proposed_theta[free_par_names],
                                                       loc = samples['theta_hat'].loc[{'sample': s, 'ident': i}],
                                                       scale = samples['sigma'].loc[{'sample': s}]).sum()
            proposed_log_lik = sr.log_lik(model, ds_i, logistic_transform(proposed_theta))
            log_accept_prob = np.min([0, proposed_log_prior + proposed_log_lik - old_log_prior[i] - old_log_lik[i]])
            samples['accept_prob'].loc[{'sample': s, 'ident': i}] = np.exp(log_accept_prob)
            if (np.log(u) <= log_accept_prob) or (s == 0):
                # accept the proposed new value of theta
                samples['theta'].loc[{'sample': s, 'ident': i}] = proposed_theta[free_par_names]
                samples['par'].loc[{'sample': s, 'ident': i}] = logistic_transform(proposed_theta)[free_par_names]
                old_log_prior[i] = proposed_log_prior
                old_log_lik[i] = proposed_log_lik
            else:
                # reject the proposed new value of theta (keep the old one)
                samples['theta'].loc[{'sample': s, 'ident': i}] = samples['theta'].loc[{'sample': s - 1, 'ident': i}]
    
        # adjust the proposal distribution to keep the acceptance rate within desired bounds
        if s > last_adjustment + 10: # wait 10 iterations since last adjustment before considering adjusting again
            mean_accept_prob = samples['accept_prob'].loc[{'sample': range(s - 10, s)}].mean() # mean over the last 10 sampling iterations
            if mean_accept_prob < 0.1: # acceptance rate too low
                proposal_width_factor /= adjustment_factor # tighten proposal distribution
                last_adjustment = s
            elif mean_accept_prob > 0.4: # acceptance rate too high
                proposal_width_factor *= adjustment_factor # widen proposal distribution
                last_adjustment = s
        
    return samples.loc[{'sample': range(0, n_samples)}]

def fit_multi_task_mcmc(models, ds_list, fixed_pars = None, n_samples = 2000, proposal_width_factor = 0.1):
    '''
    Estimates psychological parameters from multiple tasks in a hierarchical Bayesian manner
    using a Markov chain Monte Carlo (MCMC) method.
    
    Parameters
    ----------
    
    models: list
        List of learning model objects for each task.
        
    ds_list: dataset (xarray)
        List of datasets from each task of time step level experimental data
        (cues, outcomes etc.).
        
    fixed_pars: dict or None, optional
        List of dictionaries for each task with names of parameters held fixed
        (keys) and fixed values.  Defaults to None.
        
    n_samples: int, optional
        Number of samples to take.  Defaults to 2000.
        
    proposal_width_factor: float, optional
        Initial value for the proposal width factor, which controls how wide
        the Metropolis-Hastings random walk proposal distribution for theta
        is.  Defaults to 0.1
        
    Returns
    -------
    An xarray dataset of samples.
    
    Notes
    -----    
    'theta' indicates logit-transformed psychological parameters, while 'par'
    indicates these parameter in their original space.
    
    theta (logit-transformed psychological parameters) are assumed to have independent
    normal group level priors, which in turn are given normal-gamma hyperpriors.
    
    Currently the function only produces a single MCMC chain.
    
    This runs a Gibbs sampler that uses a single Metropolis-Hastings step to
    sample theta.
    '''
    # count things, set up parameter space boundaries etc.
    idents = ds_list[0]['ident'].values
    n = len(idents) # number of individuals
    n_tasks = len(models)
    all_par_names = []; task_par_names = []; fixed_par_names = []; fixed_par_values = []; par_max = []; par_min = []; par_range = []; logit_transform = []; logistic_transform = []
    for m in range(n_tasks):
        n_mp = len(list(models[m].pars.index.values))
        new_names = np.array(models[m].pars.index.values, dtype = 'str')
        task_par_names += [list(np.char.add(new_names, n_mp*['_' + str(m)]))] # par names for task m
        all_par_names += task_par_names[m]
        if not fixed_pars[m] is None:
            n_mfxp = len(list(fixed_pars[m].keys()))
            fixed_par_names += list(np.char.add(np.array(list(fixed_pars[m].keys()), dtype = 'str'), n_mfxp*['_' + str(m)]))
            fixed_par_values += list(fixed_pars[m].values())
        par_max += [models[m].pars.loc[new_names, 'max'].values]
        par_min += [models[m].pars.loc[new_names, 'min'].values]
        par_range += [par_max[m] - par_min[m]]
    
        # logit function to transform parameters from their original space to -infty, infty
        logit_transform += [lambda par: np.log(par - par_min[m]) - np.log(par_max[m] - par)]
        # logistic function to transform parameters from -infty, infty to their original space
        logistic_transform += [lambda theta: par_min[m] + par_range[m]/(1 + np.exp(-theta))]

    free_par_names = sorted(list(set(all_par_names).difference(fixed_par_names)), key = lambda name: name[-1]) # free parameters
    n_p = len(free_par_names) # number of free parameters
    # logistic function to transform parameters from -infty, infty to their original space
    all_logistic_transform = lambda theta: np.concatenate(par_min) + np.concatenate(par_range)/(1 + np.exp(-theta))
        
    # set up xarray dataset for storing MCMC samples
    samples = xr.Dataset(data_vars = {'theta': (['sample', 'ident', 'par_name'], np.zeros((n_samples + 1, n, n_p))), # logit-transformed psych pars
                                      'par': (['sample', 'ident', 'par_name'], np.zeros((n_samples + 1, n, n_p))), # psych pars
                                      'mu': (['sample', 'par_name'], np.zeros((n_samples + 1, n_p))), # mean of theta
                                      'Sigma': (['sample', 'par_name', 'par_name1'], np.zeros((n_samples + 1, n_p, n_p))), # covariance matrix of theta
                                      'accept_prob': (['sample', 'ident'], np.zeros((n_samples + 1, n)))}, # probability of accepting the proposed new theta vector
                         coords = {'sample': range(-1, n_samples),
                                   'ident': idents,
                                   'par_name': free_par_names,
                                   'par_name1': free_par_names})

    # define multivariate normal on mu (prior mean is 0, prior covariance matrix is 10*I_{n_p})
    Sigma0_inv = (1/10)*np.identity(n_p) # prior precision matrix for mu (prior variance is 10)
    # define inverse Wishart prior on Sigma (this should have a prior mean of I_{n_p})
    nu = n_p + 2
    Psi = (nu - n_p - 1)*np.identity(n_p)
    
    # initialize the chain (sample "-1" is not included in the actual output)
    samples['theta'].loc[{'sample': -1}] = stats.norm.rvs(size = n*n_p, scale = 4).reshape((n, n_p))
    samples['mu'].loc[{'sample': -1}] = stats.norm.rvs(size = n_p, scale = 4)
    
    # define other required variables
    old_log_lik = pd.Series(0.0, index = idents)
    old_log_prior = pd.Series(0.0, index = idents)
    adjustment_factor = np.sqrt(1.1) # affects the standard deviation of the proposal distribution (adjusted to keep acceptance rate between 0.1 and 0.4)
    last_adjustment = 0 # last time step that proposal_width_factor was changed
    
    # SOME INFO ON MULTIVARIATE NORMAL DISTRIBUTIONS
    # https://www.stat.pitt.edu/sungkyu/course/2221Spring15/lec1.pdf
    
    # advance through the chain
    for s in range(0, n_samples):
        # SAMPLE SIGMA (COVARIANCE MATRIX OF THETA)
        theta = samples['theta'].loc[{'sample': s - 1}].values
        mu = samples['mu'].loc[{'sample': s - 1}].values
        dev = np.array(theta - mu).reshape([n_p, n])
        samples['Sigma'].loc[{'sample': s}] = stats.invwishart.rvs(df = nu + n,
                                                                   scale = Psi + dev@dev.transpose())
        
        # SAMPLE MU (MEAN OF THETA)
        Sigma_inv = np.linalg.inv(samples['Sigma'].loc[{'sample': s}].values)
        Sigma0_prime = np.linalg.inv(Sigma0_inv + n*Sigma_inv)
        theta_bar = theta.mean(axis = 0).reshape((n_p, 1))
        samples['mu'].loc[{'sample': s}] = stats.multivariate_normal.rvs(mean = (Sigma0_prime@(n*Sigma_inv)@theta_bar).squeeze(),
                                                                         cov = Sigma0_prime)
        
        # SAMPLE THETA (LOGIT-TRANSFORMED PSYCH PARS) FOR EACH INDIVIDUAL
        for i in samples['ident'].values:
            # perform a single Metropolis-Hastings step
            proposed_theta = pd.Series(0.0, index = all_par_names)
            if not fixed_pars is None:
                proposed_theta[fixed_par_names] = fixed_par_values
            proposal_mean = samples['theta'].loc[{'sample': s - 1, 'ident': i}].values
            proposal_cov = np.diag(proposal_width_factor*np.diag(samples['Sigma'].loc[{'sample': s - 1}].values)) # diagonal proposal distribution
            proposed_theta[free_par_names] = stats.multivariate_normal.rvs(mean = proposal_mean,
                                                                           cov = proposal_cov)
            u = stats.uniform.rvs()
            proposed_log_prior = stats.multivariate_normal.logpdf(proposed_theta[free_par_names],
                                                                  mean = samples['mu'].loc[{'sample': s}],
                                                                  cov = samples['Sigma'].loc[{'sample': s}])
            proposed_log_lik = 0.0
            for m in range(n_tasks): # loop through tasks
                pars_im = list(logistic_transform[m](proposed_theta[task_par_names[m]])) # psych parameters for participant i, task m
                ds_im = ds_list[m].loc[{'ident' : i}].squeeze() # data from participant i, task m
                proposed_log_lik += sr.log_lik(models[m], ds_im, pars_im) # log-likelihood of participant i, task m
            log_accept_prob = np.min([0, proposed_log_prior + proposed_log_lik - old_log_prior[i] - old_log_lik[i]])
            samples['accept_prob'].loc[{'sample': s, 'ident': i}] = np.exp(log_accept_prob)
            if (np.log(u) <= log_accept_prob) or (s == 0):
                # accept the proposed new value of theta
                samples['theta'].loc[{'sample': s, 'ident': i}] = proposed_theta[free_par_names]
                samples['par'].loc[{'sample': s, 'ident': i}] = all_logistic_transform(proposed_theta)[free_par_names]
                old_log_prior[i] = proposed_log_prior
                old_log_lik[i] = proposed_log_lik
            else:
                # reject the proposed new value of theta (keep the old one)
                samples['theta'].loc[{'sample': s, 'ident': i}] = samples['theta'].loc[{'sample': s - 1, 'ident': i}]
                
        # adjust the proposal distribution to keep the acceptance rate within desired bounds
        if s > last_adjustment + 10: # wait 10 iterations since last adjustment before considering adjusting again
            mean_accept_prob = samples['accept_prob'].loc[{'sample': range(s - 10, s)}].mean() # mean over the last 10 sampling iterations
            if mean_accept_prob < 0.1: # acceptance rate too low
                proposal_width_factor /= adjustment_factor # tighten proposal distribution
                last_adjustment = s
            elif mean_accept_prob > 0.4: # acceptance rate too high
                proposal_width_factor *= adjustment_factor # widen proposal distribution
                last_adjustment = s
        
    return samples.loc[{'sample': range(0, n_samples)}]
