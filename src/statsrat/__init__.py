import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats
from plotnine import *
import nlopt
import multiprocessing
from time import perf_counter
from numpy.random import default_rng

def par_logit_transform(model, phi, par_names = None):
    '''
    Arguments
    ---------
    model: object
        Model to use.
        
    phi: array-like of floats
        Numbers to be transformed.

    par_names: array-like of strings or None, optional
        Names of parameters to transform, or None if all parameters should
        be included.  Defaults to None.

    Perform logit (inverse logistic) transformation on model parameters.

    Notes
    -----
    The functions par_logit_transform and par_logistic_transform are designed
    for use with hierarchical model fitting methods, which in this package use
    logit-normal priors.  That is, model parameters (denoted phi) are logit
    transformed into the space -infty, infty (the logit-transformed parameters
    are denoted theta) where they are given some variation on normal distribution
    prior.  Inference is done on the logit-transformed parameters (theta), and
    the logistic transformation can be used to transform them back to their
    original range (theta -> phi).
    '''
    if par_names is None:
        theta = np.log(phi - model.pars['min']) - np.log(model.pars['max'] - phi)
    else:
        theta = np.log(phi - model.pars.loc[par_names, 'min']) - np.log(model.pars.loc[par_names, 'max'] - phi)
    return theta

def par_logistic_transform(model, theta, par_names = None):
    '''
    Arguments
    ---------
    model: object
        Model to use.
        
    theta: array-like of floats
        Numbers to be transformed.

    par_names: array-like of strings or None, optional
        Names of parameters to transform, or None if all parameters should
        be included.  Defaults to None.

    Perform logistic transform on numbers in (-infty, infty) to bring
    them back to the model's specified parameter range.

    Notes
    -----
    The functions par_logit_transform and par_logistic_transform are designed
    for use with hierarchical model fitting methods, which in this package use
    logit-normal priors.  That is, model parameters (denoted phi) are logit
    transformed into the space -infty, infty (the logit-transformed parameters
    are denoted theta) where they are given some variation on normal distribution
    prior.  Inference is done on the logit-transformed parameters (theta), and
    the logistic transformation can be used to transform them back to their
    original range (theta -> phi).
    '''
    if par_names is None:
        phi = model.pars['min'] + (model.pars['max'] - model.pars['min'])/(1 + np.exp(-theta))
    else:
        phi = model.pars.loc[par_names, 'min'] + (model.pars.loc[par_names, 'max'] - model.pars.loc[par_names, 'min'])/(1 + np.exp(-theta))
    return phi

def multi_sim(model, trials_list, par_val, random_resp = False, sim_type = None):
    """
    Simulate one or more trial sequences from the same schedule with known parameters.

    Parameters
    ----------
    model : object
        Model to use.
    
    trials_list : list
        List of time step level experimental data (cues, outcomes
        etc.) for each participant.  These should be generated from
        the same experimental schedule.

    par_val : list
        Learning model parameters (floats or ints).
        
    random_resp : boolean
        Should responses be random?

    sim_type: str or None, optional
        Type of simulation to perform (passed to the model's .simulate() method).
        Should be a string indicating the type of simulation if there is more than
        one type (e.g. latent cause models), and otherwise should be None.
        Defaults to None.

    Returns
    -------
    ds : dataset
    """
    n_sim = len(trials_list)
    ds_list = []
    if sim_type is None:
        for i in range(n_sim):
            ds_new = model.simulate(trials = trials_list[i],
                                    par_val = par_val,
                                    random_resp = random_resp,
                                    ident = 'sim_' + str(i))
            ds_list += [ds_new]
    else:
        for i in range(n_sim):
            ds_new = model.simulate(trials = trials_list[i],
                                    par_val = par_val,
                                    random_resp = random_resp,
                                    ident = 'sim_' + str(i),
                                    sim_type = sim_type)
            ds_list += [ds_new]
    ds = xr.combine_nested(ds_list, concat_dim = ['ident'])
    return ds

def log_lik(model, ds, par_val):
    """
    Compute log-likelihood of individual time step data.

    Parameters
    ----------
    model : object
        A learning model object.
    
    ds : dataset
        Experimental data, including cues, behavioral responses,
        outcomes etc. from one individual and schedule.

    par_val : list
        Learning model parameters (floats or ints).

    Returns
    -------
    ll : float
        Log-likelihood of the data given parameter values.
    """
    # For now, this assumes discrete choice data (i.e. resp_type = 'choice')
    # 'b' has the same dimensions as 'b_hat' with 0 for choices not made and 1 for choices made
    b_hat = model.simulate(ds, par_val = par_val)['b_hat'].values # run simulation
    b_hat[b_hat == 0] = 0.00000001
    log_prob = np.log(b_hat) # logarithms of choice probabilities
    resp = np.array(ds['b'])
    ll = np.sum(log_prob*resp) # log-likelihood of choice sequence
    return ll

def perform_oat(model, experiment, minimize = True, oat = None, n = 5, max_time = 60, verbose = False, algorithm = nlopt.GN_ORIG_DIRECT, sim_type = None):
    """
    Perform an ordinal adequacy test (OAT).
    
    Parameters
    ----------
    model: learning model object
    
    experiment: experiment
    
    minimize: boolean, optional
        Should the OAT score by minimized as well as maximized?
        Defaults to True.

    oat: str or None, optional
        Name of the OAT to use.  Defaults to None, in which
        case the alphabetically first OAT in the experiment.

    n: int, optional
        Number of individuals to simulate.  Defaults to 5.
            
    max_time: int, optional
        Maximum time for each optimization (in seconds), i.e.
        about half the maximum total time running the whole OAT should take.
        Defaults to 60.
        
    verbose: boolean, optional
        Should the parameter values be printed as the search is going on?
        Defaults to False.
        
    algorithm: object, optional
        NLopt algorithm to use for optimization.
        Defaults to nlopt.GN_ORIG_DIRECT.
        
    sim_type: str or None, optional
        Type of simulation to perform (passed to the model's .simulate() method).
        Should be a string indicating the type of simulation if there is more than
        one type (e.g. latent cause models), and otherwise should be None.
        Defaults to None.

    Returns
    -------
    output: dataframe (Pandas)
        Model parameters that produce maximum and minimum mean OAT score,
        along with those maximum and minimum mean OAT scores and (if n > 1)
        their associated 95% confidence intervals.
        
    mean_resp_max: dataframe
        Relevant responses at OAT maximum (and minimum if applicable), averaged
        across individuals and trials.

    Notes
    -----
    The experiment's OAT object defines a behavioral score function
    designed such that positive values reflect response patterns
    consistent with empirical data and negative values reflect the
    opposite.  This method maximizes and minimizes the score produced
    by the learning model.  If the maximum score is positive, the model
    CAN behave reproduce empirical results.  If the minimum score is
    also positive, the model ALWAYS reproduces those results.
    """
    # determine which OAT to use
    if oat is None:
        oat_used = experiment.oats[list(experiment.oats.keys())[0]]
    else:
        oat_used = experiment.oats[oat]
    
    # make a list of all schedules (groups) to simulate
    if oat_used.schedule_neg is None:
        s_list = oat_used.schedule_pos
    else:
        s_list = oat_used.schedule_pos + oat_used.schedule_neg

    # for each schedule, create a list of trial sequences to use in simulations
    trials_list = dict(keys = s_list)
    for s in s_list:
        new = []
        for j in range(n):
            new += [experiment.make_trials(schedule = s)]
        trials_list[s] = new

    # set up parameter space
    par_names = model.pars.index.tolist()
    free_names = par_names.copy()
    if 'resp_scale' in free_names: # get rid of resp_scale as a free parameter (it's fixed at 5)
        free_names.remove('resp_scale') # modifies list in place
    n_free = len(free_names) # number of free parameters
    free_pars = model.pars.loc[free_names] # free parameters
    mid_pars = (free_pars['max'] + free_pars['min'])/2 # midpoint of each parameter's allowed interval
    
    # set up objective function
    if 'resp_scale' in par_names:
        if verbose:
            def f(x, grad = None):
                if grad.size > 0:
                    grad = None
                par_val = np.append(x, 5)
                print(par_val)
                sim_data = {}
                for s in s_list:
                    sim_data[s] = multi_sim(model, trials_list[s], par_val, random_resp = False, sim_type = sim_type)
                oat_total = oat_used.compute_total(data = sim_data)
                return oat_total
        else:
            def f(x, grad = None):
                if grad.size > 0:
                    grad = None
                par_val = np.append(x, 5)
                sim_data = {}
                for s in s_list:
                    sim_data[s] = multi_sim(model, trials_list[s], par_val, random_resp = False, sim_type = sim_type)
                oat_total = oat_used.compute_total(data = sim_data)
                return oat_total
    else:
        if verbose:
            def f(x, grad = None):
                if grad.size > 0:
                    grad = None
                par_val = x
                print(par_val)
                sim_data = {}
                for s in s_list:
                    sim_data[s] = multi_sim(model, trials_list[s], par_val, random_resp = False, sim_type = sim_type)
                oat_total = oat_used.compute_total(data = sim_data)
                return oat_total
        else:
            def f(x, grad = None):
                if grad.size > 0:
                    grad = None
                par_val = x
                sim_data = {}
                for s in s_list:
                    sim_data[s] = multi_sim(model, trials_list[s], par_val, random_resp = False, sim_type = sim_type)
                oat_total = oat_used.compute_total(data = sim_data)
                return oat_total    
    
    # maximize the OAT score
    print('Maximizing OAT score.')
    # global optimization (to find approximate optimum)
    gopt_max = nlopt.opt(algorithm, n_free)
    gopt_max.set_max_objective(f)
    gopt_max.set_lower_bounds(np.array(free_pars['min'] + 0.001))
    gopt_max.set_upper_bounds(np.array(free_pars['max'] - 0.001))
    gopt_max.set_maxtime(max_time/2)
    par_max_aprx = gopt_max.optimize(mid_pars)
    # local optimization (to refine answer)
    lopt_max = nlopt.opt(nlopt.LN_SBPLX, n_free)
    lopt_max.set_max_objective(f)
    lopt_max.set_lower_bounds(np.array(free_pars['min'] + 0.001))
    lopt_max.set_upper_bounds(np.array(free_pars['max'] - 0.001))
    lopt_max.set_maxtime(max_time/2)
    par_max = lopt_max.optimize(par_max_aprx)

    if minimize:
        # minimize the OAT score
        print('Minimizing OAT score.')
        # global optimization
        gopt_min = nlopt.opt(algorithm, n_free)
        gopt_min.set_min_objective(f)
        gopt_min.set_lower_bounds(np.array(free_pars['min'] + 0.001))
        gopt_min.set_upper_bounds(np.array(free_pars['max'] - 0.001))
        gopt_min.set_maxtime(max_time/2)
        par_min_aprx = gopt_min.optimize(mid_pars)
        # local optimization (to refine answer)
        lopt_min = nlopt.opt(nlopt.LN_SBPLX, n_free)
        lopt_min.set_min_objective(f)
        lopt_min.set_lower_bounds(np.array(free_pars['min'] + 0.001))
        lopt_min.set_upper_bounds(np.array(free_pars['max'] - 0.001))
        lopt_min.set_maxtime(max_time/2)
        par_min = lopt_min.optimize(par_min_aprx)

    # simulate data to compute resulting OAT scores at max and min
    par_names = model.pars.index.tolist()
    min_data = dict(keys = s_list)
    max_data = dict(keys = s_list)
    if 'resp_scale' in par_names:
        for s in s_list:
            max_data[s] = multi_sim(model, trials_list[s], np.append(par_max, 5), random_resp = False, sim_type = sim_type)
            if minimize:
                min_data[s] = multi_sim(model, trials_list[s], np.append(par_min, 5), random_resp = False, sim_type = sim_type)
    else:
        for s in s_list:
            max_data[s] = multi_sim(model, trials_list[s], par_max, random_resp = False, sim_type = sim_type)
            if minimize:
                min_data[s] = multi_sim(model, trials_list[s], par_min, random_resp = False, sim_type = sim_type)
    # package results for output
    output_dict = dict()
    if n > 1:
        if minimize:
            min_conf = oat_used.conf_interval(data = min_data, conf_level = 0.95)
            max_conf = oat_used.conf_interval(data = max_data, conf_level = 0.95)    
            for i in range(n_free):
                output_dict[free_names[i]] = [par_min[i], par_max[i]]
            output_dict['mean'] = [min_conf['mean'], max_conf['mean']]
            output_dict['lower'] = [min_conf['lower'], max_conf['lower']]
            output_dict['upper'] = [min_conf['upper'], max_conf['upper']]
            index = ['min', 'max']
        else:
            max_conf = oat_used.conf_interval(data = max_data, conf_level = 0.95)    
            for i in range(n_free):
                output_dict[free_names[i]] = [par_max[i]]
            output_dict['mean'] = [max_conf['mean']]
            output_dict['lower'] = [max_conf['lower']]
            output_dict['upper'] = [max_conf['upper']]
            index = ['max']
    else:
        if minimize:
            min_value = oat_used.compute_total(data = min_data)
            max_value = oat_used.compute_total(data = max_data)
            for i in range(n_free):
                output_dict[free_names[i]] = [par_min[i], par_max[i]]
            output_dict['value'] = [min_value, max_value]
            index = ['min', 'max']
        else:
            max_value = oat_used.compute_total(data = max_data)
            for i in range(n_free):
                output_dict[free_names[i]] = [par_max[i]]
            output_dict['value'] = [max_value]
            index = ['max']
    output = pd.DataFrame(output_dict, index)
    # compute relevant mean responses
    mean_resp_max = oat_used.mean_resp(data = max_data)
    if minimize:
        mean_resp_min = oat_used.mean_resp(data = min_data)
        mean_resp_max['parameters'] = 'max'
        mean_resp_min['parameters'] = 'min'
        mean_resp = pd.concat([mean_resp_min, mean_resp_max])
    else:
        mean_resp_min = None
        mean_resp = mean_resp_max
    return (output, mean_resp, max_data, min_data)    

def oat_grid(model, experiment, free_par, fixed_values, n_points = 10, oat = None, n = 20):
    """
    Compute ordinal adequacy test (OAT) scores while varying one model parameter
    (at evenly spaced intervals across its entire domain) and keeping the other parameters fixed.
    Useful for examining model behavior via plots.
    
    Parameters
    ----------
    model: learning model object
    
    experiment: experiment

    free_par: str
        Name of parameter to vary.
        
    fixed_values: dict
        Dict of values to be given to fixed parameters (keys are
        parameter names).
        
    n_points: int, optional
        How many values of the free parameter should be
        used.  Defaults to 10.

    oat: str, optional

    n: int, optional
        Number of individuals to simulate.  Defaults to 20.
        
    Returns
    -------
    df: data frame
        Parameter combinations with their mean OAT scores.
    """
    # determine which OAT to use
    if oat is None:
        oat_used = experiment.oats[list(experiment.oats.keys())[0]]
    else:
        oat_used = experiment.oats[oat]
    
    # make a list of all schedules (groups) to simulate
    if oat_used.schedule_neg is None:
        s_list = oat_used.schedule_pos
    else:
        s_list = oat_used.schedule_pos + oat_used.schedule_neg

    # for each schedule, create a list of trial sequences to use in simulations
    trials_list = dict(keys = s_list)
    for s in s_list:
        new = []
        for j in range(n):
            new += [experiment.make_trials(schedule = s)]
        trials_list[s] = new
    
    # set up data frame of parameter combinations
    par_names = model.pars.index.tolist()
    df = pd.DataFrame(0, index = range(n_points), columns = par_names, dtype = 'float')
    fixed_par_names = par_names
    fixed_par_names.remove(free_par) # modifies list in place
    for p in fixed_par_names:
        df[p] = fixed_values[p]
    free_min = model.pars['min'].loc[free_par] + 0.001
    free_max = model.pars['max'].loc[free_par] - 0.001
    step_size = (free_max - free_min)/n_points
    df[free_par] = np.arange(free_min, free_max, step_size)
    
    # loop through parameter combinations
    oat_score = np.zeros(n_points)
    for i in range(n_points):
        # loop through schedules to simulate behavior
        sim_data = dict(keys = s_list)
        for s in s_list:
            sim_data[s] = multi_sim(model, trials_list[s], df.iloc[i], random_resp = False)
        oat_score[i] = oat_used.compute_total(data = sim_data)
        
    # package data together for output
    df['oat_score'] = oat_score
    return df
        
def _fit_person_i_(arg):
    '''
    This is necessary because locally defined functions (i.e. defined within another function)
    apparently cannot be used with multiprocessing.
    
    Notes
    -----
    Because the pool.map method for multiprocessing only allows one argument to be mapped
    to a function (this function in our case), all of the details about the optimization problem
    are contained in a single dictionary called "arg" which is produced by fit_indv for each person.
    '''
    try:
        df_row_i = dict.fromkeys(arg['all_par_names'] + ['prop_log_post'])
        df_row_i['ident'] = arg['ident']
        if arg['fixed_par_names'] is None:
            df_row_i['fixed_par_names'] = None
            df_row_i['fixed_par_values'] = None
        else:
            df_row_i['fixed_par_names'] = ', '.join(arg['fixed_par_names'])
            df_row_i['fixed_par_values'] = str(arg['fixed_par_values'])

        # function to return full set of parameter values
        if arg['fixed_par_names'] is None:
            def full_phi(phi):
                return phi
        else:
            # incorporate fixed parameters (this could probably be much more efficient)
            def full_phi(phi):
                pvs = pd.Series(0.0, index = arg['all_par_names'])
                pvs[arg['free_par_names']] = phi
                pvs[arg['fixed_par_names']] = arg['fixed_par_values']
                return list(pvs)

        # objective function (proportional to log posterior)
        if arg['log_prior'] is None: # no prior distribution -> perform MLE
            def f(phi, grad = None):
                if grad.size > 0:
                    grad = None
                return log_lik(arg['model'], arg['ds_i'], full_phi(phi))
        else:
            def f(phi, grad = None): # prior distribution specified -> perform MAP
                if grad.size > 0:
                    grad = None
                return log_lik(arg['model'], arg['ds_i'], full_phi(phi)) + arg['log_prior'](phi, arg['X'])            

        # global optimization (to find approximate optimum)
        if arg['phi0'] is None:
            phi0_i = (arg['par_max'] + arg['par_min'])/2 # midpoint of each parameter's allowed interval
        else:
            phi0_i = arg['phi0']
        gopt = nlopt.opt(arg['algorithm'], arg['n_p'])
        gopt.set_max_objective(f)
        gopt.set_lower_bounds(np.array(arg['par_min'] + 0.001))
        gopt.set_upper_bounds(np.array(arg['par_max'] - 0.001))
        gopt.set_maxeval(arg['global_maxeval'])
        tic = perf_counter()
        gxopt = gopt.optimize(phi0_i)
        toc = perf_counter()
        df_row_i['global_time_used'] = toc - tic
        if arg['local_maxeval'] > 0:
            # local optimization (to refine answer)
            lopt = nlopt.opt(nlopt.LN_SBPLX, arg['n_p'])
            lopt.set_max_objective(f)
            lopt.set_lower_bounds(np.array(arg['par_min'] + 0.001))
            lopt.set_upper_bounds(np.array(arg['par_max'] - 0.001))
            lopt.set_maxeval(arg['local_maxeval'])
            lopt.set_xtol_rel(arg['local_tolerance'])
            tic = perf_counter()
            lxopt = lopt.optimize(gxopt)
            toc = perf_counter()
            df_row_i['local_time_used'] = toc - tic
            for p in range(arg['n_p']):
                df_row_i[arg['free_par_names'][p]] = lxopt[p]
            df_row_i['prop_log_post'] = lopt.last_optimum_value()
        else:
            df_row_i['local_time_used'] = 0
            for p in range(arg['n_p']):
                df_row_i[arg['free_par_names'][p]] = gxopt[p]
            df_row_i['prop_log_post'] = gopt.last_optimum_value()
        return df_row_i
    except:
        pass

def fit_indv(model, ds, fixed_pars = None, X = None, phi0 = None, log_prior = None, global_maxeval = 200, local_maxeval = 1000, local_tolerance = 0.05, algorithm = nlopt.GD_STOGO, use_multiproc = True):
    """
    Fit the model to time step data by individual maximum likelihood
    estimation (ML) or maximum a posteriori (MAP) estimation.
    
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
        Optional data frame of regressors that can allow the log_prior function
        to give different output for each individual.  The index (row names) should
        be the same participant IDs as in ds.  Defaults to None.

    phi0: array-like of floats or None, optional
        Start points for each individual in the dataset.
        If None, then parameter search starts at the midpoint
        of each parameter's allowed interval.  Defaults to None
    
    log_prior: function or None, optional
        Returns the logarithm of the prior distribution over logit-transformed psych
        model parameters.  This is used for maximum a priori (MAP) estimation. Defaults
        to None, i.e. no prior is used and thus maximum likelihood estimation (MLE) is
        performed.
    
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
    df: dataframe
        This dataframe has the following columns:
        
        ident: Participant ID (dataframe index).
        
        fixed_par_names: Names of fixed parameters (if any).
        
        fixed_par_values: Values of fixed parameters (if any).
        
        global_time_used: Actual global optimization time (in seconds).
        
        local_time_used: Actual local optimization time (in seconds).
        
        prop_log_post: Quantity proportional to the maximum log-posterior (equal
                       to log-likelihood given a uniform prior).
                       
        One column for each free parameter estimated.
        
        model: Learning model name.
        
        global_maxeval: Maximum number of function evaluations per individual for
                        global optimization.
        
        local_maxeval: Maximum number of function evaluations per individual for
                       local optimization.
        
        local_tolerance: Relative tolerance (xtol_rel) for terminating local optimization.
        
        algorithm: Name of the global optimization algorithm.
        
        Columns added if performing MLE (uniform prior):
        
        log_lik: Maximum log-likelihood (equal in this case to prop_log_post).
        
        aic:  Akaike information criterion (AIC) = 2*(number of free parameters - log_lik)
        
        log_lik_guess: Log-likelihood of the guessing model (each response has equal probability),
                       which has no parameters.  This is for detecting participants who did not
                       really try to perform the task.
        
        aic_guess: AIC for the guessing model = 2*(0 - log_lik_guess)
        
    Notes
    -----
    The log_prior function - if specified - should have two arguments: 'phi' and 'X'.
    'phi' is the vector of psychological model parameter values, while 'X' is a vector
    of regressors.  In some cases 'X' will allow the log prior to vary between individuals,
    although the log_prior function does not need to actually use it.
    
    If tau is None (default) then MLE is performed (i.e. you use a uniform prior).

    This currently assumes logit-normal priors on all model parameters.  See the documentation for the
    fit_em function for more information.

    For now, this assumes discrete choice data (i.e. resp_type = 'choice').
    
    The model is fitted by first using a global non-linear optimization algorithm (specified by the
    'algorithm parameter', with GN_ORIG_DIRECT as default), and then a local non-linear optimization
    algorithm (LN_SBPLX) for refining the answer.  Both algorithms are from the nlopt package.
    
    Global optimization is run for a set number of function evaluations (global_maxeval).  Experience
    suggests that it is very hard to get global optimization of model fit to converge using an relative
    change criterion such as that used for local optimization.
    
    Local optimization is run for either a set number of function evaluations (local_maxeval) or until
    the relative change in parameter estimates falls below the tolerance criterion (local_tolerance).
    Increasing local_tolerance will tend to decrease the accuracy of the final answer, but speed up
    optimization.
    
    If local_maxeval = 0, then local optimization is not run.  This should not generally be how the function is
    used.
    """
    # count things etc.
    idents = ds['ident'].values
    n = len(idents)
    all_par_names = list(model.pars.index)
    free_par_names = all_par_names.copy()
    if not fixed_pars is None:
        fixed_par_names = list(fixed_pars.keys())
        fixed_par_values = list(fixed_pars.values())
        for fxpn in fixed_par_names:
            free_par_names.remove(fxpn)
    else:
        fixed_par_names = None
        fixed_par_values = None
    par_max = model.pars.loc[free_par_names, 'max'].values
    par_min = model.pars.loc[free_par_names, 'min'].values
    n_p = len(free_par_names)
    
    args = [] # this list of dicts is needed due to pecularities with how multiprocessing works
    for i in range(n):
        new_arg = {}
        new_arg['ident'] = idents[i]
        new_arg['par_min'] = par_min
        new_arg['par_max'] = par_max
        new_arg['algorithm'] = algorithm
        new_arg['n_p'] = n_p
        new_arg['all_par_names'] = all_par_names
        new_arg['fixed_par_names'] = fixed_par_names
        new_arg['fixed_par_values'] = fixed_par_values
        new_arg['free_par_names'] = free_par_names        
        new_arg['ds_i'] = ds.loc[{'ident' : idents[i]}].squeeze()
        if 'valid_resp' in list(new_arg['ds_i'].keys()): # exclude time steps (trials) without valid responses
            new_arg['ds_i'] = new_arg['ds_i'].loc[{'t': new_arg['ds_i']['valid_resp']}]
        new_arg['global_maxeval'] = global_maxeval
        new_arg['local_maxeval'] = local_maxeval
        new_arg['local_tolerance'] = local_tolerance
        new_arg['log_prior'] = log_prior
        new_arg['model'] = model
        if X is None:
            new_arg['X'] = None
        else:
            new_arg['X'] = X.loc[idents[i]]
        if phi0 is None:
            new_arg['phi0'] = None
        else:
            new_arg['phi0'] = phi0[i, :]
        args += [new_arg]
    if use_multiproc:
        # use the multiprocessing library to optimize in parallel
        pool = multiprocessing.Pool()
        fit_rows = pool.map(_fit_person_i_, args)
    else:
        fit_rows = []
        for i in range(n):
            fit_rows += [_fit_person_i_(args[i])]
    
    # stitch dataframe together
    valid_fit_rows = [e for e in fit_rows if not e is None]
    df = pd.DataFrame.from_records(valid_fit_rows, index = 'ident')
    
    # record information about the model, optimization algorithm and length of optimization time per person
    df['model'] = model.name
    df['global_maxeval'] = global_maxeval
    df['local_maxeval'] = local_maxeval
    df['local_tolerance'] = local_tolerance
    df['algorithm'] = nlopt.algorithm_name(algorithm)
    
    # if some parameters are fixed, add the fixed values to the dataframe
    if not fixed_pars is None:
        for fxpn in fixed_par_names:
            df[fxpn] = fixed_pars[fxpn]
    
    # log likelihood of choices
    if log_prior is None:
        df['log_lik'] = df['prop_log_post'] 
    else:
        df['log_lik'] = 0.0
        for i in range(n):
            df.loc[idents[i], 'log_lik'] = log_lik(model, args[i]['ds_i'], df.loc[idents[i], list(model.pars.index)])
    # AIC
    df['aic'] = 2*(n_p - df['log_lik']) # Akaike information criterion (AIC)    
    # compute log-likelihood and AIC of the guessing model (all choices have equal probability) for comparison
    for i in range(n):
        choices_per_time_step = args[i]['ds_i']['y_psb'].values.sum(1) 
        df.loc[idents[i], 'log_lik_guess'] = np.sum(np.log(1/choices_per_time_step))
        df.loc[idents[i], 'aic_guess'] = 2*(0 - df.loc[idents[i], 'log_lik_guess'])
    
    return df

def fit_em(model, ds, fixed_pars = None, phi0 = None, max_em_iter = 5, global_maxeval = 200, local_maxeval = 1000, local_tolerance = 0.05, algorithm = nlopt.GD_STOGO, use_multiproc = True):
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
        
    phi0: array-like of floats or None, optional
        Start points for each individual in the dataset.  If None (the default),
        then initial estimates are obtained using maximum likelihood estimation.
        If phi0 is provided, then typically this is from previous maximum likelihood
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
    
    Let phi be defined as any model parameter.  Then (phi - phi_min)/(phi_max - phi_min) is
    in the interval [0, 1].  We know re-parameterize in terms of a new variable called theta which is in
    (-inf, inf), assuming that (phi - phi_min)/(phi_max - phi_min) is a logistic function of theta:
    
    (phi - phi_min)/(phi_max - phi_min) = (1 + e^{-theta})^{-1}
    
    It follows that:
    
    theta = log(phi - phi_min) - log(phi_max - phi)
    
    This is the logit function, which is the inverse of the logistic function.  We have now transformed phi
    (which is confined to a finite interval) to theta (which is on the whole real line).  Finally, we assume that
    theta has a normal distribution:

    theta ~ N(mu, sigma^2) 
    
    with mean mu and precision (inverse variance) rho.  The corresponding natural parameters are
    tau0 = mu/sigma^2 and tau1 = -1/(2*sigma^2).  This logit-normal distribution can take the same types of
    shapes as the beta distribution, including left skewed, right skewed, bell shaped, and valley shaped.     
    
    We perform the EM algorithm to estimate theta (treating tau0 and tau1 as our latent variables)
    where theta' is the current estimate and D is the behavioral data:
    
    Q(theta | theta') = E[log p(theta | D, tau0, tau1)]
    = log p(D | theta) + E[tau0]*theta + E[tau1]*theta^2 + constant term with respect to theta
    
    This is obtained by using Bayes' theorem along with the canonical exponential form of the
    log-normal prior.  Thus the E step consists of computing E[tau0] and E[tau1], where the
    expectation is taken according to the posterior distribution of tau0 and tau1 (i.e. of mu and rho)
    given D and theta'.  Recognizing that this posterior is normal-gamma allows us to make the neccesary
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

    # keep track of relative change in phi (estimated psych model parameters)
    rel_change = np.zeros(max_em_iter)
    
    if phi0 is None:
        # initialize (using MLE, i.e. uniform priors)
        print('\n initial estimation with uniform priors')
        result = fit_indv(model = model, ds = ds, fixed_pars = fixed_pars, log_prior = None, phi0 = None, global_maxeval = global_maxeval, local_maxeval = local_maxeval, local_tolerance = local_tolerance, algorithm = algorithm)
        phi = result.loc[:, free_par_names].values # estimated psych model parameters
    else:
        phi = phi0
    
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
            theta = np.log(phi[:, j] - par_min[j]) - np.log(par_max[j] - phi[:, j])
            theta_bar = theta.mean()
            # posterior hyperparameters for tau0 and tau1
            mu0_prime = (nu*mu0 + n*theta_bar)/(nu + n)
            nu_prime = nu + n
            alpha_prime = alpha + n/2 
            beta_prime = beta + 0.5*np.sum((theta - theta_bar)**2) + 0.5*(n*nu/(n + nu))*(theta_bar - mu0)**2
            # expectations of natural hyperparameters (https://en.wikipedia.org/wiki/Normal-gamma_distribution)
            E_tau0[j] = mu0_prime*(alpha_prime/beta_prime) # see "Moments of the natural statistics" on the above page
            E_tau1[j] = -0.5*(alpha_prime/beta_prime)
        # transform E_tau0 and E_tau1 back into conventional hyperparameters (E_step_mu and E_step_sigma)
        E_step_sigma = np.sqrt(-1/(2*E_tau1))
        E_step_mu = E_tau0*(E_step_sigma**2)
        # define log-prior function (logit-normal) using the results of the E step
        global logit_normal_log_prior # needs this for multiprocessing to work
        def logit_normal_log_prior(phi, X):
            theta = par_logit_transform(model, phi, free_par_names)
            return np.sum(stats.norm.logpdf(theta, loc = E_step_mu, scale = E_step_sigma))
        # M step (MAP estimates of psych_par given results of E step)
        result = fit_indv(model = model, ds = ds, fixed_pars = fixed_pars, phi0 = phi,
                          log_prior = logit_normal_log_prior,
                          global_maxeval = global_maxeval, local_maxeval = local_maxeval, local_tolerance = local_tolerance, algorithm = algorithm, use_multiproc = use_multiproc)
        new_phi = result.loc[:, free_par_names].values
        # relative change (to assess convergence)
        rel_change[i] = np.sum(abs(new_phi - phi))/np.sum(abs(phi))
        print('relative change: ' + '{:.8}'.format(rel_change[i]))
        # update phi
        phi = new_phi
        # exit loop if have achieved tolerance
        if rel_change[i] < 0.0001:
            break
    
    # output
    return result

def compare_optimization_algorithms(model, ds, algorithm_list, fixed_pars = None, phi0 = None, tau = None, global_maxeval = 200):
    """
    Compare global optimization algorithms (in fit_indv).
    This can be run on a subset of the data prior to the main model fit.
    
    Parameters
    ----------
    model: object
        Learning model.
        
    ds: dataset (xarray)
        Dataset of time step level experimental data (cues, outcomes etc.)
        for each participant.
        
    algorithm_list: list
        List of global optimization algorithms to compare.
        
    fixed_pars: dict or None, optional
        Dictionary with names of parameters held fixed (keys) and fixed values.
        Defaults to None.

    phi0: data frame/array-like of floats or None, optional
        Start points for each individual in the dataset.
        If None, then parameter search starts at the midpoint
        of each parameter's allowed interval.  Defaults to None

    tau: array-like of floats or None, optional
        Natural parameters of the log-normal prior.
        Defaults to None (to not use log-normal prior).   
        
    global_maxeval: int, optional
        Maximum number of function evaluations per individual for global optimization.
        Defaults to 200.

    Notes
    -----
    No local optimization is run.
    """
    df_list = []
    
    for algorithm in algorithm_list:
        new_df = fit_indv(model = model, 
                          ds = ds,
                          fixed_pars = fixed_pars,
                          phi0 = phi0, 
                          tau = tau, 
                          global_maxeval = global_maxeval, 
                          local_maxeval = 0, 
                          algorithm = algorithm)
        new_df.index = new_df.index.rename('ident')
        new_df.reset_index(inplace = True, drop = False)
        df_list += [new_df]
    df = pd.concat(df_list)
    
    plot = ggplot(df, aes(df.index, 'prop_log_post', color = 'algorithm')) + geom_point() + geom_line() + labs(x = 'ident')
    plot.draw()
    
    return {'df': df, 'table': df.groupby('algorithm')['prop_log_post'].mean()}

def make_sim_data(model, experiment, schedule = None, pars_to_sample = None, n = 10):
    """
    Generate simulated data given an experiment and schedule (with random parameter vectors).
    
    Parameters
    ----------
    model : object
        Learning model.
        
    experiment : object
        The experiment to be used.
        
    schedule : str, optional
        Name of the experimental schedule to be used for the test.
        Defaults to the first schedule in the experiment definition.
        
    pars_to_sample : dataframe or None, optional
        If None, then model parameters are drawn from uniform distributions.
        Otherwise, parameters are sampled (with replacement) from the rows of
        the dataframe; each parameter is sampled independently.
        
    n : int, optional
        Number of individuals to simulate.  Defaults to 10.

    Returns
    -------
    Dictionary with the following items:
        par: Parameter values for each simulated individual.
        
        ds: Simulated data.

    Notes
    -----
    For now, this assumes discrete choice data (i.e. resp_type = 'choice').
    """
    # count things, set up parameter space boundaries etc.
    par_names = list(model.pars.index)
    n_p = len(par_names)
    loc = model.pars['min']
    scale = model.pars['max'] - model.pars['min']

    # sample simulated 'subjects', i.e. parameter vectors
    idents = []
    for i in range(n):
        idents += ['sub' + str(i)]
    if pars_to_sample is None:
        par = pd.DataFrame(stats.uniform.rvs(loc = loc,
                                             scale = scale,
                                             size = (n, n_p)),
                           index = idents,
                           columns = par_names)
    else:
        par = pd.DataFrame(np.zeros((n, n_p)),
                           index = idents,
                           columns = par_names)
        rng = default_rng()
        for par_name in par_names:
            par.loc[:, par_name] = rng.choice(pars_to_sample[par_name].values, size = n, replace = True)

    # create a list of trial sequences to use in simulations
    trials_list = []
    for i in range(n):
        trials_list += [experiment.make_trials(schedule = schedule)]

    # generate simulated data
    ds_list = []
    for i in range(n):
        ds_list += [model.simulate(trials_list[i],
                                   par_val = par.loc[idents[i]],
                                   random_resp = True,
                                   ident = idents[i])]
    ds = xr.combine_nested(ds_list, concat_dim = 'ident', combine_attrs = 'override') # copy attributes from the first individual
    ds.attrs.pop('sim_pars') # sim_pars differ between individuals, so that attribute should be dropped; all other attributes are the same
    ds = ds[['x', 'y', 'y_psb', 'y_lrn', 'b']] # only keep behavioral responses and experimental variables, i.e. drop simulated psychological variables
    # output
    output = {'par': par, 'ds': ds}
    return output

def recovery_test(model, experiment, schedule = None, pars_to_sample = None, n = 10, fixed_pars = None, global_maxeval = 200, local_maxeval = 1000, local_tolerance = 0.05, algorithm = nlopt.GD_STOGO, method = "indv"):
    """
    Perform a parameter recovery test.
    
    Parameters
    ----------
    model: object
        Learning model.
        
    experiment: object
        The experiment to be used for the recovery test.
        
    schedule: str, optional
        Name of the experimental schedule to be used for the test.
        Defaults to the first schedule in the experiment definition.
        
    pars_to_sample: dataframe or None, optional
        If None, then model parameters are drawn from uniform distributions.
        Otherwise, parameters are sampled (with replacement) from the rows of
        the dataframe; each parameter is sampled independently.
        
    n: int, optional
        Number of individuals to simulate.  Defaults to 10.
        
    fixed_pars: dict or None, optional
        Dictionary with names of parameters held fixed (keys) and fixed values.
        Defaults to None.
        
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
        
    method: str, optional
        Either 'indv' (individual fits) or 'em' (EM algorithm).  Defaults
        to 'indv'.

    Returns
    -------
    A dictionary with the following items:
    
    par: Dataframe of true and estimated parameters.
    
    fit: Model fit results (output of fitting function).
    
    comp: Dataframe summarizing recovery statistics for each parameter.
    Columns include:
        par: Name of the parameter.
        mse: Mean squared error, i.e. mean of (estimate - true)^2.
        r: Correlation between true and estimated parameter values.
        rsq: R^2 (squared correlation between true and estimated parameter values).
        bias: Mean of (estimate - true); measures estimation bias.
        bias_effect_size: Standardized version of the bias measure (divided by SD of differences).
        
    sim_data: Simulated trial by trial data used for the recovery test.

    Notes
    -----
    A parameter recovery test consists of the following steps:
    1) generate random parameter vectors (simulated individuals)
    2) simulate data for each parameter vector
    3) fit the model to the simulated data to estimate individual parameters
    4) compare the estimated parameters (from step 3) to the true ones (from step 1)
    This procedure allows one to test how well a given learning model's parameters
    can be identified from data.  Some models and experimental schedules will have
    better estimation properties than others.
    
    For now, this assumes discrete choice data (i.e. resp_type = 'choice').
    """
    # generate simulated data
    sim_data = make_sim_data(model, experiment, schedule, pars_to_sample, n)

    # estimate parameters
    fit_dict = {'indv': lambda ds : fit_indv(model = model, ds = sim_data['ds'], fixed_pars = fixed_pars, global_maxeval = global_maxeval, local_maxeval = local_maxeval, local_tolerance = local_tolerance, algorithm = algorithm),
                'em': lambda ds : fit_em(model = model, ds = sim_data['ds'], fixed_pars = fixed_pars, global_maxeval = global_maxeval, local_maxeval = local_maxeval, local_tolerance = local_tolerance, algorithm = algorithm)}
    fit = fit_dict[method](sim_data['ds'])

    # combine true and estimated parameters into one dataframe
    par_names = list(model.pars.index)
    n_p = len(par_names) # number of parameters
    par = pd.concat((sim_data['par'], fit[par_names]), axis = 1)
    par.columns = pd.MultiIndex.from_product([['true', 'est'], par_names])

    # compare parameter estimates to true values
    if fixed_pars is None:
        free_par_names = par_names
    else:
        fixed_par_names = list(fixed_pars.keys())
        par_names = list(model.pars.index)
        free_par_names = list(set(par_names).difference(fixed_par_names))
    n_fp = len(free_par_names)
    comp = pd.DataFrame(0.0, index = range(n_fp), columns = ['par', 'mse', 'r', 'rsq', 'bias', 'bias_effect_size'])
    comp.loc[:, 'par'] = free_par_names
    for i in range(n_fp):
        true = par.loc[:, ('true', free_par_names[i])]
        est = par.loc[:, ('est', free_par_names[i])]
        comp.loc[i, 'mse'] = np.mean((est - true)**2)
        comp.loc[i, 'r'] = est.corr(true)
        comp.loc[i, 'rsq'] = comp.loc[i, 'r']**2
        comp.loc[i, 'bias'] = np.mean(est - true)
        comp.loc[i, 'bias_effect_size'] = comp.loc[i, 'bias']/np.std(est - true)

    # assemble data for output
    output = {'par': par, 'fit': fit, 'comp': comp, 'sim_data': sim_data}
    return output

def split_pred(model, ds, t_fit, fixed_pars = None, phi0 = None, tau = None, global_maxeval = 200, local_maxeval = 1000, local_tolerance = 0.05, algorithm = nlopt.GD_STOGO, method = 'indv'):
    """
    Split prediction test (similar to cross-validation): fit the model to
    earlier learning data in order to predict later learning data.
    
    Parameters
    ----------
    model: object
        Learning model.
        
    ds: dataset (xarray)
        Dataset of time step level experimental data (cues, outcomes etc.)
        for each participant.
        
    t_fit : int
        The first 't_fit' trials are used to predict the remaining
        ones.
        
    fixed_pars: dict or None, optional
        Dictionary with names of parameters held fixed (keys) and fixed values.
        Defaults to None.

    phi0: data frame/array-like of floats or None, optional
        Start points for each individual in the dataset.
        If None, then parameter search starts at the midpoint
        of each parameter's allowed interval.  Defaults to None

    tau: array-like of floats or None, optional
        Natural parameters of the log-normal prior.
        Defaults to None (don't use log-normal prior).
    
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
        
    method: str, optional
        Either 'indv' (individual fits) or 'em' (EM algorithm).  Deafults
        to 'indv'.
    
    Notes
    -----
    For now, this assumes discrete choice data (i.e. resp_type = 'choice').
    
    This is similar to the 'one_step_pred' method described above, but simply predict the last part of the data from the first.
    
    It is thus much faster to run and (at least for now) more practical. 
    """
    # split data into earlier and later parts
    ds_early = ds.loc[{'t': ds['t'] <= t_fit}]
    ds_late = ds.loc[{'t': ds['t'] > t_fit}]

    # estimate parameters using earlier data
    fit_dict = {'indv': lambda ds : fit_indv(model = model, ds = ds_early, fixed_pars = fixed_pars, global_maxeval = global_maxeval, local_maxeval = local_maxeval, local_tolerance = local_tolerance, algorithm = algorithm),
                'em': lambda ds : fit_em(model = model, ds = ds_early, fixed_pars = fixed_pars, global_maxeval = global_maxeval, local_maxeval = local_maxeval, local_tolerance = local_tolerance, algorithm = algorithm)}
    fit = fit_dict[method](ds_early)

    # compute log-likelihood of later responses
    idents = ds['ident'].values
    n = len(idents)
    pred_log_lik = np.zeros(n)
    for i in range(n):
        ds_i = ds_late.loc[{'ident' : idents[i]}].squeeze()
        par_val_i = fit.loc[idents[i], model.pars.index]
        pred_log_lik[i] = log_lik(model, ds_i, par_val_i)
            
    return {'pred_log_lik': pred_log_lik, 'fit': fit}
