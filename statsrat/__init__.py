import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats
from plotnine import *
import nlopt

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
    sim_ds = model.simulate(ds, par_val = par_val) # run simulation
    b_hat = np.array(sim_ds['b_hat'])
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
        
def fit_indv(model, ds, x0 = None, tau = None, global_time = 15, local_time = 15, algorithm = nlopt.GD_STOGO):
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

    x0: data frame/array-like of floats or None, optional
        Start points for each individual in the dataset.
        If None, then parameter search starts at the midpoint
        of each parameter's allowed interval.  Defaults to None

    tau: array-like of floats or None, optional
        Natural parameters of the log-normal prior.
        Defaults to None (don't use log-normal prior).
        
    global_time: int, optional
        Maximum time (in seconds) per individual for global optimization.
        Defaults to 15.
        
    local_time: int, optional
        Maximum time (in seconds) per individual for local optimization.
        Defaults to 15.
        
    algorithm: object, optional
        The algorithm used for global optimization.  Defaults to nlopt.GD_STOGO.

    Returns
    -------
    df: dataframe
        This dataframe has the following columns:
        
        ident: Participant ID (dataframe index).
        
        prop_log_post: Quantity proportional to the maximum log-posterior (equal
                       to log-likelihood given a uniform prior).
                       
        One column for each free parameter estimated.
        
        model: Learning model name.
        
        global_time: Maximum time (in seconds) per individual for global optimization.
        
        local_time: Maximum time (in seconds) per individual for local optimization.
        
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
    If tau is None (default) then MLE is performed (i.e. you use a uniform prior).

    This currently assumes log-normal priors on all model parameters.  This may be an
    improper prior for some cases (e.g. a learning rate parameter that must be between
    0 and 1 might be better modeled using something like a beta prior).  I may add different
    types of prior in the future.

    For now, this assumes discrete choice data (i.e. resp_type = 'choice').
    
    The model is fitted by first using a global non-linear optimization algorithm (specified by the
    'algorithm parameter', with GN_ORIG_DIRECT as default), and then a local non-linear optimization
    algorithm (LN_SBPLX) for refining the answer.  Both algorithms are from the nlopt package.
    """
    # count things etc.
    idents = ds['ident'].values
    n = len(idents)
    par_names = list(model.pars.index)
    n_p = len(par_names)
    lower = model.pars['min']   
    # set up data frame
    col_index = par_names + ['prop_log_post']
    df = pd.DataFrame(0.0, index = pd.Series(idents, dtype = str), columns = col_index)
    # list of participants to drop because their data could not be fit (if any)
    idents_to_drop = []
    
    # maximize log-likelihood/posterior
    for i in range(n):
        try:
            pct = np.round(100*(i + 1)/n, 1)
            print('Fitting ' + str(i + 1) + ' of ' + str(n) + ' (' + str(pct) + '%)')
            ds_i = ds.loc[{'ident' : idents[i]}].squeeze()

            # define objective function
            if tau is None:
                # uniform prior
                def f(x, grad = None):
                    if grad.size > 0:
                        grad = None
                    par_val = x
                    return log_lik(model, ds_i, par_val)
            else:
                # log-normal prior
                def f(x, grad = None):
                    if grad.size > 0:
                        grad = None
                    par_val = x
                    ll = log_lik(model, ds_i, par_val)
                    # loop through parameters to compute prop_log_prior (the part of the log prior that depends on par_val)
                    prop_log_prior = 0
                    for j in range(n_p):
                        y = np.log(np.sign(par_val[j] - lower[j]))
                        prop_log_prior += tau[0]*y + tau[1]*y**2
                    prop_log_post = ll + prop_log_prior
                    return prop_log_post

            # global optimization (to find approximate optimum)
            if x0 is None:
                x0_i = (model.pars['max'] + model.pars['min'])/2 # midpoint of each parameter's allowed interval
            else:
                x0_i = np.array(x0.iloc[i])
            gopt = nlopt.opt(algorithm, n_p)
            gopt.set_max_objective(f)
            gopt.set_lower_bounds(np.array(model.pars['min'] + 0.001))
            gopt.set_upper_bounds(np.array(model.pars['max'] - 0.001))
            gopt.set_maxtime(global_time)
            gxopt = gopt.optimize(x0_i)
            if local_time > 0:
                # local optimization (to refine answer)
                lopt = nlopt.opt(nlopt.LN_SBPLX, n_p)
                lopt.set_max_objective(f)
                lopt.set_lower_bounds(np.array(model.pars['min'] + 0.001))
                lopt.set_upper_bounds(np.array(model.pars['max'] - 0.001))
                lopt.set_maxtime(local_time)
                lxopt = lopt.optimize(gxopt)
                df.loc[idents[i], par_names] = lxopt
                df.loc[idents[i], 'prop_log_post'] = lopt.last_optimum_value()
            else:
                df.loc[idents[i], par_names] = gxopt
                df.loc[idents[i], 'prop_log_post'] = gopt.last_optimum_value()
        except:
            print('There was a problem fitting the model to data from participant ' + idents[i] + ' (' + str(i + 1) + ' of ' + str(n) + ')')
            idents_to_drop += [idents[i]] # record that this participant's data could not be fit.
    
    # drop participants (rows) if data could not be fit (if any)
    if len(idents_to_drop) > 0:
        df = df.drop(idents_to_drop)
        
    # record information about the model, optimization algorithm and length of optimization time per person
    df['model'] = model.name
    df['global_time'] = global_time
    df['local_time'] = local_time
    df['algorithm'] = nlopt.algorithm_name(algorithm)
    
    # if performing maximum likelihood estimation, then add some columns
    if tau is None:
        df['log_lik'] = df['prop_log_post'] # log likelihood
        df['aic'] = 2*(n_p - df['log_lik']) # Akaike information criterion (AIC)
        # compute log-likelihood and AIC of the guessing model (all choices have equal probability) for comparison
        choices_per_time_step = ds_i['y_psb'].values.sum(1) 
        df['log_lik_guess'] = np.sum(np.log(1/choices_per_time_step))
        df['aic_guess'] = 2*(0 - df['log_lik_guess'])
    
    return df

def fit_em(model, ds, max_em_iter = 5, global_time = 15, local_time = 15, algorithm = nlopt.GD_STOGO):
    """
    Fit the model to time step data using the expectation-maximization (EM) algorithm.
    
    Parameters
    ----------
    model: object
        Learning model.
        
    ds: dataset (xarray)
        Dataset of time step level experimental data (cues, outcomes etc.)
        for each participant.

    max_em_iter: int, optional
        Maximum number of EM algorithm iterations.
        Defaults to 5.
        
    global_time: int, optional
        Maximum time (in seconds) per individual for global optimization.
        Defaults to 15.
        
    local_time: int, optional
        Maximum time (in seconds) per individual for local optimization.
        Defaults to 15.
        
    algorithm: object, optional
            The algorithm used for global optimization.  Defaults to nlopt.GD_STOGO.

    Returns
    -------
    dict
    
    Notes
    -----
    This assumes that all (psychological) model parameters (when shifted to (0, Inf)) have a log-normal distribution.
    
    Let theta be defined as any model parameter, and y be that the natural logarithm of that 
    parameter after being shifted to the interval (0, Inf):
    y = log(sign(theta - min theta)*(theta - min theta))
    
    Then we assume y ~ N(mu, 1/rho) where rho is a precision parameters.
    The corresponding natural parameters are tau0 = mu*rho and tau1 = -0.5*rho.
    
    We perform the EM algorithm to estimate y, treating tau0 and tau1 as our latent variables,
    where y' is the current estimate and x is the behavioral data:
    Q(y | y') = E[log p(y | x, tau0, tau1)]
    = log p(x | y) + E[tau0]*y + E[tau1]*y^2 + constant term with respect to y
    This is obtained by using Bayes' theorem along with the canonical exponential form of the
    log-normal prior.  Thus the E step consists of computing E[tau0] and E[tau1], where the
    expectation is taken according to the posterior distribution of tau0 and tau1 (i.e. of mu and rho)
    given x and y'.  Recognizing that this posterior is normal-gamma allows us to make the neccesary calculations
    (details not provided here).
    """
    
    # count things, set up parameter space boundaries etc.
    n = len(ds.ident)
    par_names = list(model.pars.index)
    n_p = len(par_names) # number of psychological parameters
    lower = list(model.pars['min'])
    size = list(model.pars['max'] - model.pars['min'])
    bounds = []
    for i in range(len(model.pars)):
        bounds += [(model.pars['min'][i] + 0.001, model.pars['max'][i] - 0.001)]

    # keep track of relative change in est_psych_par
    rel_change = np.zeros(max_em_iter)
    
    # initialize (using MLE, i.e. uniform priors)
    print('\n initial estimation with uniform priors')
    result = fit_indv(model, ds, None, None, global_time, local_time)
    est_psych_par = np.array(result.loc[:, par_names])
    
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
        for j in range(n_p):
            y = np.log(np.sign(est_psych_par[:, j] - lower[j]))
            y_bar = y.mean()
            # posterior hyperparameters for tau0 and tau1
            mu0_prime = (nu*mu0 + n*y_bar)/(nu + n)
            nu_prime = nu + n
            alpha_prime = alpha + n/2
            beta_prime = beta + 0.5*(y - y_bar).sum() + 0.5*(n*nu/(n + nu))*(y_bar - mu0)**2
            # expectations of natural hyperparameters (https://en.wikipedia.org/wiki/Normal-gamma_distribution)
            E_tau0 = mu0_prime*(alpha_prime/beta_prime) # see "Moments of the natural statistics" on the above page
            E_tau1 = -0.5*(alpha_prime/beta_prime)
        # M step (MAP estimates of psych_par given results of E step)
        x0 = result.drop(columns = 'prop_log_post')
        result = fit_indv(model, ds, x0, [E_tau0, E_tau1], global_time, local_time, algorithm)
        new_est_psych_par = np.array(result.loc[:, par_names])
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

def fit_algorithm_plots(model, ds, x0 = None, tau = None, n_time_intervals = 6, time_interval_size = 10, algorithm = nlopt.GD_STOGO, algorithm_list = None):
    """
    Used to figure compare global optimization algorithms and/or test how long to
    run global optimization (in fit_indv) by generating plots.
    This should be run on a subset of the data prior to the main model fit.
    
    Parameters
    ----------
    model: object
        Learning model.
        
    ds: dataset (xarray)
        Dataset of time step level experimental data (cues, outcomes etc.)
        for each participant.

    x0: data frame/array-like of floats or None, optional
        Start points for each individual in the dataset.
        If None, then parameter search starts at the midpoint
        of each parameter's allowed interval.  Defaults to None

    tau: array-like of floats or None, optional
        Natural parameters of the log-normal prior.
        Defaults to None (to not use log-normal prior).
        
    n_time_intervals: int, optional
        Number of time intervals to use for testing global optimization
        (the global_time parameter of fit_indv).  Defaults to 6.
        
    time_interval_size: int, optional
        Size of time intervals to test (in seconds).  Defaults to 10.
        
    algorithm: object or None, optional
        The algorithm used for global optimization.  Defaults to nlopt.GD_STOGO.
        Is ignored and can be None if algorithm_list (to compare multiple algorithms)
        is specified instead.
        
    algorithm_list: list or None, optional
        Used in place of the algorithm argument to specify a list of algorithms to compare.
        Can be None (the default) if algorithm is specified instad (to only test only algorithm).

    Returns
    -------
    dict containing:
    
    df: dataframe
        Parameter estimates and log-likelihood/log-posterior values per person per
        global optimization run time.
        
    plot: plotnine plot object
        Plot of log-likelihood/log-posterior by optimization time, which can be
        used to graphically assess convergence.
        
    Notes
    -----
    No local optimization is run.
    """
    if algorithm_list is None:
        alg_list = [algorithm]
    else:
        alg_list = algorithm_list
    n = len(ds.ident) # number of people
    df_list = []
    
    for alg in alg_list:
        for i in range(n_time_intervals):
            new_df = fit_indv(model, ds, x0, tau, (i + 1)*time_interval_size, 0, alg)
            new_df.index = new_df.index.rename('ident')
            new_df.reset_index(inplace = True, drop = False)
            df_list += [new_df]
    df = pd.concat(df_list)
    
    if n_time_intervals > 1:
        if len(alg_list) == 1:
            plot = ggplot(df, aes('global_time', 'prop_log_post', color = df.index.astype(str))) + geom_point() + geom_line() + labs(color = 'ident')
        else:
            plot = ggplot(df, aes('global_time', 'prop_log_post', color = 'algorithm')) + geom_point() + geom_line() + facet_grid('. ~ ident')
    else:
        plot = ggplot(df, aes(df.index, 'prop_log_post', color = 'algorithm')) + geom_point() + geom_line() + labs(x = 'ident')
    plot.draw()
    
    return {'df': df, 'plot': plot}        

def make_sim_data(model, experiment, schedule = None, a_true = 1, b_true = 1, n = 10):
    # UPDATE THIS TO USE LOG-NORMAL PRIORS.
    """
    Generate simulated data given an experiment and schedule (with random parameter vectors).
    
    Parameters
    ----------
    model : object
        Learning model.
        
    experiment : object
        The experiment to be used for the recovery test.
        
    schedule_name : str, optional
        Name of the experimental schedule to be used for the test.
        Defaults to the first schedule in the experiment definition.
        
    a_true : int or list, optional
        Hyperarameter of the beta distribution used to generate true
        parameters.  Can be either a scalar or a list equal in length
        to the the number of parameters.  Defaults to 1.
        
    b_true : int or list, optional
        Hyperarameter of the beta distribution used to generate true
        parameters.  Can be either a scalar or a list equal in length
        to the the number of parameters.  Defaults to 1.
        
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
    
    If a = b = 1 (default), parameters will be drawn from uniform distributions.
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
    par = pd.DataFrame(stats.beta.rvs(a = a_true,
                                      b = b_true,
                                      loc = loc,
                                      scale = scale,
                                      size = (n, n_p)),
                       index = idents,
                       columns = par_names)

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

def recovery_test(model, experiment, schedule = None, a_true = 1, b_true = 1, n = 10, method = "indv"):
    """
    Perform a parameter recovery test.
    
    Parameters
    ----------
    model : object
        Learning model.
        
    experiment : object
        The experiment to be used for the recovery test.
        
    schedule : str, optional
        Name of the experimental schedule to be used for the test.
        Defaults to the first schedule in the experiment definition.
        
    a_true : int or list, optional
        Hyperarameter of the beta distribution used to generate true
        parameters.  Can be either a scalar or a list equal in length
        to the the number of parameters.  Defaults to 1.
        
    b_true : int or list, optional
        Hyperarameter of the beta distribution used to generate true
        parameters.  Can be either a scalar or a list equal in length
        to the the number of parameters.  Defaults to 1.
        
    n : int, optional
        Number of individuals to simulate.  Defaults to 10.

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
    # count things, set up parameter space boundaries etc.
    par_names = list(model.pars.index)
    n_p = len(par_names)
    loc = model.pars['min']
    scale = model.pars['max'] - model.pars['min']
    bounds = []
    for i in range(len(model.pars)):
        bounds += [(model.pars['min'][i] + 0.001, model.pars['max'][i] - 0.001)]

    # generate simulated data
    sim_data = make_sim_data(model, experiment, schedule, a_true, b_true, n)

    # estimate parameters
    fit_dict = {'indv': lambda ds : fit_indv(model = model, ds = sim_data['ds']),
                'em': lambda ds : fit_em(model = model, ds = sim_data['ds'])}
    fit = fit_dict[method](sim_data['ds'])

    # combine true and estimated parameters into one dataframe
    par = pd.concat((sim_data['par'], fit[par_names]), axis = 1)
    par.columns = pd.MultiIndex.from_product([['true', 'est'], par_names])

    # compare parameter estimates to true values
    comp = pd.DataFrame(0, index = range(n_p), columns = ['par', 'mse', 'r', 'rsq', 'bias', 'bias_effect_size'])
    comp.loc[:, 'par'] = par_names
    for i in range(n_p):
        true = par.loc[:, ('true', par_names[i])]
        est = par.loc[:, ('est', par_names[i])]
        comp.loc[i, 'mse'] = np.mean((est - true)**2)
        comp.loc[i, 'r'] = est.corr(true)
        comp.loc[i, 'rsq'] = comp.loc[i, 'r']**2
        comp.loc[i, 'bias'] = np.mean(est - true)
        comp.loc[i, 'bias_effect_size'] = comp.loc[i, 'bias']/np.std(est - true)

    # assemble data for output
    output = {'par': par, 'fit': fit, 'comp': comp, 'sim_data': sim_data}
    return output

# UPDATE        
def one_step_pred(model, ds, n_pred = 10, method = "indv"):
    """
    One step ahead prediction test (similar to cross-validation).

    Parameters
    ----------
    ds : dataset (xarray)
        Dataset of time step level experimental data (cues, outcomes etc.)
        for each participant.
    n_pred : int
        The number of trials to be predicted (at the end of each data
        set).
    method : string
        The method used to fit the model, either "indv" or "em".  Defaults
        to 'indv'.

    Returns
    -------
    dict

    Notes
    -----
    This tests how well each of the last few choices is predicted by the model when fit to preceding trials.
    
    For now, this assumes discrete choice data (i.e. resp_type = 'choice')
    
    It is based on the 'prediction method' of Yechiam and Busemeyer (2005).

    We assume that each trial/response sequence has the same length.
    """

    # count things, set up parameter space boundaries etc.
    n = len(ds.ident) # number of individuals
    par_names = list(model.pars.index)
    n_p = len(par_names)
    n_t = len(ds.t) # number of time steps
    bounds = []
    for i in range(len(model.pars)):
        bounds += [(model.pars['min'][i] + 0.001, model.pars['max'][i] - 0.001)]
    pred_log_lik = np.zeros(n)
        
    # loop through time steps
    for t in range(n_t - n_pred, n_t):
        # trial and response data from time steps before t
        prev_ds = ds.loc[{'t' : np.array(ds.t <= t), 'trial' : np.array(ds.t <= t)}]        
        # fit model to data from time steps before t
        fit_dict = {'indv': lambda ds : fit_indv(ds = ds, model = model),
                'em': lambda ds : fit_em(ds = ds, model = model)}
        est_par = fit_dict[method](prev_ds).loc[:, 'est_par']
        # simulate model to predict response on time step t
        for i in range(n):
            ds_i = ds.loc[{'t' : np.array(ds.t <= t + 1), 'trial' : np.array(ds.t <= t + 1), 'ident' : ds.ident[i]}].squeeze()
            sim = model.simulate(ds_i, resp_type = 'choice', par_val = est_par.iloc[i, :])
            prob_t = np.array(sim['b_hat'].loc[{'t' : t}], dtype = 'float64')
            choice_matrix = np.array(ds_i['b'][{'t' : t}])
            pred_log_lik += np.sum( np.log(prob_t)*choice_matrix )
            
    return {'pred_log_lik': pred_log_lik, 'mean': pred_log_lik.mean(), 'std': pred_log_lik.std()}

# UPDATE        
def split_pred(model, trials_list, eresp_list, t_fit, method = "indv"):
    """
    Split prediction test (similar to cross-validation).

    Parameters
    ----------
    trials_list : list
        List of time step level experimental data (cues, outcomes
        etc.) for each participant.
    eresp_list : list
        List of empirical response arrays for each participant.
    t_fit : int
        The first 't_fit' trials are used to predict the remaining
        ones.
    method : string
        The method used to fit the model, either "indv" or "em".

    Returns
    -------
    dict

    Notes
    -----
    For now, this assumes discrete choice data (i.e. resp_type = 'choice').
    
    This is similar to the 'one_step_pred' method described above, but simply predict the last part of the data from the first.
    
    It is thus much faster to run and (at least for now) more practical.
    """

    # count things, set up parameter space boundaries etc.
    n = len(trials_list)
    par_names = list(model.pars.index)
    n_p = len(par_names)
    n_t = trials_list[0].shape[0] # number of time steps
    loc = model.pars['min']
    scale = model.pars['max'] - model.pars['min']
    bounds = []
    for i in range(len(model.pars)):
        bounds += [(model.pars['min'][i] + 0.001, model.pars['max'][i] - 0.001)]
    pred_log_lik = np.zeros(n)
        
    # trial and response data from time steps before t
    # THIS WILL BE MUCH MORE EFFICIENT I THINK ONCE I USE A BIG DATA FRAME FOR GROUP DATA INSTEAD OF LISTS
    prev_trials_list = []
    prev_eresp_list = []
    ftr_eresp_list = []
    for i in range(n):
        prev_trials_list += [trials_list[i].iloc[range(0, t_fit), :]]
        prev_eresp_list += [eresp_list[i][range(0, t_fit), :]]
        ftr_eresp_list += [eresp_list[i][range(t_fit, n_t), :]]
    # fit model to data from time steps before t
    fit_dict = {'indv': model.fit_indv,
                'em': model.fit_em}
    est_par = fit_dict[method](prev_trials_list, prev_eresp_list)['df'].loc[:, 'est_par']
    # simulate model to predict responses on remaining time steps
    for i in range(n):
        sim = model.simulate(trials_list[i], resp_type = 'choice', par_val = est_par.iloc[i, :])
        ftr_prob = np.array(sim.loc[range(t_fit, n_t), 'resp'], dtype = 'float64')
        pred_log_lik[i] = np.sum(np.log(ftr_prob)*ftr_eresp_list[i])
            
    return {'pred_log_lik': pred_log_lik, 'mean': pred_log_lik.mean(), 'std': pred_log_lik.std()}
