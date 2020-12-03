import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats
import nlopt
from plotnine import ggplot, geom_point, geom_line, aes, stat_smooth, facet_wrap

def learn_plot(ds, var, sel = None, color_var = None, facet_var = None, drop_zeros = False, only_main = False):
    """
    Plots learning simulation data as a function of time.
    
    Parameters
    ----------
    ds : dataset (xarray)
        Learning simulation data (output of a model's 'simulate' method).
    var : string
        Variable to plot.
    sel : dict, optional
        Used to select a subset of 'var'.  Defaults to 'None' (i.e. all
        data in 'var' are plotted).
    color_var : string, optional
        Variable to be represented by color.
        Defaults to None (see notes).
    facet_var : string, optional
        Variable to control faceting.
        Defaults to None (see notes).
    drop_zeros : boolean, optional
        Drop rows where 'var' is zero.  Defaults to False.
    only_main : boolean, optional
        Only keep rows where 't_name' is 'main', i.e. time steps with
        punctate cues and/or non-zero outcomes.  Defaults to False.
        
    Returns
    -------
    plot : object
        A plotnine plot object.
    
    Notes
    -----
    The variable plotted should not have more than two dimensions besides time ('t').
    
    By default, the first non-time dimension will be used for color and the
    second one for faceting.
    
    The 'sel' argument is used to index 'ds' via the latter's 'loc' method.
    It should be a dictionary of the form {'dim0' : ['a', 'b'], 'dim1' : ['c']},
    where 'dim0', 'dim1' etc. are one or more dimensions of 'var'.
    
    """
    if sel is None:
        ds_var = ds[var]
    else:
        ds_var = ds[var].loc[sel].squeeze()    
    dims = list(ds_var.dims)
    dims.remove('t') # dimensions other than time ('t')
    n_dims = len(dims)
    df = ds_var.to_dataframe()
    df = df.reset_index()
    if only_main:
        df = df[df['t_name'] == 'main'] # only keep 'main' time steps (punctate cue and/or non-zero outcome)
    if drop_zeros:
        df = df[-(df[var] == 0)] # remove rows where var is zero 
    if n_dims == 0:
        plot = (ggplot(df, aes(x='t', y=var)) + geom_line())
    else:
        if color_var is None:
            color_var = dims[0]
        if n_dims == 1:
            plot = (ggplot(df, aes(x='t', y=var, color=color_var)) + geom_line())
        else:
            if facet_var is None:
                facet_var = dims[1]
            plot = (ggplot(df, aes(x='t', y=var, color=color_var)) + geom_line() + facet_wrap('~' + facet_var))
    return plot

def multi_sim(model, trials_list, resp_type, par_val, random_resp = False):
    """
    Simulate one or more trial sequences from the same schedule with known parameters.

    Parameters
    ----------
    trials_list : list
        List of time step level experimental data (cues, outcomes
        etc.) for each participant.  These should be generated from
        the same experimental schedule.

    resp_type : str
        Type of behavioral response: one of 'choice', 'exct' or 'supr'.

    par_val : list
        Learning model parameters (floats or ints).

    Returns
    -------
    ds : dataset
    """
    n_sim = len(trials_list)
    ds_list = []
    for i in range(n_sim):
        ds_new = model.simulate(trials = trials_list[i],
                                resp_type = resp_type,
                                par_val = par_val,
                                random_resp = random_resp,
                                ident = 'sim_' + str(i))
        ds_list += [ds_new]
    ds = xr.combine_nested(ds_list, concat_dim = ['ident'])
    return ds

def log_lik(model, ds, par_val):
    """
    Compute log-likelihood of individual time step data.

    Parameters
    ----------
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
    sim_ds = model.simulate(ds, resp_type = 'choice', par_val = par_val) # run simulation
    b_hat = np.array(sim_ds['b_hat'])
    b_hat[b_hat == 0] = 0.00000001
    log_prob = np.log(b_hat) # logarithms of choice probabilities
    resp = np.array(ds['b'])
    ll = np.sum(log_prob * resp) # log-likelihood of choice sequence
    return ll

def perform_oat(model, experiment, oat = None, n = 5, max_time = 60, algorithm = nlopt.GN_ORIG_DIRECT):
    """
    Perform an ordinal adequacy test (OAT).
    
    Parameters
    ----------
    model : learning model object
    
    experiment : experiment

    oat : str, optional

    n : int, optional
        Number of individuals to simulate.  Defaults to 5.
    
    max_time: int, optional
        Maximum time for each optimization (in seconds), i.e.
        about half the maximum total time running the whole OAT should take.
        Defaults to 60.
        
    algorithm: object
        NLopt algorithm to use for optimization.
        Defaults to nlopt.GN_ORIG_DIRECT.

    Returns
    -------
    output : dataframe (Pandas)
        Model parameters that produce maximum and minimum mean OAT score,
        along with those maximum and minimum mean OAT scores and (if n > 1)
        their associated 95% confidence intervals.

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
    s_list = oat_used.schedule_pos + oat_used.schedule_neg

    # for each schedule, create a list of trial sequences to use in simulations
    trials_list = dict(keys = s_list)
    for s in s_list:
        new = []
        for j in range(n):
            new += [experiment.make_trials(schedule = s)]
        trials_list[s] = new

    # get rid of resp_scale as a free parameter (it's fixed at 5)
    free_names = model.pars.index.tolist()
    if 'resp_scale' in free_names:
        free_names.remove('resp_scale') # modifies list in place
        # define objective function
        def f(x, grad = None):
                if grad.size > 0:
                    grad = None
                par_val = x
                sim_data = {}
                for s in s_list:
                    sim_data[s] = multi_sim(model, trials_list[s], experiment.resp_type, np.append(par_val, 5), random_resp = False)
                oat_total = oat_used.compute_total(data = sim_data)
                return oat_total
    else:
        # define objective function
        def f(x, grad = None):
                if grad.size > 0:
                    grad = None
                par_val = x
                sim_data = {}
                for s in s_list:
                    sim_data[s] = multi_sim(model, trials_list[s], experiment.resp_type, np.append(par_val, 5), random_resp = False)
                oat_total = oat_used.compute_total(data = sim_data)
                return oat_total
    n_free = len(free_names) # number of free parameters
    free_pars = model.pars.loc[free_names] # free parameters
    
    # maximize the OAT score
    opt_max = nlopt.opt(algorithm, n_free)
    opt_max.set_max_objective(f)
    opt_max.set_lower_bounds(np.array(free_pars['min'] + 0.001))
    opt_max.set_upper_bounds(np.array(free_pars['max'] - 0.001))
    opt_max.set_maxtime(max_time)
    par_max = opt_max.optimize(np.array(free_pars['default']))

    # minimize the OAT score
    opt_min = nlopt.opt(algorithm, n_free)
    opt_min.set_min_objective(f)
    opt_min.set_lower_bounds(np.array(free_pars['min'] + 0.001))
    opt_min.set_upper_bounds(np.array(free_pars['max'] - 0.001))
    opt_min.set_maxtime(max_time)
    par_min = opt_min.optimize(np.array(free_pars['default']))
    
    # simulate data to compute resulting OAT scores at max and min
    par_names = model.pars.index.tolist()
    if 'resp_scale' in par_names:
        min_data = dict(keys = s_list)
        max_data = dict(keys = s_list)
        for s in s_list:
            min_data[s] = multi_sim(model, trials_list[s], experiment.resp_type, np.append(par_min, 5), random_resp = False)
            max_data[s] = multi_sim(model, trials_list[s], experiment.resp_type, np.append(par_max, 5), random_resp = False)
    else:
        for s in s_list:
            min_data[s] = multi_sim(model, trials_list[s], experiment.resp_type, par_min, random_resp = False)
            max_data[s] = multi_sim(model, trials_list[s], experiment.resp_type, par_max, random_resp = False)
    # package results for output
    output_dict = dict()
    if n > 1:
        min_conf = oat_used.conf_interval(data = min_data, conf_level = 0.95)
        max_conf = oat_used.conf_interval(data = max_data, conf_level = 0.95)    
        for i in range(n_free):
            output_dict[free_names[i]] = [par_min[i], par_max[i]]
        output_dict['mean'] = [min_conf['mean'], max_conf['mean']]
        output_dict['lower'] = [min_conf['lower'], max_conf['lower']]
        output_dict['upper'] = [min_conf['upper'], max_conf['upper']]  
    else:
        min_value = oat_used.compute_total(data = min_data)
        max_value = oat_used.compute_total(data = max_data)
        for i in range(n_free):
            output_dict[free_names[i]] = [par_min[i], par_max[i]]
        output_dict['value'] = [min_value, max_value]
    output = pd.DataFrame(output_dict, index = ['min', 'max'])
        
    return output

def fit_indv(model, ds, a = 1, b = 1):
    """
    Fit the model to time step data by individual MLE/MAP.
    
    Parameters
    ----------
    model : object
        Learning model.
        
    ds : dataset (xarray)
        Dataset of time step level experimental data (cues, outcomes etc.)
        for each participant.

    a : int, optional

    b : int, optional

    Returns
    -------
    df : dataframe

    Notes
    -----
    MLE when a = b = 1 -> uniform prior.

    For now, this assumes discrete choice data (i.e. resp_type = 'choice').
    """
    # count things, set up parameter space boundaries etc.
    n = len(ds.ident)
    par_names = list(model.pars.index)
    n_p = len(par_names)
    loc = model.pars['min']
    scale = model.pars['max'] - model.pars['min']
    bounds = []
    for i in range(len(model.pars)):
        bounds += [(model.pars['min'][i] + 0.001, model.pars['max'][i] - 0.001)]

    # set up data frame
    lvl0_names = n_p*['est_par'] + ['prop_log_post']
    lvl1_names = par_names + ['']
    arrays = [lvl0_names, lvl1_names]
    tuples = list(zip(*arrays))
    col_index = pd.MultiIndex.from_tuples(tuples, names=['lvl0', 'lvl1'])
    df = pd.DataFrame(0.0, index = range(n), columns = col_index)
    df.loc[:, 'ident'] = ds.ident

    # maximize log-likelihood/posterior
    for i in range(n):
        ds_i = ds.loc[{'ident' : ds.ident[i]}].squeeze()
        
        def f(x, grad = None):
            if grad.size > 0:
                grad = None
            par_val = x
            ll = log_lik(model, ds_i, par_val)
            log_prior = np.sum(stats.beta.logpdf(par_val, a, b, loc, scale))
            prop_log_post = ll + log_prior
            return prop_log_post
    
        opt = nlopt.opt(nlopt.GN_ORIG_DIRECT, n_p)
        opt.set_max_objective(f)
        opt.set_lower_bounds(np.array(model.pars['min'] + 0.001))
        opt.set_upper_bounds(np.array(model.pars['max'] - 0.001))
        opt.set_maxtime(5)
        xopt = opt.optimize(np.array(model.pars['default']))
        df.loc[i, 'est_par'] = xopt
        df.loc[i, 'prop_log_post'] = opt.last_optimum_value()

    return df

def fit_em(model, ds, max_em_iter = 5):
    """
    Fit the model to time step data using the expectation-maximization (EM) algorithm.
    
    Parameters
    ----------
    model : object
        Learning model.
        
    ds : dataset (xarray)
        Dataset of time step level experimental data (cues, outcomes etc.)
        for each participant.

    max_em_iter : int, optional
        Maximum number of EM algorithm iterations.
        Defaults to 5.

    Returns
    -------
    dict
    """
    
    # count things, set up parameter space boundaries etc.
    n = len(ds.ident)
    par_names = list(model.pars.index)
    n_p = len(par_names) # number of psychological parameters
    loc = list(model.pars['min'])
    scale = list(model.pars['max'] - model.pars['min'])
    bounds = []
    for i in range(len(model.pars)):
        bounds += [(model.pars['min'][i] + 0.001, model.pars['max'][i] - 0.001)]

    # keep track of relative change in est_psych_par
    rel_change = np.zeros(max_em_iter)

    # initialize
    est_psych_par = np.zeros((n, n_p))
    for j in range(n_p):
        est_psych_par[:, j] = scale[j] * 0.5 + loc[j]
    est_a = np.zeros(n_p)
    est_b = np.zeros(n_p)

    # loop through EM algorithm
    for i in range(max_em_iter):
        print('EM iteration ' + str(i + 1))
        # E step (posterior means of hyperparameters given current estimates of psych_par)
        for j in range(n_p):
            theta_star = (est_psych_par[:, j] - loc[j]) / scale[j] # parameter re-scaled to the interval (0, 1)
            est_a[j] = 10 / (10 - np.sum(np.log(theta_star))) # prior on a is gamma(10, 10) -> posterior of a is gamma(10, 10 - np.sum(np.log(theta_star)))
            est_b[j] = 10 / (10 - np.sum(np.log(1 - theta_star))) # prior on a is gamma(10, 10) -> posterior of a is gamma(10, 10 - np.sum(np.log(1 - theta_star)))
        # M step (MAP estimates of psych_par given results of E step)
        new_est_psych_par = np.array(fit_indv(model = model, ds = ds, a = est_a, b = est_b).loc[:, 'est_par'])
        # relative change (to assess convergence)
        rel_change[i] = np.sum(abs(new_est_psych_par - est_psych_par)) / np.sum(abs(est_psych_par))
        print('relative change: ' + '{:.8}'.format(rel_change[i]))
        # update est_psych_par
        est_psych_par = new_est_psych_par
        # exit loop if have achieved tolerance
        if rel_change[i] < 0.0001:
            break

    # set up data frame
    lvl0_names = n_p*['est_par']
    lvl1_names = par_names
    arrays = [lvl0_names, lvl1_names]
    tuples = list(zip(*arrays))
    col_index = pd.MultiIndex.from_tuples(tuples, names=['lvl0', 'lvl1'])
    df = pd.DataFrame(est_psych_par, index = range(n), columns = col_index)
    df.loc[:, 'ident'] = ds.ident
    
    # output
    return df

def make_sim_data(model, experiment, schedule = None, a_true = 1, b_true = 1, n = 10):
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
    dict

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
    lvl0_names = n_p*['true_par']
    lvl1_names = par_names
    arrays = [lvl0_names, lvl1_names]
    tuples = list(zip(*arrays))
    col_index = pd.MultiIndex.from_tuples(tuples, names=['lvl0', 'lvl1'])
    par = pd.DataFrame(0.0, index = range(n), columns = col_index)
    par.loc[:, 'true_par'] = stats.beta.rvs(a = a_true,
                                            b = b_true,
                                            loc = loc,
                                            scale = scale,
                                            size = (n, n_p))

    # create a list of trial sequences to use in simulations
    trials_list = []
    for i in range(n):
        trials_list += [experiment.make_trials(schedule = schedule)]

    # generate simulated data
    ds_list = []
    for i in range(n):
        ds_list += [model.simulate(trials_list[i],
                                   resp_type = 'choice',
                                   par_val = par.loc[i, 'true_par'],
                                   ident = 'sim' + str(i))]
    ds = xr.combine_nested(ds_list, concat_dim = 'ident')
    
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
    dict

    Notes
    -----
    A parameter recovery test consists of three steps:
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
    fit_dict = {'indv': lambda ds : fit_indv(model = model, ds = ds),
                'em': lambda ds : fit_em(model = model, ds = ds)}
    fit_df = fit_dict[method](sim_data['ds'])

    # concatenate data frames
    df = pd.concat((sim_data['par'], fit_df), axis = 1)

    # compare parameter estimates to true values
    comp = pd.DataFrame(0, index = range(n_p), columns = ['par', 'rsq'])
    comp.loc[:, 'par'] = par_names
    for i in range(n_p):
        true = df.loc[:, 'true_par'].iloc[:, i]
        est = df.loc[:, 'est_par'].iloc[:, i]
        comp.loc[i, 'rsq'] = est.corr(true)**2

    # assemble data for output
    output = {'df': df, 'comp': comp}
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
