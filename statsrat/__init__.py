import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats
import nlopt
from plotnine import ggplot, geom_point, geom_line, aes, stat_smooth, facet_wrap, scale_x_continuous, theme, element_text, position_dodge, position_identity

def learn_plot(ds, var, sel = None, rename_coords = None, color_var = None, facet_var = None, draw_points = False, drop_zeros = False, only_main = False, stage_labels = True, text_size = 10.0, dodge_width = 1.0):
    """
    Plots learning simulation data from a single schedule (condition, group) as a function of time.
    
    Parameters
    ----------
    ds : dataset (xarray)
        Learning simulation data (output of a model's 'simulate' method).
    var : string
        Variable to plot.
    sel : dict, optional
        Used to select a subset of 'var'.  Defaults to 'None' (i.e. all
        data in 'var' are plotted).
    rename_coords : dict or None, optional
        Either a dictionary for re-naming coordinates (keys are old names and
        values are new names), or None (don't re-name).  Defaults to None.
    color_var : string, optional
        Variable to be represented by color.
        Defaults to None (see notes).
    facet_var : string, optional
        Variable to control faceting.
        Defaults to None (see notes).
    draw_points : boolean, optional
        Whether or not points should be drawn as well as lines.
        Defaults to False.
    drop_zeros : boolean, optional
        Drop rows where 'var' is zero.  Defaults to False.
    only_main : boolean, optional
        Only keep rows where 't_name' is 'main', i.e. time steps with
        punctate cues and/or non-zero outcomes.  Defaults to False.
    stage_labels : boolean, optional
        Whether the x-axis should be labeled with 'stage_name' (if True) or
        't', i.e. time step (if False).  Defaults to True.
    text_size : float, optional
        Specifies text size.  Defaults to 10.0.
    dodge_width : float, optional
        Amount to separate overlapping lines so that they appear visually
        distinct (using Plotnine's position_dodge).  Defaults to 1.0.
        
    Returns
    -------
    plot : object
        A plotnine plot object.
    
    Notes
    -----
    The variable plotted should not have more than two dimensions besides time step ('t').
    
    By default, the first non-time dimension will be used for color and the
    second one for faceting.
    
    The 'sel' argument is used to index 'ds' via the latter's 'loc' method.
    It should be a dictionary of the form {'dim0' : ['a', 'b'], 'dim1' : ['c']},
    where 'dim0', 'dim1' etc. are one or more dimensions of 'var'.
    
    """
    ### SET UP DATA FRAME ###
    if sel is None:
        ds_var = ds[var].squeeze()
    else:
        ds_var = ds[var].loc[sel].squeeze()
    ds_var['t'] = range(ds_var['t'].values.shape[0])
    dims = list(ds_var.dims)
    dims.remove('t') # remove dimensions other than time step ('t')
    n_dims = len(dims)
    df = ds_var.to_dataframe()
    df = df.reset_index()
    if only_main:
        df = df[df['t_name'] == 'main'] # only keep 'main' time steps (punctate cue and/or non-zero outcome)
    if drop_zeros:
        df = df[-(df[var] == 0)] # remove rows where var is zero
    
    ### SET UP VARIABLE NAMES ###
    if not rename_coords is None:
        df = df.rename(columns = rename_coords)
        var_names = rename_coords
    else:
        var_names = dict.fromkeys(dims)
        for i in range(n_dims):
            var_names[dims[i]] = dims[i]
        
    ### CREATE PLOT ###
    dpos = position_dodge(width = dodge_width)
    if n_dims == 0:
        dpos = position_identity()
        plot = (ggplot(df, aes(x='t', y=var)) + geom_line())
    else:
        if color_var is None:
            color_var = var_names[dims[0]]
        if n_dims == 1:
            plot = (ggplot(df, aes(x='t', y=var, color=color_var)) + geom_line(position = dpos))
        else:
            if facet_var is None:
                facet_var = var_names[dims[1]]
                plot = (ggplot(df, aes(x='t', y=var, color=color_var)) + geom_line(position = dpos) + facet_wrap('~' + facet_var))
    
    if draw_points:
        plot += geom_point(position = dpos)
    
    if stage_labels:
        # add labels for stage names
        stage = df.stage.values
        stage_start = []
        stage_labels = []
        for s in np.unique(df.stage):
            start_point = df.t.loc[df.stage == s].min()
            stage_start += [start_point]
            stage_labels += [ds_var.stage_name.loc[{'t': start_point}].values]
        plot += scale_x_continuous(name = 'stage', breaks = stage_start, labels = stage_labels)
    
    plot += theme(text=element_text(size = text_size)) # set text size
    
    return plot

def multi_plot(ds_list, var, sel = None, rename_coords = None, schedule_facet = False, draw_points = False, drop_zeros = False, only_main = False, stage_labels = True, text_size = 10.0, dodge_width = 1.0):
    """
    Plots learning simulation data from multiple schedules (conditions, groups) as a function of time.
    
    Parameters
    ----------
    ds_list : list of datasets (xarray)
        Each element of the list consists of learning simulation data
        (output of a model's 'simulate' method) from a different schedule.
    var : string
        Variable to plot.
    sel : list of dicts or None, optional
        If a list, then elements correspond to elements of ds_list.
        Each list element is either None (to include everything in the 
        corresponding data set) or a dict used to select a subset of 'var'.
        Defaults to None (i.e. all data in 'var' are plotted for all data
        sets).
    rename_coords : dict or None, optional
        Either a dictionary for re-naming coordinates (keys are old names and
        values are new names), or None (don't re-name).  Defaults to None.
    schedule_facet : boolean, optional
        Whether or not schedules should be on different facets instead
        of on a single graph distinguished by color.  Defaults to False.
    draw_points : boolean, optional
        Whether or not points should be drawn as well as lines.
        Defaults to False.
    drop_zeros : boolean, optional
        Drop rows where 'var' is zero.  Defaults to False.
    only_main : boolean, optional
        Only keep rows where 't_name' is 'main', i.e. time steps with
        punctate cues and/or non-zero outcomes.  Defaults to False.
    stage_labels : boolean, optional
        Whether the x-axis should be labeled with 'stage_name' (if True) or
        't', i.e. time step (if False).  Defaults to True.
    text_size : float, optional
        Specifies text size.  Defaults to 10.0.
    dodge_width : float, optional
        Amount to separate overlapping lines so that they appear visually
        distinct (using Plotnine's position_dodge).  Defaults to 1.0.
        
    Returns
    -------
    plot : object
        A plotnine plot object.
    
    Notes
    -----
    It is assumed that either the schedules all have the same stage names, or else
    that the 'sel' argument is used to select time steps that all have the same stage names.
    
    The variable plotted should not have more than one dimension besides time step ('t').
    
    If there is an extra dimension besides 't', it will be used for facetting (if
    schedule_facet = False) or for color (if schedule_facet = True).
    
    The 'sel' argument is used to index 'ds' via the latter's 'loc' method.
    It should be a dictionary of the form {'dim0' : ['a', 'b'], 'dim1' : ['c']},
    where 'dim0', 'dim1' etc. are one or more dimensions of 'var'.
    """
    ### CREATE INDIVIDUAL DATA FRAMES ###
    df_list = []
    i = 0
    for ds in ds_list:
        if sel is None:
            new_ds_var = ds[var].squeeze()
        else:
            if sel[i] is None:
                new_ds_var = ds[var].squeeze()
            else:
                new_ds_var = ds[var].loc[sel[i]].squeeze()
        new_ds_var['t'] = range(new_ds_var['t'].values.shape[0])
        new_dims = list(new_ds_var.dims)
        new_dims.remove('t') # remove dimensions other than time step ('t')
        new_df = new_ds_var.to_dataframe()
        new_df = new_df.reset_index()
        new_df['schedule'] = new_ds_var.schedule.values
        df_list += [new_df]
        i += 1

    ### CONCATENATE DATA FRAMES ###
    df = pd.concat(df_list)
    if only_main:
        df = df[df['t_name'] == 'main'] # only keep 'main' time steps (punctate cue and/or non-zero outcome)
    if drop_zeros:
        df = df[-(df[var] == 0)] # remove rows where var is zero
    
    ### SET UP VARIABLE NAMES ###
    if not rename_coords is None:
        df = df.rename(columns = rename_coords)
        var_names = rename_coords
        if not 'schedule' in rename_coords.keys():
            var_names['schedule'] = 'schedule'
    else:
        n_dims = len(new_dims)
        var_names = dict.fromkeys(new_dims)
        for i in range(n_dims):
            var_names[new_dims[i]] = new_dims[i]
        var_names['schedule'] = 'schedule'

    ### CREATE PLOT ###
    dims = new_dims
    n_dims = len(dims)
    dpos = position_dodge(width = dodge_width)
    if n_dims == 0:
        if schedule_facet:
            dpos = position_identity()
            plot = (ggplot(df, aes(x='t', y=var)) + geom_line() + facet_wrap('~' + var_names['schedule']))
        else:
            plot = (ggplot(df, aes(x='t', y=var, color=var_names['schedule'])) + geom_line(position = dpos))
    else:
        if schedule_facet:
            plot = (ggplot(df, aes(x='t', y=var, color=var_names[dims[0]])) + geom_line(position = dpos) + facet_wrap('~' + var_names['schedule']))
        else:
            plot = (ggplot(df, aes(x='t', y=var, color=var_names['schedule'])) + geom_line(position = dpos) + facet_wrap('~' + var_names[dims[0]]))
    
    if draw_points:
        plot += geom_point(position = dpos)
    
    if stage_labels:
        # add labels for stage names
        stage = df.stage.values
        stage_start = []
        stage_labels = []
        for s in np.unique(df.stage):
            start_point = df.t.loc[df.stage == s].min()
            stage_start += [start_point]
            stage_labels += [df.stage_name.loc[df.t.values == start_point].values[0]]
        plot += scale_x_continuous(name = 'stage', breaks = stage_start, labels = stage_labels)
    
    plot += theme(text=element_text(size = text_size)) # set text size
                        
    return plot

def multi_sim(model, trials_list, par_val, random_resp = False):
    """
    Simulate one or more trial sequences from the same schedule with known parameters.

    Parameters
    ----------
    trials_list : list
        List of time step level experimental data (cues, outcomes
        etc.) for each participant.  These should be generated from
        the same experimental schedule.

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
    sim_ds = model.simulate(ds, par_val = par_val) # run simulation
    b_hat = np.array(sim_ds['b_hat'])
    b_hat[b_hat == 0] = 0.00000001
    log_prob = np.log(b_hat) # logarithms of choice probabilities
    resp = np.array(ds['b'])
    ll = np.sum(log_prob*resp) # log-likelihood of choice sequence
    return ll

def perform_oat(model, experiment, minimize = True, oat = None, n = 5, max_time = 60, verbose = False, algorithm = nlopt.GN_ORIG_DIRECT):
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
        # define objective function
        if verbose:
            def f(x, grad = None):
                    if grad.size > 0:
                        grad = None
                    par_val = np.append(x, 5)
                    print(par_val)
                    sim_data = {}
                    for s in s_list:
                        sim_data[s] = multi_sim(model, trials_list[s], par_val, random_resp = False)
                    oat_total = oat_used.compute_total(data = sim_data)
                    return oat_total
        else:
            def f(x, grad = None):
                    if grad.size > 0:
                        grad = None
                    par_val = np.append(x, 5)
                    sim_data = {}
                    for s in s_list:
                        sim_data[s] = multi_sim(model, trials_list[s], par_val, random_resp = False)
                    oat_total = oat_used.compute_total(data = sim_data)
                    return oat_total
    else:
        # define objective function
        if verbose:
            def f(x, grad = None):
                    if grad.size > 0:
                        grad = None
                    par_val = x
                    print(par_val)
                    sim_data = {}
                    for s in s_list:
                        sim_data[s] = multi_sim(model, trials_list[s], par_val, random_resp = False)
                    oat_total = oat_used.compute_total(data = sim_data)
                    return oat_total
        else:
            def f(x, grad = None):
                    if grad.size > 0:
                        grad = None
                    par_val = x
                    sim_data = {}
                    for s in s_list:
                        sim_data[s] = multi_sim(model, trials_list[s], par_val, random_resp = False)
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
    if 'resp_scale' in par_names:
        min_data = dict(keys = s_list)
        max_data = dict(keys = s_list)
        for s in s_list:
            max_data[s] = multi_sim(model, trials_list[s], np.append(par_max, 5), random_resp = False)
            if minimize:
                min_data[s] = multi_sim(model, trials_list[s], np.append(par_min, 5), random_resp = False)
    else:
        for s in s_list:
            max_data[s] = multi_sim(model, trials_list[s], par_max, random_resp = False)
            if minimize:
                min_data[s] = multi_sim(model, trials_list[s], par_min, random_resp = False)
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
    
    return (output, mean_resp)    

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
        
def fit_indv(model, ds, x0 = None, tau = None, max_time = 10):
    """
    Fit the model to time step data by individual MLE/MAP.
    
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
        
    max_time: int, optional
        Maximum time (in seconds) per individual.
        Defaults to 10.

    Returns
    -------
    df: dataframe

    Notes
    -----
    If tau is None (default) then MLE is performed (i.e. you use a uniform prior).

    This currently assumes log-normal priors on all model parameters.  This may be an
    improper prior for some cases (e.g. a learning rate parameter that must be between
    0 and 1 might be better modeled using something like a beta prior).  I may add different
    types of prior in the future.

    For now, this assumes discrete choice data (i.e. resp_type = 'choice').
    """
    # count things, set up parameter space boundaries etc.
    n = len(ds.ident)
    par_names = list(model.pars.index)
    n_p = len(par_names)
    lower = model.pars['min']
    bounds = []
    for i in range(len(model.pars)):
        bounds += [(model.pars['min'][i] + 0.001, model.pars['max'][i] - 0.001)]
        
    # set up data frame
    col_index = par_names + ['prop_log_post']
    df = pd.DataFrame(0.0, index = range(n), columns = col_index) 
    
    # maximize log-likelihood/posterior
    for i in range(n):
        pct = np.round(100*(i + 1)/n, 1)
        print('Fitting ' + str(i + 1) + ' of ' + str(n) + ' (' + str(pct) + '%)')
        ds_i = ds.loc[{'ident' : ds.ident[i]}].squeeze()
        
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
        gopt = nlopt.opt(nlopt.GN_ORIG_DIRECT, n_p)
        gopt.set_max_objective(f)
        gopt.set_lower_bounds(np.array(model.pars['min'] + 0.001))
        gopt.set_upper_bounds(np.array(model.pars['max'] - 0.001))
        gopt.set_maxtime(max_time/2)
        gxopt = gopt.optimize(x0_i)
        # local optimization (to refine answer)
        lopt = nlopt.opt(nlopt.LN_SBPLX, n_p)
        lopt.set_max_objective(f)
        lopt.set_lower_bounds(np.array(model.pars['min'] + 0.001))
        lopt.set_upper_bounds(np.array(model.pars['max'] - 0.001))
        lopt.set_maxtime(max_time/2)
        lxopt = lopt.optimize(gxopt)
        
        df.loc[i, par_names] = lxopt
        df.loc[i, 'prop_log_post'] = lopt.last_optimum_value()
    # set index to 'ident'
    df = df.set_index(ds.ident.to_series(), drop = True)
        
    return df

def fit_em(model, ds, max_em_iter = 5, max_time = 10):
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
        
    max_time: int, optional
        Maximum time (in seconds) per individual per EM iteration.
        Defaults to 10.

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
    result = fit_indv(model, ds, None, None, max_time)
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
        result = fit_indv(model, ds, x0, [E_tau0, E_tau1], max_time)
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
