import numpy as np
import pandas as pd
from plotnine import *

def single(ds, var, hline = None, sel = None, rename_coords = None, color_var = None, facet_var = None, color = None, draw_points = False, drop_zeros = False, only_main = False, trial_name_list = None, stage_name_list = None, stage_labels = True, text_size = 15.0, figure_size = (4.0, 2.5), dodge_width = 1.0, y_axis_label = None):
    """
    Plots learning simulation data from a single schedule (condition, group) as a function of time.
    
    Parameters
    ----------
    ds : dataset (xarray)
        Learning simulation data (output of a model's 'simulate' method).
    var : string
        Variable to plot.
    hline : float or None, optional
        If a float, then a horizontal dashed line is drawn at that y axis position.
        If None (the default), then no such line is drawn.
    sel : dict, optional
        Used to select a subset of 'var'.  Defaults to 'None' (i.e. all
        data in 'var' are plotted).
    rename_coords : dict or None, optional
        Either a dictionary for re-naming coordinates (keys are old names and
        values are new names), or None (don't re-name).  This dict must include
        ALL variable names.  Defaults to None.
    color_var : string, optional
        Variable to be represented by color.  Defaults to None (see notes).
    facet_var : string, optional
        Variable to control faceting.  Defaults to None (see notes).
    color : string or dict or None, optional
        If there is only one level of the variable plotted, then it should be a
        string specifying line color.  If there are several levels then it should
        be a dictionary with level names as keys as strings indicating colors as
        values.  Defaults to None, which leads to the default colors.
    draw_points : boolean, optional
        Whether or not points should be drawn as well as lines.
        Defaults to False.
    drop_zeros : boolean, optional
        Drop rows where 'var' is zero.  Defaults to False.
    only_main : boolean, optional
        Only keep rows where 't_name' is 'main', i.e. time steps with
        punctate cues and/or non-zero outcomes.  Defaults to False.
    trial_name_list : list or None, optional
        Either a list of trial type names (strings) to keep, or else None
        (keep all trial types, the default).
    stage_name_list : list or None, optional
        Either a list of stage names (strings) to keep, or else None
        (keep all stages, the default).
    stage_labels : boolean, optional
        Whether the x-axis should be labeled with 'stage_name' (if True) or
        't', i.e. time step (if False).  Defaults to True.
    text_size : float, optional
        Specifies text size.  Defaults to 15.0.
    figure_size : tuple of floats, optional
        Figure width and height in inches.  Defaults to (4.0, 2.5).
    dodge_width : float, optional
        Amount to separate overlapping lines so that they appear visually
        distinct (using Plotnine's position_dodge).  Defaults to 1.0.
    y_axis_label : str or None, optional
        Specifies an alternative name for the y axis label (or is None).
        Defaults to None.
        
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
    if not stage_name_list is None:
        ds = ds.loc[{'t': ds['stage_name'].isin(stage_name_list)}]
    if not trial_name_list is None:
        ds = ds.loc[{'t': ds['trial_name'].isin(trial_name_list)}]
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
        if color is None:
            plot = (ggplot(df, aes(x='t', y=var)) + geom_line())
        else:
            plot = (ggplot(df, aes(x='t', y=var)) + geom_line(color = color))
    else:
        if color_var is None:
            color_var = var_names[dims[0]]
        if n_dims == 1:
            plot = (ggplot(df, aes(x='t', y=var, color=color_var)) + geom_line(position = dpos))
        else:
            if facet_var is None:
                facet_var = var_names[dims[1]]
            plot = (ggplot(df, aes(x='t', y=var, color=color_var)) + geom_line(position = dpos) + facet_wrap('~' + facet_var))
        if not color is None:
            plot += scale_color_manual(values = color)
    
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
    
    plot += theme_classic(base_size = text_size) # set text size and use "classic" theme
    plot += theme(figure_size = figure_size, axis_text_x = element_text(ha = 'left'), legend_key_height = 3) # Change figure size, align x axis text to the left, and squish legend lines closer together, 
    if not y_axis_label is None:
        plot += ylab(y_axis_label)
    
    # draw a dashed horizontal line if desired
    if not hline is None:
        plot += geom_hline(yintercept = hline, linetype = 'dashed')
    
    return plot

def multiple(ds_list, var, hline = None, condition_names = None, sel = None, rename_coords = None, linetype_var = None, rename_schedules = None, color = None, draw_points = False, drop_zeros = False, only_main = False, trial_name_list = None, stage_name_list = None, stage_labels = True, text_size = 15.0, figure_size = (4.0, 2.5), dodge_width = 1.0, y_axis_label = None):
    """
    Plots learning simulation data from multiple conditions (e.g. groups) as a function of time.
    Conditions are represented by color.
    
    Parameters
    ----------
    ds_list : list of datasets (xarray)
        Each element of the list consists of learning simulation data
        (output of a model's 'simulate' method) from a different schedule.
    var : string
        Variable to plot.
    hline : float or None, optional
        If a float, then a horizontal dashed line is drawn at that y axis position.
        If None (the default), then no such line is drawn.
    condition_names: list or None, optional
        List of names for conditions, corresponding to the datasets in ds_list.
        If None, then the datasets' schedule names are used as a default (assuming
        that they come from different schedules).
    sel : list of dicts or None, optional
        If a list, then elements correspond to elements of ds_list.
        Each list element is either None (to include everything in the 
        corresponding data set) or a dict used to select a subset of 'var'.
        Defaults to None (i.e. all data in 'var' are plotted for all data
        sets).
    rename_coords : dict or None, optional
        Either a dictionary for re-naming coordinates (keys are old names and
        values are new names), or None (don't re-name).  Defaults to None.
    linetype_var : string, optional
        Variable to be represented by linetype.  Defaults to None (see notes).
    rename_schedules : dict or None, optional
        DEPRECATED: use condition_names instead.  Used only when condition_names is
        None, and hence condition names are schedule names.  Either a dictionary for
        re-naming schedules (keys are old names and values are new names), or None 
        (don't re-name).  Defaults to None.
    color : dict or None, optional
        If a dict, specifies the color for each schedule (group).  If None
        (default) then the default colors are used.
    draw_points : boolean, optional
        Whether or not points should be drawn as well as lines.
        Defaults to False.
    drop_zeros : boolean, optional
        Drop rows where 'var' is zero.  Defaults to False.
    only_main : boolean, optional
        Only keep rows where 't_name' is 'main', i.e. time steps with
        punctate cues and/or non-zero outcomes.  Defaults to False.
    trial_name_list : list or None, optional
        Either a list of trial type names (strings) to keep, or else None
        (keep all trial types, the default).
    stage_name_list : list or None, optional
        Either a list of stage names (strings) to keep, or else None
        (keep all stages, the default).
    stage_labels : boolean, optional
        Whether the x-axis should be labeled with 'stage_name' (if True) or
        't', i.e. time step (if False).  Defaults to True.
    text_size : float, optional
        Specifies text size.  Defaults to 15.0.
    figure_size : tuple of floats, optional
        Figure width and height in inches.  Defaults to (4.0, 2.5).
    dodge_width : float, optional
        Amount to separate overlapping lines so that they appear visually
        distinct (using Plotnine's position_dodge).  Defaults to 1.0.
    y_axis_label : str or None, optional
        Specifies an alternative name for the y axis label (or is None).
        Defaults to None.
        
    Returns
    -------
    plot : object
        A plotnine plot object.
    
    Notes
    -----
    It is assumed that either the schedules all have the same stage names, or else
    that the 'sel' argument is used to select time steps that all have the same stage names.
    
    By default, the first non-time dimension will be used for linetype and the
    second one for faceting.
    
    The 'sel' argument is used to index 'ds' via the latter's 'loc' method.
    It should be a dictionary of the form {'dim0' : ['a', 'b'], 'dim1' : ['c']},
    where 'dim0', 'dim1' etc. are one or more dimensions of 'var'.
    """
    ### CREATE INDIVIDUAL DATA FRAMES ###
    df_list = []
    i = 0
    for ds in ds_list:
        if not stage_name_list is None:
            ds = ds.loc[{'t': ds['stage_name'].isin(stage_name_list)}]
        if not trial_name_list is None:
            ds = ds.loc[{'t': ds['trial_name'].isin(trial_name_list)}]
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
        if condition_names is None:
            if rename_schedules is None:
                new_df['condition'] = ds.attrs['schedule']
            else:
                new_df['condition'] = rename_schedules[ds.attrs['schedule']]
        else:
            new_df['condition'] = condition_names[i]
        df_list += [new_df]
        i += 1
    dims = new_dims # names of dimensions in graph (excluding schedule/group)
    n_dims = len(dims) # number of dimensions in graph (excluding schedule/group)

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
        if not 'condition' in rename_coords.keys():
            var_names['condition'] = 'condition'
    else:
        var_names = dict.fromkeys(dims)
        for i in range(n_dims):
            var_names[dims[i]] = dims[i]
        var_names['condition'] = 'condition'

    ### FIGURE OUT HOW VARIABLES ARE REPRESENTED IN THE GRAPH ###
    if n_dims > 0:
        if linetype_var is None:
            linetype_var = var_names[dims[0]]
        if n_dims > 1: # n_dims == 2 (doesn't work properly if n_dims is 3 or more)
            var_names.pop(linetype_var)
            facet_var = list(var_names.values())[0]
    
    ### CREATE PLOT ###
    dpos = position_dodge(width = dodge_width)
    if n_dims == 0:
        plot = (ggplot(df, aes(x='t', y=var, color=var_names['condition'])) + geom_line(position = dpos))
    elif n_dims == 1:
        plot = (ggplot(df, aes(x='t', y=var, color=var_names['condition'], linetype=linetype_var)) + geom_line(position = dpos))
    else: # n_dims == 2 (doesn't work properly if n_dims is 3 or more)
        plot = (ggplot(df, aes(x='t', y=var, color=var_names['condition'], linetype=linetype_var)) + geom_line(position = dpos) + facet_wrap('~' + facet_var))
    
    if not color is None:
        plot += scale_color_manual(values = color)
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
    
    plot += theme_classic(base_size = text_size) # set text size and use "classic" theme
    plot += theme(figure_size = figure_size, axis_text_x = element_text(ha = 'left'), legend_key_height = 3) # Change figure size, align x axis text to the left, and squish legend lines closer together
    if not y_axis_label is None:
        plot += ylab(y_axis_label)
    
    # draw a dashed horizontal line if desired
    if not hline is None:
        plot += geom_hline(yintercept = hline, linetype = 'dashed')
    
    return plot

def mean_resp(ds_list, stage_name, trial_name, y_name, t_name = 'main', condition_names = None, text_size = 15.0, figure_size = (4.0, 2.5)):
    """
    Plots mean responses (b_hat) across multiple conditions (e.g. groups).
    These mean responses are from a specified stage (stage_name), trial type (trial_name),
    time step name (t_name), and response/outcome name (y_name).
    
    Parameters
    ----------
    ds_list : list of datasets (xarray)
        Each element of the list consists of learning simulation data
        (output of a model's 'simulate' method) from a different schedule.
    stage_name : str
        Name of the stage from which responses will be selected.
    trial_name : str
        Name of the trial type from which responses will be selected.
    y_name : str
        Name of the response/outcome selected.
    t_name : str, optional
        Name of the time step type from which responses will be selected.
        Defaults to 'main'.
    condition_names: list or None, optional
        List of names for conditions, corresponding to the datasets in ds_list.
        If None, then the datasets' schedule names are used as a default (assuming
        that they come from different schedules).
    text_size : float, optional
        Specifies text size.  Defaults to 15.0.
    figure_size : tuple of floats, optional
        Figure width and height in inches.  Defaults to (4.0, 2.5).
        
    Returns
    -------
    plot : object
        A plotnine plot object.
    """
    if condition_names is None:
        c_names = []
        for ds in ds_list:
            c_names += [ds.attrs['schedule']]
    n_c = len(ds_list)
    df = pd.DataFrame({'mean response': np.zeros(n_c),
                       'condition': c_names})
    for i in range(n_c):
        selector = ((ds_list[i]['stage_name'] == stage_name)&(ds_list[i]['trial_name'] == trial_name)&(ds_list[i]['t_name'] == t_name)).values
        df.loc[df['condition'] == c_names[i], 'mean response'] = ds_list[i].loc[{'t': selector, 'y_name': y_name}]['b'].mean().values
    
    plot = (ggplot(df, aes(x = 'condition', y = 'mean response')) + geom_point())
    plot += theme_classic(base_size = text_size) # set text size and use "classic" theme
    plot += theme(figure_size = figure_size, axis_text_x = element_text(ha = 'left'), legend_key_height = 3) # Change figure size, align x axis text to the left, and squish legend lines closer together
    
    return plot