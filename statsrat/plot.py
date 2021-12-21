import numpy as np
import pandas as pd
from plotnine import *

def single(ds, var, sel = None, rename_coords = None, color_var = None, facet_var = None, color = None, draw_points = False, drop_zeros = False, only_main = False, stage_labels = True, text_size = 15.0, figure_size = (4.0, 2.5), dodge_width = 1.0, y_axis_label = None):
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
    plot += theme(figure_size = figure_size)
    plot += theme(axis_text_x = element_text(ha = 'left'))
    if not y_axis_label is None:
        plot += ylab(y_axis_label)
    
    return plot

def multiple(ds_list, var, sel = None, rename_coords = None, linetype_var = None, rename_schedules = None, color = None, draw_points = False, drop_zeros = False, only_main = False, stage_labels = True, text_size = 15.0, figure_size = (4.0, 2.5), dodge_width = 1.0, y_axis_label = None):
    """
    Plots learning simulation data from multiple schedules (conditions, groups) as a function of time.
    Schedules (groups) are represented by color.
    
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
    linetype_var : string, optional
        Variable to be represented by linetype.  Defaults to None (see notes).
    rename_schedules : dict or None, optional
        Either a dictionary for re-naming schedules (keys are old names and
        values are new names), or None (don't re-name).  Defaults to None.
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
        if rename_schedules is None:
            new_df['schedule'] = ds.attrs['schedule']
        else:
            new_df['schedule'] = rename_schedules[ds.attrs['schedule']]
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
        if not 'schedule' in rename_coords.keys():
            var_names['schedule'] = 'schedule'
    else:
        var_names = dict.fromkeys(dims)
        for i in range(n_dims):
            var_names[dims[i]] = dims[i]
        var_names['schedule'] = 'schedule'

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
        plot = (ggplot(df, aes(x='t', y=var, color=var_names['schedule'])) + geom_line(position = dpos))
    elif n_dims == 1:
        plot = (ggplot(df, aes(x='t', y=var, color=var_names['schedule'], linetype=linetype_var)) + geom_line(position = dpos))
    else: # n_dims == 2 (doesn't work properly if n_dims is 3 or more)
        plot = (ggplot(df, aes(x='t', y=var, color=var_names['schedule'], linetype=linetype_var)) + geom_line(position = dpos) + facet_wrap('~' + facet_var))
    
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
    plot += theme(figure_size = figure_size)
    plot += theme(axis_text_x = element_text(ha = 'left'))
    if not y_axis_label is None:
        plot += ylab(y_axis_label)
                        
    return plot