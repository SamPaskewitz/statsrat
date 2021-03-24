import numpy as np
import pandas as pd
import xarray as xr

class schedule:
    """
    Class to represent an experimental schedule, i.e. a scheme of
    related stimuli and outcomes given to a group of learners.

    Attributes
    ----------
    name: str
        The schedule's name.
    resp_type : str
        The type of behavioral response made by the learner.  Can
        be either 'choice' (discrete responses), 'exct' (excitatory)
        or 'supr' (suppression of an ongoing activity).
    stage_list: list
        List of dictionaries defining the schedule's stages.
    trial_def: data frame
        Defines the trial types implied by 'stage_list'.
    x_names: list of str
        Names of cues (stimulus attributes).
    x_dims: dict or None
        If not None, then a dictionary specifying the cues belonging to
        each stimulus dimension.  Keys are dimension names and values
        are cue names (i.e. 'x_names').
    dim_names: list of str or None
        Names of stimulus dimensions.
    u_names: list of str
        Names of outcomes.
    ex_names: list of str
        Names of exemplars (unique cue combinations).
    n_stage: int
        The number of stages.
    n_x: int
        The number of cues.
    n_dim: int or None
        The number of stimulus dimensions.
    n_u: int
        The number of outcomes.
    n_t: int
        The number of time steps in the entire schedule.
    n_ex: int
        The number of exemplars (unique cue combinations).

    Notes
    -----
    In an experiment with several between subjects conditions, each such
    condition corresponds to a schedule.

    Experiments without any between subjects conditions only have a
    single schedule.

    Each schedule is composed of several stages, as described in the
    documentation for the '__init__' method below.
    
    Stimulus dimensions (specified by the 'x_dims' attribute) are sets
    of related cues, e.g. the cues 'red', 'blue' and 'yellow' might
    all belong to the dimension 'color'.  This is optional information
    that is used by certain learning models.

    In trial_def and trials dataset objects, 't' is the dimension that
    indicates time steps (it is simply an ascending integer).  't_name',
    'trial', 'trial_name', 'stage' and 'stage_name' are all alternative
    labels for the time dimension.  Because xarray does not fully support
    multi-indexing (e.g. vectorized indexing doesn't work, at least in
    v0.15.1), these other time labels are specified as separate data
    variable instead of coordinates of the time dimension alongside 't'.

    Data variables of trial_def and trials dataset objects:
    x: float
        Stimulus attributes, i.e. cues.  Absent cues are indicated by 0.0
        and present cues by 1.0.
    u: float
        Outcomes, e.g. unconditioned stimuli or category labels.  A value
        of 0.0 indicates that an outcome does not occur.
    u_psb: float
        Indicates whether an outcome is possible (1.0) or not possible
        (0.0) as far as the learner knows.  The learner will not try to
        predict outcomes that it knows are not possible.
    u_lrn: float
        Indicates whether learning about each outcome should occur (1.0) or
        not (0.0).  Should be 0.0 only for test stages without feedback in
        human experiments.
    
    Coordinate variables of trial_def and trials dataset objects:
    t: int
        Time step dimension.
    t_name: str
        Alternative coordinate for time steps (dimension t).
        Labels time steps as 'main' when at least one punctate cue is present
        and 'bg' ('background') otherwise (e.g. during the inter-trial
        interval).
    ex: str
        Alternative coordinate for time steps (dimension t).
        Specifies the combination of cues (both background and punctate)
        present during that time step.
    trial: int
        Alternative coordinate for time steps (dimension t).
        Each trial consists of one or more time steps.  This indicates
        which time steps correspond to each trial.  The ITI (inter-trial
        interval) is considered part of the trial that it precedes.
    trial_name: str
        Alternative coordinate for time steps (dimension t).
        Name of the trial type.  Has the form 'cues -> outcomes'.
    stage : int
        Alternative coordinate for time steps (dimension t).
        Indicates experimental stage by order.
    stage_name: str
        Alternative coordinate for time steps (dimension t).
        Indicates experimental stage by name.
    u_name: str
        Outcome/CS/response dimension.
    x_name: str
        Cue name dimension.        
    """
    def __init__(self, name, resp_type, stage_list, x_dims = None):
        """
        Parameters
        ----------
        name: str
            The name of the schedule.
        resp_type: str
            The type of behavioral response made by the learner.  Can
            be either 'choice' (discrete responses), 'exct' (excitatory)
            or 'supr' (suppression of an ongoing activity).
        stage_list: list
            List of experimental stages (stage objects).
        x_dims: dict or None, optional
            If not None, then a dictionary specifying the cues belonging to
            each stimulus dimension.  Keys are dimension names and values
            are cue names (i.e. 'x_names').  Defaults to None.
        """
        # assemble names; count things
        n_stage = len(stage_list)
        stage_names = []
        stage_number = []
        x_names = []
        u_names = []
        n_t = 0 # total number of time steps in the whole experiment
        n_t_trial_def = 0 # total number of time steps with one trial of each type
        # loop through stages
        has_varying_x_value = [] # cues that have x_value which varies (more accurately is sometimes different from 0 or 1)
        for i in range(n_stage):
            n_t_trial_def += (stage_list[i].iti + 1)*stage_list[i].n_trial_type
            n_t += (stage_list[i].iti + 1)*np.sum(stage_list[i].freq)*stage_list[i].n_rep
            stage_names += [stage_list[i].n_trial_type*stage_list[i].name]
            stage_number += [stage_list[i].n_trial_type*i]
            x_names += stage_list[i].x_names
            u_names += stage_list[i].u_psb
            for j in range(stage_list[i].n_trial_type):
                u_new = stage_list[i].u[j]
                u_names += u_new
            for xn in stage_list[i].x_names:
                if not stage_list[i].x_value[xn] in [0.0, 1.0]:
                    has_varying_x_value += [xn]
        has_varying_x_value = list(np.unique(has_varying_x_value))
        x_names = list(np.unique(x_names))
        u_names = list(np.unique(u_names))
        n_x = len(x_names)
        n_u = len(u_names)
        # this is the number of trial types, not the number of trials in the experiment
        n_trial = 0
        for s in stage_list:
            n_trial += s.n_trial_type            
            
        # loop through trial types to add information
        x = xr.DataArray(np.zeros((n_t_trial_def, n_x)), [range(n_t_trial_def), x_names], ['row', 'x_name'])
        u = xr.DataArray(np.zeros((n_t_trial_def, n_u)), [range(n_t_trial_def), u_names], ['row', 'u_name'])
        u_psb = xr.DataArray(np.zeros((n_t_trial_def, n_u)), [range(n_t_trial_def), u_names], ['row', 'u_name'])
        u_lrn = xr.DataArray(np.zeros((n_t_trial_def, n_u)), [range(n_t_trial_def), u_names], ['row', 'u_name'])
        stage = []
        stage_name = []
        trial = []
        trial_name = []
        t = []
        t_name = []
        ex_name = []           
        
        k = 0 # time step index
        # loop through stages
        for i in range(n_stage):
            iti = stage_list[i].iti
            stage += (iti + 1)*stage_list[i].n_trial_type*[i]
            stage_name += (iti + 1)*stage_list[i].n_trial_type*[stage_list[i].name]
            # figure out cue names for use in exemplar naming (may include x_value if that is not only 0.0 or 1.0)
            x_names_ex = pd.Series('', index = stage_list[i].x_names)
            for xn in stage_list[i].x_names:
                if xn in has_varying_x_value:
                    x_names_ex[xn] = xn + str(stage_list[i].x_value[xn])
                else:
                    x_names_ex[xn] = xn
            # loop through trial types
            for j in range(stage_list[i].n_trial_type):
                trial += (iti + 1)*[j]
                # figure out trial name
                has_x_pn = len(stage_list[i].x_pn[j]) > 0 # indicates whether there are punctate cues (x_pn)
                has_u = len(stage_list[i].u[j]) > 0 # indicates whether there are outcomes (u)
                possible_names = {(True, True): '.'.join(stage_list[i].x_pn[j]) + ' -> ' + '.'.join(stage_list[i].u[j]),
                                  (True, False): '.'.join(stage_list[i].x_pn[j]) + ' -> ' + 'nothing',
                                  (False, True): 'nothing' + ' -> ' + '.'.join(stage_list[i].u[j]),
                                  (False, False): 'background'}
                new_trial_name = possible_names[(has_x_pn, has_u)]
                trial_name += (iti + 1)*[new_trial_name]                
                # other information
                x.loc[{'row': range(k, k + iti + 1), 'x_name': stage_list[i].x_bg}] = stage_list[i].x_value.loc[stage_list[i].x_bg] # background cues (x_bg)
                u_psb.loc[{'row': range(k, k + iti + 1), 'u_name': stage_list[i].u_psb}] = 1.0
                if stage_list[i].lrn == True:
                    u_lrn.loc[{'row': range(k, k + iti + 1), 'u_name': stage_list[i].u_psb}] = 1.0
                # yet more information
                has_main = has_x_pn or has_u # indicates whether there is a 'main' time step
                if has_main:
                    # set up time steps before 'main' (if there are any)
                    if iti > 0:
                        t_name += (iti - 1)*['bg']
                        t_name += ['pre_main']
                        ex_name += iti*['.'.join(x_names_ex[stage_list[i].x_bg])]
                    # set up 'main', i.e. time step with punctate cues/outcomes
                    t_name += ['main']
                    if has_x_pn:
                        x.loc[{'row': k + iti, 'x_name': stage_list[i].x_pn[j]}] = stage_list[i].x_value.loc[stage_list[i].x_pn[j]] # punctate cues (x_pn)
                        ex_name += ['.'.join(x_names_ex[stage_list[i].x_bg + stage_list[i].x_pn[j]])]
                    else:
                        ex_name += ['.'.join(x_names_ex[stage_list[i].x_bg])]
                    if has_u:
                        u.loc[{'row': k + iti, 'u_name': stage_list[i].u[j]}] = stage_list[i].u_value.loc[stage_list[i].u[j]]
                else:
                    t_name += (iti + 1)*['bg']
                    ex_name += (iti + 1)*['.'.join(x_names_ex[stage_list[i].x_bg])]
                # advance time step index
                k += iti + 1                

        # create dataset for trial type definitions ('trial_def')
        trial_def = xr.Dataset(data_vars = {'x': (['t', 'x_name'], x),
                                            'u': (['t', 'u_name'], u),
                                            'u_psb': (['t', 'u_name'], u_psb),
                                            'u_lrn': (['t', 'u_name'], u_lrn)},
                               coords = {'t': range(len(stage)),
                                         't_name': ('t', t_name),
                                         'ex': ('t', ex_name),
                                         'trial': ('t', trial),
                                         'trial_name': ('t', trial_name),
                                         'stage': ('t', stage),
                                         'stage_name': ('t', stage_name),
                                         'x_name': x_names,
                                         'u_name': u_names,
                                         'schedule': name})
        
        # create a dataframe for exemplars, and attach to trial type dataset as an attribute
        ex_array, ex_index = np.unique(trial_def['x'], axis = 0, return_index = True)
        ex_names = trial_def['ex'].loc[{'t': ex_index}].values
        x_ex = pd.DataFrame(ex_array, index = ex_names, columns = x_names)
        trial_def = trial_def.assign_attrs(x_ex = x_ex, ex_names = ex_names, resp_type = resp_type, schedule = name)

        # make sure that no trial type is duplicated within any stage
        for i in range(n_stage):
            indexer = (trial_def.stage == i) & (trial_def.t_name == 'main')
            trial_names = trial_def.loc[{'t' : indexer}].trial_name
            all_unique = len(trial_names) == len(np.unique(trial_names))
            assert all_unique, 'Duplicate trial definition found in stage "{}" of schedule "{}".'.format(stage_list[i].name, name)
                
        # record information in new object ('self')
        self.name = name
        self.resp_type = resp_type
        self.stage_list = stage_list
        self.trial_def = trial_def
        self.x_names = x_names
        self.u_names = u_names
        self.ex_names = ex_names
        self.n_stage = n_stage        
        self.n_x = n_x
        self.n_u = n_u
        self.n_t = n_t
        self.n_ex = len(self.ex_names)
        
        # record stimulus dimension info, if any
        if not x_dims is None:
            self.x_dims = x_dims
            self.dim_names = list(x_dims.keys())
            self.n_dim = len(self.dim_names)
            self.trial_def = trial_def.assign_attrs(x_dims = self.x_dims)
        else:
            self.x_dims = None
            self.dim_names = None
            self.n_dim = None