import numpy as np
import pandas as pd
import xarray as xr
from copy import deepcopy

class schedule:
    """
    Class to represent an experimental schedule, i.e. a scheme of
    related stimuli and outcomes given to a group of learners.

    Attributes
    ----------
    name: str or None
        The schedule's name.  By default this is None
        until the schedule is used in the creation of
        an experiment object.
    resp_type : str
        The type of behavioral response made by the learner.  Can
        be either 'choice' (discrete responses), 'exct' (excitatory)
        or 'supr' (suppression of an ongoing activity).
    stages: dict
        Dictionary of experimental stages (stage objects).
    delays: list of int
        List of time delays between stages (e.g. if delay[1] = 100
        then there is a 100 unit delay between the end of stage 0 and
        the start of stage 1).
    trial_def: data frame
        Defines the trial types implied by 'stages'.
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
    Although it is a dict, the 'stages' attribute retains stage order 
    in modern versions of Python (>= 3.7).  This is important.
    
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
    'trial', 'trial_name', 'stage' and 'stage_name' are all 
    alternative labels for the time dimension.  Because xarray does not
    fully support multi-indexing (e.g. vectorized indexing doesn't work, 
    at least in v0.15.1), these other time labels are specified as separate
    data variable instead of coordinates of the time dimension alongside 't'.

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
    def __init__(self, resp_type, stages, delays = None, x_dims = None):
        """
        Parameters
        ----------
        resp_type: str
            The type of behavioral response made by the learner.  Can
            be either 'choice' (discrete responses), 'exct' (excitatory)
            or 'supr' (suppression of an ongoing activity).
        stages: dict
            Dictionary of experimental stages (stage objects).
        delays: list of int or None, optional
            List of time delays between stages (e.g. if delay[0] = 100
            then there is a 100 unit delay between the end of stage 0 and
            the start of stage 1).  If None, then there are no delays (all are 0).
            Defaults to None.
        x_dims: dict or None, optional
            If not None, then a dictionary specifying the cues belonging to
            each stimulus dimension.  Keys are dimension names and values
            are cue names (i.e. 'x_names').  Defaults to None.
        """
        # assemble names; count things
        n_stage = len(stages)
        stage_names = []
        stage_number = []
        x_names = []
        u_names = []
        n_t = 0 # total number of time steps in the whole experiment
        n_t_trial_def = 0 # total number of time steps with one trial of each type
        # loop through stages
        i = 0 # index for stages
        for st in stages:
            n_t_trial_def += (stages[st].iti + 1)*stages[st].n_trial_type
            n_t += (stages[st].iti + 1)*np.sum(stages[st].freq)*stages[st].n_rep
            stage_names += [stages[st].n_trial_type*st]
            stage_number += [stages[st].n_trial_type*i]
            x_names += stages[st].x_names
            u_names += stages[st].u_psb
            for j in range(stages[st].n_trial_type):
                u_new = stages[st].u[j]
                u_names += u_new
            i += 1
        x_names = list(np.unique(x_names))
        u_names = list(np.unique(u_names))
        n_x = len(x_names)
        n_u = len(u_names)
        # this is the number of trial types, not the number of trials in the experiment
        n_trial = 0
        for st in stages:
            n_trial += stages[st].n_trial_type
        # check if cues (x) have varying values
        has_varying_x_value = [] # cues that have x_value which varies
        for xn in stages[st].x_names:
            x_val_st = [] # value for cue xn in each stage
            for st in stages:
                if xn in stages[st].x_names:
                    x_val_st += [stages[st].x_value[xn]]
            if len(np.unique(x_val_st)) > 1:
                has_varying_x_value += [xn] # more than one value of cue xn across stages -> add to cue xn list
            
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
        i = 0 # stage index
        # loop through stages
        for st in stages:
            iti = stages[st].iti
            stage += (iti + 1)*stages[st].n_trial_type*[i]
            stage_name += (iti + 1)*stages[st].n_trial_type*[st]
            # figure out cue names for use in exemplar naming (may include x_value if that is not only 0.0 or 1.0)
            x_names_ex = pd.Series('', index = stages[st].x_names)
            for xn in stages[st].x_names:
                if xn in has_varying_x_value:
                    x_names_ex[xn] = xn + str(stages[st].x_value[xn])
                else:
                    x_names_ex[xn] = xn
            # loop through trial types
            for j in range(stages[st].n_trial_type):
                trial += (iti + 1)*[j]
                # figure out trial name
                has_x_pn = len(stages[st].x_pn[j]) > 0 # indicates whether there are punctate cues (x_pn)
                has_u = len(stages[st].u[j]) > 0 # indicates whether there are outcomes (u)
                possible_names = {(True, True): '.'.join(stages[st].x_pn[j]) + ' -> ' + '.'.join(stages[st].u[j]),
                                  (True, False): '.'.join(stages[st].x_pn[j]) + ' -> ' + 'nothing',
                                  (False, True): 'nothing' + ' -> ' + '.'.join(stages[st].u[j]),
                                  (False, False): 'background'}
                new_trial_name = possible_names[(has_x_pn, has_u)]
                trial_name += (iti + 1)*[new_trial_name]                
                # other information
                x.loc[{'row': range(k, k + iti + 1), 'x_name': stages[st].x_bg}] = stages[st].x_value.loc[stages[st].x_bg] # background cues (x_bg)
                u_psb.loc[{'row': range(k, k + iti + 1), 'u_name': stages[st].u_psb}] = 1.0
                if stages[st].lrn == True:
                    u_lrn.loc[{'row': range(k, k + iti + 1), 'u_name': stages[st].u_psb}] = 1.0
                # yet more information
                has_main = has_x_pn or has_u # indicates whether there is a 'main' time step
                if has_main:
                    # set up time steps before 'main' (if there are any)
                    if iti > 0:
                        t_name += (iti - 1)*['bg']
                        t_name += ['pre_main']
                        ex_name += iti*['.'.join(x_names_ex[stages[st].x_bg])]
                    # set up 'main', i.e. time step with punctate cues/outcomes
                    t_name += ['main']
                    if has_x_pn:
                        x.loc[{'row': k + iti, 'x_name': stages[st].x_pn[j]}] = stages[st].x_value.loc[stages[st].x_pn[j]] # punctate cues (x_pn)
                        ex_name += ['.'.join(x_names_ex[stages[st].x_bg + stages[st].x_pn[j]])]
                    else:
                        ex_name += ['.'.join(x_names_ex[stages[st].x_bg])]
                    if has_u:
                        u.loc[{'row': k + iti, 'u_name': stages[st].u[j]}] = stages[st].u_value.loc[stages[st].u[j]]
                else:
                    t_name += (iti + 1)*['bg']
                    ex_name += (iti + 1)*['.'.join(x_names_ex[stages[st].x_bg])]
                # advance time step and stage indices
                k += iti + 1
                i += 1

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
                                         'u_name': u_names})
        
        # create a dataframe for exemplars, and attach to trial type dataset as an attribute
        ex_array, ex_index = np.unique(trial_def['x'], axis = 0, return_index = True)
        ex_names = trial_def['ex'].loc[{'t': ex_index}].values
        x_ex = pd.DataFrame(ex_array, index = ex_names, columns = x_names)
        trial_def = trial_def.assign_attrs(x_ex = x_ex, ex_names = ex_names, resp_type = resp_type)

        # make sure that no trial type is duplicated within any stage
        for st in stages:
            indexer = (trial_def.stage_name == st) & (trial_def.t_name == 'main')
            trial_names = trial_def.loc[{'t' : indexer}].trial_name
            all_unique = len(trial_names) == len(np.unique(trial_names))
            assert all_unique, 'Duplicate trial definition found in stage "{}".'.format(st)
                
        # record information in new object ('self')
        self.name = None # the schedule doesn't get a real name attribute until put in an experiment object
        self.resp_type = resp_type
        self.stages = deepcopy(stages)
        for st in stages:
            self.stages[st].name = st # assign stage name attribute based on dictionary key
        if delays is None:
            self.delays = (n_stage - 1)*[0]
        else:
            self.delays = delays
        self.trial_def = trial_def
        self.x_names = x_names
        self.u_names = u_names
        self.ex_names = ex_names
        self.n_stage = n_stage        
        self.n_x = n_x
        self.n_u = n_u
        self.n_t = int(n_t)
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