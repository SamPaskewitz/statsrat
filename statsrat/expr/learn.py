import numpy as np
import pandas as pd
import xarray as xr
import glob
from scipy.stats import t

class experiment:
    """
    A class used to represent learning experiments.

    Attributes
    ----------
    resp_type : str
        The type of behavioral response made by the learner.
    schedules : dict
        A dictionary of the experiment's schedules (sequences of stimuli and feedback etc
        that typically correspond to groups in the experimental design).
    oats : dict
        A dictionary of the experiment's ordinal adequacy tests (OATs).
    notes : str or None
        Notes on the experiment (e.g. explanation of design, references).

    Methods
    -------
    make_trials(self)
        Create a time step level dataset for the whole experiment.
    read_csv(self, path, x_col, resp_col, resp_map, ident_col = None, conf_col = None, schedule = None, other_info = None, header = 'infer', n_final = 8)
        Import empirical data from .csv files.

    See Also
    --------
    See 'predef.cat' for category learning examples.
    
    See 'predef.pvl' for Pavlovian conditioning examples.
    """
    def __init__(self, resp_type, schedules, oats, notes = None):
        """
        Parameters
        ----------
        resp_type : str
            The type of behavioral response made by the learner.
        schedules : dict
            A dictionary of the experiment's schedules (sequences of stimuli and feedback etc
            that typically correspond to groups in the experimental design).
        oats : dict
            A dictionary of the experiment's ordinal adequacy tests (OATs).
        notes : str or None, optional
            Notes on the experiment (e.g. explanation of design, references).
            Defaults to None (i.e. no notes).
        """
        # check that everything in the 'schedules' argument is a schedule object
        is_scd = []
        for s in schedules:
            is_scd += [isinstance(s, schedule)]
        assert False in is_scd, 'Non-schedule object input as schedule.'
        # check that everything in the 'oat' argument is an oat object
        if len(oats) > 0:
            is_oat = []
            for o in oats:
                is_oat += [isinstance(o, oat)]
            assert False in is_oat, 'Non-oat object input as oat.'
        # add stuff to 'self'
        self.resp_type = resp_type
        self.schedules = schedules
        self.oats = oats
        self.notes = notes

    def make_trials(self, schedule = None):
        """
        Create a time step level dataset for the whole experiment.

        Parameters
        ----------
        schedule : str, optional
            Name of the schedule from which to make trials.  By default
            selects the first schedule in the experiment object's
            definition.

        Returns
        -------
        dataset (xarray)
            Contains time step level data (stimuli, outcomes etc.).  See
            documentation on the schedule class for more details.
        """
        # determine experimental schedule to use
        if schedule is None:
            scd = self.schedules[list(self.schedules.keys())[0]]
        else:
            scd = self.schedules[schedule]
        
        # make list of time steps
        t_order = []
        trial_index = []
        m = 0 # index for trials
        for i in range(scd.n_stage):
            iti = scd.stage_list[i].iti
            order = scd.stage_list[i].order
            for j in range(scd.stage_list[i].n_rep):
                if scd.stage_list[i].order_fixed == False:
                    np.random.shuffle(order)
                for k in range(scd.stage_list[i].n_trial):
                    trial_def_bool = np.array( (scd.trial_def.stage == i) & (scd.trial_def.trial == order[k]) )
                    trial_def_index = list( scd.trial_def.t[trial_def_bool].values )
                    t_order += trial_def_index
                    trial_index += (iti + 1)*[m]
                    m += 1
        
        # make new trials object
        trials = scd.trial_def.loc[{'t' : t_order}]
        trials = trials.assign_coords({'t' : range(scd.n_t)})
        trials = trials.assign_coords({'trial' : trial_index})
        trials.attrs['schedule'] = scd.name

        return trials

    def read_csv(self, path, x_col, resp_col, resp_map, ident_col = None, conf_col = None, schedule = None, other_info = None, header = 'infer', n_final = 8):
        """
        Import empirical data from .csv files.

        Parameters
        ----------
        path: str
            Path to the .csv files.
        x_col: list
            Names of columns (strings) indicating cues (stimulus
            attributes, i.e. columns of 'x').
        resp_col: list
            Names of columns (strings) indicating responses.
        resp_map: dict
            Maps response names in the raw data to response names in the
            schedule definition.
        ident_col: str or None, optional
            If string, name of column indicating individual identifier
            (the 'ident' variable).  If None, then file names are used
            as 'ident'.  Defaults to None.
        conf_col: str or None, optional
            Name of the column indicating confidence responses (i.e.
            a measure of confidence following choices, typically
            obtained in the test stages of human classification tasks).
            Defaults to None (suitable for data without confidence responses).
        schedule: str, optional
            Name of the schedule from which to make trials.  By default
            selects the first schedule in the experiment object's
            definition.
        other_info: dict or None, optional
            Specifies other information (e.g. demographics) to be imported.
            Dictionary keys are variable names (e.g. 'sex', 'age'), while the
            values give the corresponding row index (e.g. a question such as 
            'What is your age?') and column name as a tuple.  Defaults to None
            (do not import any additional data).
        header: int or list of int, default ‘infer’
            Passed to pandas.read_csv.  Row number(s) to use as the column names,
            and the start of the data.
        n_final: int, optional
            Number of trials at end of each stage to use for calculating percent correct
            choices.  For example, set n_final = 10 to compute percent correct choices
            using the last 10 trials of each stage.
        
        Returns
        -------
        ds : dataset (xarray)
            Contains time step level data (stimuli, outcomes, behavior,
            possible outcomes etc.).
        summary : dataframe (pandas)
            Each row corresponds to a participant.  Contains proportion of
            correct responses in each non-test stage, plus OAT scores.

        Notes
        -----
        To avoid confusion, data from different schedules (e.g. different experimental
        groups) should be kept in separate directories.
        
        It is assumed that any numeric particpant identifiers ('ident') are
        integers rather than floats.
        
        The 'correct' variable encodes whether participant behavior ('b') matched
        the outcome ('u').  It is only really valid for category learning and similar
        experiments, and does not mean anything for stages without feedback (i.e. test stages).
        
        Current Limitations:
        For now, I assume that each time step represents a trial (i.e. iti = 0).
        I also assume that all 'x_names' in the Python schedule object are lower case.
        I also assume that each stage has at most one trial type for any set of punctate cues.
        I also assume that the Python schedule object has exactly the right number of trials.
        """        
        # list .csv files in the directory
        file_set = [file for file in glob.glob(path + "**/*.csv", recursive=True)]
        assert len(file_set) > 0, 'Cannot find any files in specified path.'
        
        # determine experimental schedule to use
        if schedule is None:
            scd = self.schedules[list(self.schedules.keys())[0]]
        else:
            scd = self.schedules[schedule]

        # set up pct_correct
        n_stage = len(scd.stage_list)
        pct_correct = dict()
        for i in range(n_stage):
            not_test = scd.stage_list[i].lrn == True
            if not_test:
                stage_name = scd.stage_list[i].name
                var_name = stage_name + '_' + 'last' + str(n_final) + '_pct_correct'
                pct_correct[var_name] = []
            
        # **** loop through files ****
        n_f = len(file_set)
        ds_list = []
        did_not_work_read = []
        did_not_work_ident = []
        did_not_work_b = []
        did_not_work_misc = []
        n_xc = len(x_col) # number of cue columns in raw data frame
        n_rc = len(resp_col) # number of response columns in raw data frame
        if conf_col is None: 
            usecols = x_col + resp_col # columns to import as the data frame 'raw'
        else:
            usecols = x_col + resp_col + [conf_col] # columns to import as the data frame 'raw'
        for i in range(n_f):
            # **** import raw data ****
            try:
                raw = pd.read_csv(file_set[i], error_bad_lines = False, warn_bad_lines = False, header = header, usecols = usecols)
                raw.dropna(subset = x_col, thresh = 1, inplace = True) # drop rows without recorded cues ('x')
                raw.dropna(subset = resp_col, thresh = 1, inplace = True) # drop rows without recorded responses
                raw_full = pd.read_csv(file_set[i], error_bad_lines = False, warn_bad_lines = False, header = header, na_filter = True) # copy of 'raw' whose rows won't be dropped (used for importing 'ident' and 'other info', e.g. demographics)
                index = np.zeros(raw.shape[0])
                # drop rows in which none of the response columns has one of the expected responses
                for col in resp_col:
                    index += raw[col].isin(list(resp_map.keys()))
                raw = raw.loc[np.array(index > 0)]
                n_r = raw.shape[0] # number of rows in raw data frame
                raw.index = range(n_r) # re-index 'raw'
                assert n_r == scd.n_t, 'wrong number of trials for file {}'.format(file_set[i]) + '\n' + 'trials found: ' + str(n_r) + '\n' + 'trials expected: ' + str(scd.n_t)
            except Exception as e:
                print(e)
                did_not_work_read += [file_set[i]]
            if not file_set[i] in did_not_work_read:
                # **** figure out 'ident' (participant ID) ****
                if ident_col is None:
                    ident = file_set[i].replace('.csv', '').replace(path + '/', '') # participant ID is file name
                else:
                    try:
                        ident_col_vals = np.array(raw_full[ident_col].values, dtype = 'str')
                        lengths = np.char.str_len(ident_col_vals)
                        ident = ident_col_vals[np.argmax(lengths)]
                        if not isinstance(ident, str): # change participant ID to string if it's not already a string
                            if ident.dtype == float:
                                ident = ident.astype(int)
                            ident = ident.astype(str)
                    except Exception as e:
                        print(e)
                        did_not_work_ident += [file_set[i]]    
            if not file_set[i] in (did_not_work_read + did_not_work_ident + did_not_work_misc):   
                try:
                    # **** determine b (response) from raw data ****
                    b = xr.DataArray(0, coords = [range(scd.n_t), scd.u_names], dims = ['t', 'u_name']) # observed responses
                    for m in range(scd.n_t):
                        for k in range(n_rc):
                            if pd.notna(raw.loc[m, resp_col[k]]):
                                raw_u_name = raw.loc[m, resp_col[k]].lower()
                                assert raw_u_name in resp_map.keys(), 'raw data response name "{}" is not found in "resp_map" (trial {})'.format(raw_u_name, m)
                                mapped_u_name = resp_map[raw_u_name]
                                b.loc[{'t' : m, 'u_name' : mapped_u_name}] = 1
                except Exception as e:
                    print(e)
                    did_not_work_b += [file_set[i]]
            if not file_set[i] in (did_not_work_read + did_not_work_ident + did_not_work_b + did_not_work_misc):     
                try:
                    # **** determine trial type from raw data ****
                    t_order = [] # list of time steps to produce the 'trials' data frame
                    trial_list = []
                    m = 0 # index for trials
                    for s in range(scd.n_stage):
                        iti = scd.stage_list[s].iti
                        n_stage_trials = scd.stage_list[s].n_trial * scd.stage_list[s].n_rep
                        for j in range(n_stage_trials):
                            # determine x (stimulus vector) from raw data
                            raw_x = pd.Series(0, index = scd.x_names)
                            for k in range(n_xc):
                                if pd.notna(raw.loc[m, x_col[k]]):
                                    raw_x_name = raw.loc[m, x_col[k]].lower()
                                    if raw_x_name in scd.x_names:
                                        raw_x[raw_x_name] = 1
                            # find corresponding trial definition (will only work if ITI = 0)
                            match_raw_x = (scd.trial_def['x'] == np.array(raw_x)).all(axis = 1)
                            match_stage = scd.trial_def['stage'] == s
                            trial_def_bool = match_stage & match_raw_x
                            trial_def_index = list(scd.trial_def['t'].loc[{'t' : trial_def_bool}])
                            if np.sum(trial_def_bool) == 0:
                                print('cue combination found that is not in schedule definition for stage:') # for debugging
                                print('stage')
                                print(s)
                                print('trial')
                                print(m)
                                print('cue combination')
                                print(raw_x)
                            # add to list of time steps indices, etc.
                            t_order += trial_def_index
                            trial_list += (iti + 1)*[m]
                            m += 1
                    # **** make new dataset **** 
                    ds_new = scd.trial_def.loc[{'t' : t_order}]
                    ds_new = ds_new.assign_coords({'t' : range(len(t_order)), 'trial' : range(len(t_order))})
                    ds_new = ds_new.assign(b = b)
                    ds_new = ds_new.expand_dims(ident = [ident])
                    # **** add confidence ratings ****
                    if not conf_col is None:
                        conf_val = np.array(raw[conf_col].values, dtype = 'float')
                        conf = xr.DataArray(conf_val, coords = [range(scd.n_t)], dims = ['t'])
                        ds_new = ds_new.assign(conf = conf)
                    # **** add other information (e.g. demographics) ****
                    if not other_info is None:
                        other_dict = dict()
                        for var_name in other_info:
                            row = raw_full[other_info[var_name][0]] == other_info[var_name][1]
                            column = other_info[var_name][2]
                            var = raw_full.loc[row, column].values[0]
                            other_dict[var_name] = (['ident'], np.array([var]))
                        ds_other = xr.Dataset(data_vars = other_dict, coords = {'ident': [ident]})
                        ds_new = ds_new.merge(ds_other)
                    # **** code each trial as correct (u matches b) or incorrect ****
                    u = ds_new['u'].squeeze()
                    b = ds_new['b'].squeeze()
                    correct = np.all(u == b, axis = 1)
                    ds_new = ds_new.assign(correct = correct)
                    # **** calculate percent correct per stage (excluding test stages) ****
                    for s in range(n_stage):
                        not_test = scd.stage_list[s].lrn == True
                        if not_test:
                            stage_name = scd.stage_list[s].name
                            index = np.array(ds_new.stage_name == stage_name)
                            var_name = stage_name + '_' + 'last' + str(n_final) + '_pct_correct'
                            pct_correct[var_name] += [100*ds_new['correct'].loc[{'t': index}][-n_final:].mean().values]    
                    # **** add dataset to list ****
                    ds_list += [ds_new]
                except Exception as e:
                    print(e)
                    did_not_work_misc += [file_set[i]]

        n_dnw_r = len(did_not_work_read)
        if n_dnw_r > 0:
            print('The following files could not be read by Pandas:')
            for i in range(n_dnw_r):
                print(did_not_work_read[i])
                
        n_dnw_i = len(did_not_work_ident)
        if n_dnw_i > 0:
            print('Participant ID (ident) could not be read from the following files:')
            for i in range(n_dnw_i):
                print(did_not_work_ident[i])
                
        n_dnw_b = len(did_not_work_b)
        if n_dnw_b > 0:
            print('Behavior (b) could not be read from the following files:')
            for i in range(n_dnw_b):
                print(did_not_work_b[i])
                
        n_dnw_m = len(did_not_work_misc)
        if n_dnw_m > 0:
            print('There was a problem importing the following files:')
            for i in range(n_dnw_m):
                print(did_not_work_misc[i])   

        # **** merge datasets together ****
        try:
            ds = xr.combine_nested(ds_list, concat_dim = 'ident')
            ds.attrs['schedule'] = scd.name
            ds.attrs['x_dims'] = scd.x_dims
        except Exception as e:
            print(e)
            print('There was a problem merging individual datasets together.')
            
        # **** create summary data frame (each row corresponds to a participant) ****
        summary = ds.drop_dims(['t', 'trial', 'x_name', 'u_name']).to_dataframe()
        # **** add pct_correct ****
        for s in range(n_stage):
            not_test = scd.stage_list[s].lrn == True
            if not_test:
                stage_name = scd.stage_list[s].name
                var_name = stage_name + '_' + 'last' + str(n_final) + '_pct_correct'
                summary[var_name] = pct_correct[var_name]            
        # **** calculate behavioral scores ****
        n_oats = len(self.oats)
        if conf_col is None:
            has_conf = False
        else:
            has_conf = True
        for oat in range(n_oats):
            oat_name = list(self.oats.keys())[oat]
            oat = self.oats[oat_name]
            if scd.name in oat.schedule_pos:
                summary[oat_name] = oat.behav_score_pos.compute_scores(ds, has_conf)
            else:
                if scd.name in oat.schedule_neg:
                    summary[oat_name] = oat.behav_score_neg.compute_scores(ds, has_conf)
        summary = summary.set_index(ds.ident.to_series(), drop = True)
        
        return (ds, summary)

class schedule:
    """
    Class to represent an experimental schedule, i.e. a scheme of
    related stimuli and outcomes given to a group of learners.

    Attributes
    ----------
    name: str
        The schedule's name.
    stage_list: list
        List of dictionaries defining the schedule's stages.
    trial_def: data frame
        Defines the trial types implied by 'stage_list'.
    x_names: list of str
        List specifying the names of cues (stimulus attributes).
    x_dims: dict or None
            If not None, then a dictionary specifying the cues belonging to
            each stimulus dimension.  Keys are dimension names and values
            are cue names (i.e. 'x_names').
    dim_names: list of str or None
        Names of stimulus dimensions.
    u_names: list of str
        List specifying the names of outcomes.
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
        Alternative coordinates for time steps (dimension t).
        Labels time steps as 'main' when at least one punctate cue is present
        and 'bg' ('background') otherwise (e.g. during the inter-trial
        interval).
    trial: int
        Alternative coordinates for time steps (dimension t).
        Each trial consists of one or more time steps.  This indicates
        which time steps correspond to each trial.  The ITI (inter-trial
        interval) is considered part of the trial that it precedes.
    trial_name: str
        Alternative coordinates for time steps (dimension t).
        Name of the trial type.  Has the form 'cues -> outcomes'.
    stage : int
        Alternative coordinates for time steps (dimension t).
        Indicates experimental stage by order.
    stage_name: str
        Alternative coordinates for time steps (dimension t).
        Indicates experimental stage by name.
    u_name: str
        Outcome/CS/response dimension.
    x_name: str
        Cue name dimension.
    """
    def __init__(self, name, stage_list, x_dims = None):
        """
        Parameters
        ----------
        name: str
            The name of the schedule.
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
        for i in range(n_stage):
            n_t_trial_def += (stage_list[i].iti + 1)*stage_list[i].n_trial_type
            n_t += (stage_list[i].iti + 1)*np.sum(stage_list[i].freq)*stage_list[i].n_rep
            stage_names += [stage_list[i].n_trial_type*stage_list[i].name]
            stage_number += [stage_list[i].n_trial_type*i]
            x_names += stage_list[i].x_bg
            u_names += stage_list[i].u_psb
            for j in range(stage_list[i].n_trial_type):
                x_new = stage_list[i].x_pn[j]
                u_new = stage_list[i].u[j]
                x_names += x_new
                u_names += u_new
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
        
        k = 0 # time step index
        for i in range(n_stage):
            iti = stage_list[i].iti
            stage += (iti + 1)*stage_list[i].n_trial_type*[i]
            stage_name += (iti + 1)*stage_list[i].n_trial_type*[stage_list[i].name]
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
                x.loc[{'row': range(k, k + iti + 1), 'x_name': stage_list[i].x_bg}] = 1.0
                u_psb.loc[{'row': range(k, k + iti + 1), 'u_name': stage_list[i].u_psb}] = 1.0
                if stage_list[i].lrn == True:
                    u_lrn.loc[{'row': range(k, k + iti + 1), 'u_name': stage_list[i].u_psb}] = 1.0
                has_main = has_x_pn or has_u # indicates whether there is a 'main' time step
                if has_main:
                    if iti > 0:
                        t_name += (iti - 1)*['bg']
                        t_name += ['pre_main']
                    t_name += ['main']
                    # set up 'main', i.e. time step with punctate cues/outcomes
                    if has_x_pn:
                        x.loc[{'row': k + iti, 'x_name': stage_list[i].x_pn[j]}] = 1.0
                    if has_u:
                        u.loc[{'row': k + iti, 'u_name': stage_list[i].u[j]}] = stage_list[i].u_value.loc[stage_list[i].u[j]]
                else:
                    t_name += (iti + 1)*['bg']
                # advance time step index
                k += iti + 1                

        # create dataset for trial type definitions ('trial_def')
        trial_def = xr.Dataset(data_vars = {'x': (['t', 'x_name'], x),
                                            'u': (['t', 'u_name'], u),
                                            'u_psb': (['t', 'u_name'], u_psb),
                                            'u_lrn': (['t', 'u_name'], u_lrn)},
                               coords = {'t': range(len(stage)),
                                         't_name': ('t', t_name),
                                         'trial': ('t', trial),
                                         'trial_name': ('t', trial_name),
                                         'stage': ('t', stage),
                                         'stage_name': ('t', stage_name),
                                         'x_name': x_names,
                                         'u_name': u_names,
                                         'schedule': name})                                      

        # make sure that no trial type is duplicated within any stage
        for i in range(n_stage):
            indexer = (trial_def.stage == i) & (trial_def.t_name == 'main')
            trial_names = trial_def.loc[{'t' : indexer}].trial_name
            all_unique = len(trial_names) == len(np.unique(trial_names))
            assert all_unique, 'Duplicate trial definition found in stage "{}" of schedule "{}".'.format(stage_list[i].name, name)

        # record information in new object ('self')
        self.name = name
        self.stage_list = stage_list
        self.trial_def = trial_def
        self.x_names = x_names
        self.u_names = u_names
        self.n_stage = n_stage        
        self.n_x = n_x
        self.n_u = n_u
        self.n_t = n_t
        
        # record stimulus dimension info, if any
        if not x_dims is None:
            self.x_dims = x_dims
            self.dim_names = list(x_dims.keys())
            self.n_dim = len(self.dim_names)
            trial_def.attrs = {'x_dims': self.x_dims}
        else:
            self.x_dims = None
            self.dim_names = None
            self.n_dim = None

class stage:
    """
    Class to represent experimental stages.
    
    Attributes
    ----------
    name : str
        name of the stage
    n_rep : int
        Number of repetitions (trial blocks).
    n_trial : int
        Number of trials per repetition (trial block).
    n_trial_type : int
        Number of trial types.
    order : list
        Used by the 'make_trials' method to determine trial
        order (typically after random reshuffling).
    x_pn : list
        Should have one list for each trial type of strings
        specifying the punctate cues, i.e. cues that
        accompany outcomes.
    x_bg : list
        Strings indicating background cues (present
        throughout stage).
        Can leave blank, e.g. for most human cat. learning
        tasks.
    freq : list
        Should have one list for each trial type of integers
        specifying the number of times each trial type is
        presented during each repetition (i.e. block).
    u : list
        Should have one list for each trial type of strings
        specifying the outcomes for each trial type.
    u_psb : list
        Strings specifying outcomes the learner considers
        possible during the stage.
    u_value : Series (Pandas) or None, optional
        Indicates the value or intensity of each outcome, e.g.
        varying amounts of money, shock or food.
    lrn : logical
        Indicates whether learning is possible during this
        stage.
        Typically 'True' except for test stages of human
        tasks.
    order_fixed : logical
        Indicates whether trial order should be fixed
        or randomized.
    iti : int
        The inter-trial interval, i.e. number of time steps
        between outcomes.  Should be 0 if the learner knows
        that outcomes cannot occur during the ITI, as in most
        human category learning experiments.
    """
    def __init__(self, name, n_rep, x_pn, x_bg = [], freq = None, u = None, u_psb = None, u_value = None, lrn = True, order_fixed = False, iti = 0):
        """
        Parameters
        ----------
        name : str
            name of the stage
        n_rep: int
            Number of repetitions (trial blocks).
        x_pn : list
            List of lists: should have one list for each trial type of strings
            specifying the punctate cues, i.e. cues that accompany outcomes.
            Use an empty list for trial types without any punctate cue.
        x_bg : list, optional
            Strings indicating background cues (present
            throughout stage).  Defaults to an empty list (no background
            cues), which is suitable for most human category learning
            tasks.
        freq : list or None, optional
            List of integers specifying the number of times each trial
            type is presented during each repetition (i.e. block).
            Defaults to None, which produces equal trial frequencies.
        u : list or None, optional
            List of lists: should have one list for each trial type of strings
            specifying the outcomes for each trial type.  Defaults to None,
            which means no US.
        u_psb : list or None, optional
            Strings specifying outcomes the learner considers
            possible during the stage.  Defaults to None, which
            means that all outcomes are considered possible.
        u_value : Series (Pandas) or None, optional
            Indicates the value or intensity of each outcome, e.g.
            varying amounts of money, shock or food.  Series values
            should be floats and the series index should be the names
            of all possible outcomes.  Defaults to None, which produces
            a Series where each outcome has a value of 1.0.
        lrn : logical, optional
            Indicates whether learning is possible during this
            stage.  Typically True except for test stages of human
            tasks.  Defaults to True.
        order_fixed : logical, optional
            Indicates whether trial order should be fixed
            or randomized.  Defaults to False.
        iti : int, optional
            The inter-trial interval, i.e. number of time steps
            between outcomes.  Should be 0 if the learner knows
            that outcomes cannot occur during the ITI, as in most
            human category learning experiments.  Defaults to 0.      
        """
        self.name = name
        self.n_rep = n_rep
        self.n_trial_type = len(x_pn) # number of trial types
        self.x_pn = x_pn
        self.x_bg = x_bg
        if freq is None:
            self.freq = self.n_trial_type*[1]
        else:
            self.freq = freq
        order = []
        for j in range(self.n_trial_type):
            order += self.freq[j]*[j]
        self.order = order
        self.n_trial = len(order)
        # set u
        if u is None:
            self.u = self.n_trial_type*[[]]
        else:
            self.u = u
        # set u_psb
        if (u_psb is None) and not (u is None):
            # automatically make all outcomes possible if u_psb is not specified
            u_names = []
            for trial_type in u:
                u_names += trial_type
            self.u_psb = list(np.unique(u_names))
        else:
            # set u_psb as specified by the user
            self.u_psb = list(u_psb)
        # set u_value
        n_u = len(self.u_psb)
        if (u_value is None) and not (u is None):
            # automatically make all outcomes have a value if 1.0 if u_value is not specified
            self.u_value = pd.Series(n_u*[1.0], index = self.u_psb)
        else:
            # set u_value as specified by the user
            self.u_value = u_value
        self.lrn = lrn
        self.order_fixed = order_fixed
        self.iti = iti
    
class oat:
    """
    Ordinal adequacy tests (OATs).

    Attributes
    ----------
    schedule_pos : list
        Schedules whose scores are counted as positive.
    behav_score_pos : behav_score object
        Behavioral score used for positive schedules.
    schedule_neg : list
        Schedules whose scores are counted as negative.
    behav_score_neg : behav_score object or None
        Behavioral score used for negative schedules.
        
    Methods
    -------
    compute_total(self, data_dict)
        Compute total OAT score (contrast between schedules, i.e. groups).
    conf_interval(self, data, conf_level = 0.95)
        Compute OAT score confidence interval.
    mean_resp(self, data)
        Compute means of the responses used in computing the OAT.
    """
    def __init__(self, schedule_pos, behav_score_pos, schedule_neg = None, behav_score_neg = None):
        """
        Parameters
        ----------
        schedule_pos : list
            Schedules whose scores are counted as positive.
        behav_score_pos : behav_score object
            Behavioral score used for positive schedules.
        schedule_neg : list or None, optional
            Schedules whose scores are counted as negative.
            Defaults to None.
        behav_score_neg : behav_score object or None, optional
            Behavioral score used for negative schedules.
            Defaults to None (if there are no negative schedules)
            or otherwise to the same behavioral score as
            for the positive schedules.
        """
        self.schedule_pos = schedule_pos
        self.behav_score_pos = behav_score_pos
        self.schedule_neg = schedule_neg
        if not schedule_neg is None:
            if behav_score_neg is None:
                self.behav_score_neg = behav_score_pos
            else:
                self.behav_score_neg = behav_score_neg
        else:
            self.behav_score_neg = None
        
    def compute_total(self, data):
        """
        Compute OAT score (contrast between schedules, i.e. groups).
        
        Parameters
        ----------
        data : dict
            Dictionary of behavioral data.  Each element is an xarray dataset from a different schedule.
            The keys are schedule names.
        
        Returns
        -------
        total : float
            Difference between mean behavioral scores of positive and negative schedules,
            or mean behavioral score (if there are no negative schedules).
        """
        # positive schedules
        pos_scores = np.array([])
        for s in self.schedule_pos:
            pos_scores = np.append(pos_scores, self.behav_score_pos.compute_scores(ds = data[s]))        
        pos_mean = np.mean(pos_scores)
        
        if not self.schedule_neg is None:
        # negative schedules
            neg_scores = np.array([])
            for s in self.schedule_neg:
                neg_scores = np.append(neg_scores, self.behav_score_pos.compute_scores(ds = data[s]))
            neg_mean = np.mean(neg_scores)
            total = pos_mean - neg_mean
        else:
            total = pos_mean
        return total
    
    def conf_interval(self, data, conf_level = 0.95):
        """
        Compute OAT score confidence interval.

        Parameters
        ----------
        data : dataset or dict
            Either an xarray dataset from a single experimental schedule, or a dictionary of such
            datasets (with keys that are schedule names).
        conf_level : float, optional
            Confidence level of the interval. Defaults to 0.95.
            
        Returns
        -------
        interval : dict
            lower : float
            center : float
            upper : float
            
        Notes
        -----
        Confidence intervals are constructed using Student's t distribution.
        
        If there are only positive schedules, we get a one sample confidence interval for the mean.
        
        If there are both positive and negative schedules, we get a confidence interval for the 
        mean difference (positive schedules - negative schedules).
        """            
        # Deal with case where input is a single dataset, rather than a dictionary of datasets.
        if type(data) is dict:
            data_dict = data
        else:
            data_dict = {self.schedule_pos[0] : data} # make a dict containing the input data
            
        # positive schedules
        pos_scores = np.array([])
        for s in self.schedule_pos:
            pos_scores = np.append(pos_scores, self.behav_score_pos.compute_scores(ds = data[s]))        
        pos_mean = np.mean(pos_scores)
        pos_var = np.var(pos_scores)
        pos_df = len(pos_scores) - 1
        alpha = 1 - conf_level
        if not self.schedule_neg is None:
            # two sample interval (mean difference)
            neg_scores = np.array([])
            for s in self.schedule_neg:
                neg_scores = np.append(neg_scores, self.behav_score_pos.compute_scores(ds = data[s]))
            neg_mean = np.mean(neg_scores)
            neg_var = np.var(neg_scores)
            neg_df = len(neg_scores) - 1
            pooled_var = (pos_var*pos_df + neg_var*neg_df)/(pos_df + neg_df)
            sem = np.sqrt(pooled_var)*np.sqrt(1/(pos_df + 1) + 1/(neg_df + 1))
            mu = pos_mean - neg_mean
        else:
            # one sample interval
            neg_df = 0
            sem = np.sqrt(pos_var/pos_df)
            mu = pos_mean
        t_crit = t.ppf(q = 1 - alpha/2, df = pos_df + neg_df, loc = 0, scale = 1)
        lower = mu - sem*t_crit
        upper = mu + sem*t_crit
        interval = {'lower' : lower, 'mean' : mu, 'upper' : upper}
        return interval
    
    def mean_resp(self, data):
        """
        Compute means of the behavior used in computing the OAT
        averaged across individuals and time steps.
        
        Parameters
        ----------
        data: dataframe or dict
            Either an xarray dataset from a single experimental schedule, or a dictionary of such
            datasets (with keys that are schedule names).
            
        Returns
        -------
        mean_resp: dataframe or dict of dataframe
            If the OAT only has positive schedules, then the dataframe containing relevant mean
            responses.  Otherwise a dict containing such dataframes for positive and
            negative schedules.
        """
        # Deal with case where input is a single dataset, rather than a dictionary of datasets.
        if type(data) is dict:
            data_dict = data
        else:
            data_dict = {self.schedule_pos[0] : data} # make a dict containing the input data
        
        # ** positive schedules **
        
        # relevant trial names (i.e. trial types)
        if self.behav_score_pos.trial_neg is None:
            trial_name = np.unique(self.behav_score_pos.trial_pos)
        else:
            trial_name = np.unique(self.behav_score_pos.trial_pos + self.behav_score_pos.trial_neg)
        # relevant response (outcome) names
        if self.behav_score_pos.resp_neg is None:
            u_name = np.unique(self.behav_score_pos.resp_pos)
        else:
            u_name = np.unique(self.behav_score_pos.resp_pos + self.behav_score_pos.resp_neg)
        # set up data array
        n_s = len(self.schedule_pos) # number of schedules
        n_tn = len(trial_name) # number of trial names (i.e. trial types)
        n_u = len(u_name) # number of outcomes
        da_pos = xr.DataArray(data = np.zeros((n_s, n_tn, n_u)),
                              dims = ['schedule', 'trial_name', 'u_name'],
                              coords = {'schedule': self.schedule_pos,
                                        'trial_name': trial_name,
                                        'u_name': u_name})
        # loop through schedules
        for s in self.schedule_pos:
            df_s = data_dict[s].to_dataframe()
            df_s.reset_index(inplace=True)
            for tn in trial_name:
                index_tn = np.array(df_s.trial_name == tn)
                index_sn = np.array(df_s.stage_name == self.behav_score_pos.stage)
                for un in u_name:
                    index_un = np.array(df_s.u_name == un)
                    index = index_tn*index_sn*index_un
                    mean_resp = df_s['b'].loc[index].mean()
                    da_pos.loc[{'schedule': s, 'trial_name': tn, 'u_name': un}] = mean_resp
            df_pos = da_pos.to_dataframe(name = 'mean_resp')
            df_pos.reset_index(inplace = True)
       
        # ** negative schedules **
        
        if not self.schedule_neg is None:
            # relevant trial names (i.e. trial types)
            if self.behav_score_neg.trial_neg is None:
                trial_name = self.behav_score_neg.trial_pos
            else:
                trial_name = self.behav_score_neg.trial_pos + self.behav_score_neg.trial_neg
            # relevant response (outcome) names
            if self.behav_score_neg.resp_neg is None:
                u_name = self.behav_score_neg.resp_pos
            else:
                u_name = self.behav_score_neg.resp_pos + self.behav_score_neg.resp_neg
            # set up data array
            n_s = len(self.schedule_neg) # number of positive schedules
            n_tn = len(trial_name) # number of trial names (i.e. trial types)
            n_u = len(u_name) # number of outcomes
            da_neg = xr.DataArray(data = np.zeros((n_s, n_tn, n_u)),
                                  dims = ['schedule', 'trial_name', 'u_name'],
                                  coords = {'schedule': self.schedule_neg,
                                            'trial_name': trial_name,
                                            'u_name': u_name})
            # loop through schedules
            for s in self.schedule_neg:
                ds_s = data_dict[s].loc[{'t': data_dict[s].stage_name == self.behav_score_neg.stage}]
                for tn in trial_name:
                    for un in u_name:
                        index = ds_s.trial_name == tn
                        mean_resp = ds_s['b'].loc[{'t': index, 'u_name': un}].mean()
                        da_neg.loc[{'schedule': s, 'trial_name': tn, 'u_name': un}] = mean_resp
            df_neg = da_neg.to_dataframe(name = 'mean_resp')
            df_neg.reset_index(inplace = True)
            # package data for output
            mean_resp = {'pos': df_pos, 'neg': df_neg}
        else:
            mean_resp = df_pos
            
        return mean_resp
        
class behav_score:
    """
    Behavioral scores.  These compute a difference (contrast) between an individual's responses for different trials and/or
    response options for use in OATs.
    
    Attributes
    ----------
    stage : str
        Name of the relevant stage.
    trial_pos : list
        Trials whose scores are counted as positive.   
    resp_pos : list
        Response counted as positive for each trial type.
    trial_neg : list of str or None
        Trials whose scores are counted as negative.
    resp_neg : list of str or None
        Responses counted as negative for each trial type.
        
    Methods
    -------
    compute_scores(self, ds)
        Compute behavioral score for each individual in the data set.
    
    """
    def __init__(self, stage, trial_pos, resp_pos, trial_neg = None, resp_neg = None):
        """
        Parameters
        ----------
        stage: str
            Name of the relevant stage.
        trial_pos: list of str
            Trials whose scores are counted as positive.   
        resp_pos: list of str
            Response counted as positive for each trial type.
            Should be the same length as trial_pos.
        trial_neg: list of str or None, optional
            Trials whose scores are counted as negative.
            Defaults to None.
        resp_neg: list of str or None, optional
            Response counted as negative for each trial type.
            Should be the same length as trial_neg.
            Defaults to None.
        """
        self.stage = stage
        self.trial_pos = trial_pos
        self.resp_pos = resp_pos
        self.trial_neg = trial_neg
        self.resp_neg = resp_neg
        
    def compute_scores(self, ds, use_conf = False):
        """
        Compute behavioral score for each individual in the data set.
        
        Parameters
        ----------
        ds: dataset
            Dataset (xarray) containing behavioral and other experimental data.
        oat: str, optional
            Name of the OAT to use.  By default selects the first OAT
            in the experiment object's definition.
        use_conf: boolean, optional
            If True then then responses will be multiplied by confidence when
            computing behavioral scores.  This will only work if 'conf' (confidence
            score) is a variable in the dataset (so far, simulation data produced
            by statsrat does NOT include confidence scores).  Defaults to False
            (behavioral scores are based choices only).
            
        Returns
        -------
        scores: array of float
            Array of individual behavioral scores.
        
        """            
        # loop through replications ('ident')
        scores = []
        for name in ds.ident:
            try:
                ds_name = ds.loc[{'ident' : name}] # dataset for current individual
                # positive trials and responses
                pos_sum = 0
                n_ttype = len(self.trial_pos)
                pos_n = 0
                for i in range(n_ttype):
                    pos_index = np.array(ds_name.stage_name == self.stage) & np.array(ds_name.trial_name == self.trial_pos[i])
                    b = ds_name['b'].loc[{'t' : pos_index, 'u_name' : self.resp_pos[i]}]
                    if use_conf:
                        conf = ds_name['conf'].loc[{'t' : pos_index}]
                        pos_sum += np.sum(conf*b)
                    else:
                        pos_sum += np.sum(b)
                    pos_n += pos_index.sum()
                pos_mean = pos_sum/pos_n

                # negative trials and responses
                if not self.trial_neg is None:
                    # negative trials and responses
                    neg_sum = 0
                    n_ttype = len(self.trial_neg)
                    neg_n = 0
                    for i in range(n_ttype):
                        neg_index = np.array(ds_name.stage_name == self.stage) & np.array(ds_name.trial_name == self.trial_neg[i])
                        b = ds_name['b'].loc[{'t' : neg_index, 'u_name' : self.resp_neg[i]}]
                        if use_conf:
                            conf = ds_name['conf'].loc[{'t' : neg_index}]
                            neg_sum += np.sum(conf*b)
                        else:
                            neg_sum += np.sum(b)
                        neg_n += neg_index.sum()
                    neg_mean = neg_sum/neg_n
                    scores += [pos_mean - neg_mean]
                else:
                    scores += [pos_mean]
            except:
                scores += ['nan']
        return np.array(scores, dtype = 'float')