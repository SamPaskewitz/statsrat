import numpy as np
import pandas as pd
import xarray as xr
import glob
from statsrat.expr.schedule import schedule
from statsrat.expr.oat import oat
from copy import deepcopy

class experiment:
    """
    A class used to represent learning experiments.

    Attributes
    ----------
    resp_type : str
        The type of behavioral response made by the learner.  Must be the same for
        all schedules in the experiment.  Can be either 'choice' (discrete responses),
        'exct' (excitatory) or 'supr' (suppression of an ongoing activity).
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
    
    See 'predef.pvl_iti' for Pavlovian conditioning examples.
    """
    def __init__(self, schedules, oats = None, notes = None):
        """
        Parameters
        ----------
        schedules : dict
            A dictionary of the experiment's schedules (sequences of stimuli and feedback etc
            that typically correspond to groups in the experimental design).
        oats : dict or None, optional
            A dictionary of the experiment's ordinal adequacy tests (OATs), or
            else None (experiment has no OATs).  Defaults to None.
        notes : str or None, optional
            Notes on the experiment (e.g. explanation of design, references).
            Defaults to None (i.e. no notes).
        """
        # check that everything in the 'schedules' argument is a schedule object
        is_scd = []
        for s in schedules.values():
            is_scd += [isinstance(s, schedule)]
        assert not (False in is_scd), 'Non-schedule object input as schedule.'
        # check that everything in the 'oat' argument is an oat object
        if not oats is None:
            if len(oats) > 0:
                is_oat = []
                for o in oats.values():
                    is_oat += [isinstance(o, oat)]
                assert not (False in is_oat), 'Non-oat object input as oat.'
        # check that that all schedules have the same response type
        self.resp_type = schedules[list(schedules.keys())[0]].resp_type
        if len(schedules) > 1:
            match_resp_type = []
            for s in schedules.values():
                match_resp_type += [self.resp_type == s.resp_type]
            assert not (False in match_resp_type), 'Schedules have non-matching response types (resp_type).'
        # add other data to 'self'
        self.schedules = deepcopy(schedules)
        for s in self.schedules:
            self.schedules[s].name = s # assign schedule name attributes based on dictionary keys
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
            
        Notes
        -----
        Adds in 'time', an alternative coordinate for time steps (dimension t).
        This indicates real world time (in abstract units), including possible delays
        since previous time steps (e.g. for an experiment with several sessions
        on different days).  Starts at 0 for the first time step, and each time
        step represents a time unit of 1.
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
        for st in scd.stages:
            iti = scd.stages[st].iti
            order = scd.stages[st].order
            for j in range(scd.stages[st].n_rep):
                if scd.stages[st].order_fixed == False:
                    np.random.shuffle(order)
                for k in range(scd.stages[st].n_trial):
                    trial_def_bool = np.array( (scd.trial_def.stage_name == st) & (scd.trial_def.trial == order[k]) )
                    trial_def_index = list( scd.trial_def.t[trial_def_bool].values )
                    t_order += trial_def_index
                    trial_index += (iti + 1)*[m]
                    m += 1
                    
        # make list for 'time' coordinate
        st_names = list(scd.stages.keys())
        time = list(np.arange(scd.stages[st_names[0]].n_t))
        for i in range(1, scd.n_stage):
            time += list(np.arange(scd.stages[st_names[i]].n_t) + scd.delays[i - 1] + time[-1] + 1)
        
        # make new trials object
        trials = scd.trial_def.loc[{'t' : t_order}]
        trials = trials.assign_coords({'t' : range(scd.n_t)})
        trials = trials.assign_coords({'trial' : ('t', trial_index)})
        trials = trials.assign_coords({'time' : ('t', time)})
        trials = trials.assign_attrs({'schedule': scd.name})

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
        
        Currently, the 'time' (real world time) coordinate is only a copy of 't' (the time step
        number).  This represents the assumption that there are no delays between stages of the
        experiment.
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
        n_stage = len(scd.stages)
        pct_correct = dict()
        for st in scd.stages:
            not_test = scd.stages[st].lrn == True
            if not_test:
                var_name = st + '_' + 'last' + str(n_final) + '_pct_correct'
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
                    for st in scd.stages:
                        iti = scd.stages[st].iti
                        n_stage_trials = scd.stages[st].n_trial * scd.stages[st].n_rep
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
                            match_stage = scd.trial_def['stage_name'] == st
                            trial_def_bool = match_stage & match_raw_x
                            trial_def_index = list(scd.trial_def['t'].loc[{'t' : trial_def_bool}])
                            if np.sum(trial_def_bool) == 0:
                                print('cue combination found that is not in schedule definition for stage:') # for debugging
                                print('stage')
                                print(st)
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
                    n_t = len(t_order)
                    ds_new = ds_new.assign_coords({'t' : range(n_t), 'trial' : ('t', range(len(t_order))), 'time': ('t', range(n_t))})
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
                    for st in scd.stages:
                        not_test = scd.stages[st].lrn == True
                        if not_test:
                            stage_name = scd.stages[st].name
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
            ds = xr.combine_nested(ds_list, concat_dim = 'ident', combine_attrs = 'override')
        except Exception as e:
            print(e)
            print('There was a problem merging individual datasets together.')
            
        # **** create summary data frame (each row corresponds to a participant) ****
        summary = ds.drop_dims(['t', 'x_name', 'u_name']).to_dataframe()
        # **** add pct_correct ****
        for st in scd.stages:
            not_test = scd.stages[st].lrn == True
            if not_test:
                stage_name = scd.stages[st].name
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