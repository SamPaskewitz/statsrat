import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import t
from statsrat.expr.behav_score import behav_score

class oat:
    """
    Ordinal adequacy test (OAT).

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
    
    Class Methods (alternative constructors for different types of OAT)
    -------------------------------------------------------------------
    define_response_contrast(cls, stage, trials, resp_pos, resp_neg, schedule = 'design')
        Alternative constructor to conveniently define a within subjects (single schedule)
        OAT which contrasts responses across a single set of trials.
        
    define_trial_contrast(cls, stage, trial_pos, trial_neg, resp, schedule = 'design')
        Alternative constructor to conveniently define a within subjects (single schedule)
        OAT which contrasts a single response across two sets of trials.
        
    define_between_subjects_single_response(cls, schedule_pos, schedule_neg, stage, trials, single_resp = 'us'):
        Alternative constructor to conveniently define a between subjects
        OAT which contrasts different schedules across a single set of trials
        with a single response.
    
    Methods
    -------
    compute_total(self, data_dict)
        Compute total OAT score (contrast between schedules, i.e. groups).
    conf_interval(self, data, conf_level = 0.95)
        Compute OAT score confidence interval.
    mean_resp(self, data)
        Compute means of the responses used in computing the OAT.
        
        
    Notes
    -----
    This class has multiple alternative constructors to conveniently create different types of OAT.
    https://realpython.com/python-multiple-constructors/#instantiating-classes-in-python
    ** EXPLAIN **
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
    
    @classmethod
    def define_response_contrast(cls, stage, trials, resp_pos, resp_neg, schedule = 'design'):
        """
        Alternative constructor to conveniently define a within subjects (single schedule)
        OAT which contrasts responses across a single set of trials.
        
        Parameters
        ----------
        stage: str
            Stage name.
        trials: list of str
            Names of trial types used.
        resp_pos: list of str
            List of responses counted as positive.
        resp_neg: list of str
            List of responses counted as negative.
        schedule: str. optional
            Name of the schedule used.  Defaults to
            'design', which is typically used when there is only
            one schedule in an experiment.
        """
        bscore = behav_score(stage = stage, trial_pos = trials, resp_pos = resp_pos, trial_neg = trials, resp_neg = resp_neg)
        return cls(schedule_pos = [schedule], behav_score_pos = bscore)
    
    @classmethod
    def define_trial_contrast(cls, stage, trial_pos, trial_neg, single_resp = 'us', schedule = 'design'):
        """
        Alternative constructor to conveniently define a within subjects (single schedule)
        OAT which contrasts a single response across two sets of trials.
        
        Parameters
        ----------
        stage: str
            Name of the stage used.
        trial_pos: list of str
            Names of trial types counted as positive.
        trial_neg: list of str
            Names of trial types used.
        single_resp: str, optional
            Name of the response used.  Defaults to 'us'
            for convenient use in Pavlovian conditioning experiments.
        schedule: str. optional
            Name of the schedule used.  Defaults to
            'design', which is typically used when there is only
            one schedule in an experiment.
        """
        bscore = behav_score(stage = stage, trial_pos = trial_pos, trial_neg = trial_neg, resp_pos = [single_resp])
        return cls(schedule_pos = [schedule], behav_score_pos = bscore)
        
    
    @classmethod
    def define_between_subjects_single_response(cls, schedule_pos, schedule_neg, stage, trials, single_resp = 'us'):
        """
        Alternative constructor to conveniently define a between subjects
        OAT which contrasts different schedules across a single set of trials
        with a single response.
        
        Parameters
        ----------
        schedule_pos: str
            Name of the schedule counted as positive.
        schedule_neg: str
            Name of the schedule counted as negative.
        stage: str
            Name of the stage used.
        trials: list of str
            Names of trial types used.
        single_resp: str, optional
            Name of the response used.  Defaults to 'us'
            for convenient use in Pavlovian conditioning experiments.
        """
        bscore = behav_score(stage = stage, trial_pos = trials, resp_pos = [single_resp])
        return cls(schedule_pos = schedule_pos, schedule_neg = schedule_neg, behav_score_pos = bscore, behav_score_neg = bscore)
    
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
                neg_scores = np.append(neg_scores, self.behav_score_neg.compute_scores(ds = data[s]))
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
                neg_scores = np.append(neg_scores, self.behav_score_neg.compute_scores(ds = data[s]))
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
        mean_resp: dataframe
            Mean responses for relevant trial types.
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
            y_name = np.unique(self.behav_score_pos.resp_pos)
        else:
            y_name = np.unique(self.behav_score_pos.resp_pos + self.behav_score_pos.resp_neg)
        # set up data array
        n_s = len(self.schedule_pos) # number of schedules
        n_tn = len(trial_name) # number of trial names (i.e. trial types)
        n_y = len(y_name) # number of outcomes
        da_pos = xr.DataArray(data = np.zeros((n_s, n_tn, n_y)),
                              dims = ['schedule', 'trial_name', 'y_name'],
                              coords = {'schedule': self.schedule_pos,
                                        'trial_name': trial_name,
                                        'y_name': y_name})
        
        # loop through schedules
        for s in self.schedule_pos:
            df_s = data_dict[s].to_dataframe()
            df_s.reset_index(inplace=True)
            index_is_main = np.array(df_s.t_name == 'main')
            for tn in trial_name:
                index_tn = np.array(df_s.trial_name == tn)
                index_sn = np.array(df_s.stage_name == self.behav_score_pos.stage)
                for yn in y_name:
                    index_yn = np.array(df_s.y_name == yn)
                    index = index_tn*index_sn*index_yn*index_is_main
                    mean_resp = df_s['b'].loc[index].mean()
                    da_pos.loc[{'schedule': s, 'trial_name': tn, 'y_name': yn}] = mean_resp
            df_pos = da_pos.to_dataframe(name = 'mean_resp')
            df_pos.reset_index(inplace = True)
        
        # ** negative schedules **
        
        if not self.schedule_neg is None:
            # relevant trial names (i.e. trial types)
            if self.behav_score_neg.trial_neg is None:
                trial_name = np.unique(self.behav_score_neg.trial_pos)
            else:
                trial_name = np.unique(self.behav_score_neg.trial_pos + self.behav_score_neg.trial_neg)
            # relevant response (outcome) names
            if self.behav_score_neg.resp_neg is None:
                y_name = np.unique(self.behav_score_neg.resp_pos)
            else:
                y_name = np.unique(self.behav_score_neg.resp_pos + self.behav_score_neg.resp_neg)
            # set up data array
            n_s = len(self.schedule_neg) # number of positive schedules
            n_tn = len(trial_name) # number of trial names (i.e. trial types)
            n_y = len(y_name) # number of outcomes
            da_neg = xr.DataArray(data = np.zeros((n_s, n_tn, n_y)),
                                  dims = ['schedule', 'trial_name', 'y_name'],
                                  coords = {'schedule': self.schedule_neg,
                                            'trial_name': trial_name,
                                            'y_name': y_name})
            # loop through schedules
            for s in self.schedule_neg:
                df_s = data_dict[s].to_dataframe()
                df_s.reset_index(inplace=True)
                index_is_main = np.array(df_s.t_name == 'main')
                for tn in trial_name:
                    index_tn = np.array(df_s.trial_name == tn)
                    index_sn = np.array(df_s.stage_name == self.behav_score_neg.stage)
                    for yn in y_name:
                        index_yn = np.array(df_s.y_name == yn)
                        index = index_tn*index_sn*index_yn*index_is_main
                        mean_resp = df_s['b'].loc[index].mean()
                        da_neg.loc[{'schedule': s, 'trial_name': tn, 'y_name': yn}] = mean_resp
            df_neg = da_neg.to_dataframe(name = 'mean_resp')
            df_neg.reset_index(inplace = True)
            # package data for output
            mean_resp = pd.concat([df_pos, df_neg])
        else:
            mean_resp = df_pos
            
        return mean_resp