import numpy as np

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
    t_name_pos: str
        Time step name used for trials counted as positive.
        Should usually be 'main'.
    trial_neg : list of str or None
        Trials whose scores are counted as negative.
    resp_neg : list of str or None
        Responses counted as negative for each trial type.
    t_name_neg: str
        Time step name used for trials counted as negative.
        Should usually be 'main'.
        
    Methods
    -------
    compute_scores(self, ds)
        Compute behavioral score for each individual in the data set.
        
    Notes
    -----
    The behavioral score is defined as the mean of positive responses on positive trials
    during the positive time step, minus the mean of negative responses on negative trials
    during the negative time step.
    
    Each trial consists of one or more time steps, which have names.  The 'main' time step
    is the one featuring the predictor stimuli/cues and/or outcome.  If there is an
    inter-trial interval (ITI), then the time step preceding 'main' is labeled 'pre_main' and
    the others are labeled 'background'.  Typically, behavioral scores will always only use the
    'main' time step, which is why that is set as the default for the 't_name_pos' and 't_name_neg'
    attributes.  However, in some experiments (e.g. Pavlovian conditioning) we may wish to contrast
    responses during cue/CS presentation with baseline response levels, in order to determine how much
    response control the CS has.  In that case we would set t_name_pos = 'main' and t_name_neg = 
    'pre_main' to obtain CS response - baseline response.  This is a similar idea to the supression
    ratios traditionally computed in Pavlovian experiments, but uses subtraction rather than a ratio.
    """
    def __init__(self, stage, trial_pos, resp_pos, t_name_pos = 'main', trial_neg = None, resp_neg = None, t_name_neg = 'main'):
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
        t_name_pos: str, optional
            Time step name used for trials counted as positive.
            Should usually be 'main' (the default).
        trial_neg: list of str or None, optional
            Trials whose scores are counted as negative.
            Defaults to None.
        resp_neg: list of str or None, optional
            Response counted as negative for each trial type.
            Should be the same length as trial_neg.
            Defaults to None.
        t_name_neg: str, optional
            Time step name used for trials counted as negative.
            Should usually be 'main' (the default).
        """
        self.stage = stage
        self.trial_pos = trial_pos
        self.resp_pos = resp_pos
        self.t_name_pos = t_name_pos
        self.trial_neg = trial_neg
        self.resp_neg = resp_neg
        self.t_name_neg = t_name_neg
        
    def compute_scores(self, ds, use_conf = False):
        """
        Compute behavioral score for each individual in the data set.
        
        Parameters
        ----------
        ds: dataset
            Dataset (xarray) containing behavioral and other experimental data.
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
            #try:
                ds_name = ds.loc[{'ident' : name}] # dataset for current individual
                
                # positive trials and responses
                pos_sum = 0
                n_ttype = len(self.trial_pos)
                pos_n = 0
                for i in range(n_ttype):
                    pos_index = np.array(ds_name.stage_name == self.stage) & np.array(ds_name.trial_name == self.trial_pos[i]) & np.array(ds_name.t_name == self.t_name_pos)
                    b = ds_name['b'].loc[{'t' : pos_index, 'y_name' : self.resp_pos[i]}]
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
                        neg_index = np.array(ds_name.stage_name == self.stage) & np.array(ds_name.trial_name == self.trial_neg[i]) & np.array(ds_name.t_name == self.t_name_neg)
                        b = ds_name['b'].loc[{'t' : neg_index, 'y_name' : self.resp_neg[i]}]
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
                    #print(scores)
            #except:
                #scores += ['nan']
        return np.array(scores, dtype = 'float')