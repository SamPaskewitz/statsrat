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
                is_main = np.array(ds_name.t_name == 'main')
                # positive trials and responses
                pos_sum = 0
                n_ttype = len(self.trial_pos)
                pos_n = 0
                for i in range(n_ttype):
                    pos_index = np.array(ds_name.stage_name == self.stage) & np.array(ds_name.trial_name == self.trial_pos[i]) & is_main
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
                        neg_index = np.array(ds_name.stage_name == self.stage) & np.array(ds_name.trial_name == self.trial_neg[i]) & is_main
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