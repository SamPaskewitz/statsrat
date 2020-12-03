import numpy as np
import pandas as pd
import xarray as xr
import glob
from scipy.stats import t
# TO DO:
# make OATs

class experiment:
    '''
    A class for (non-learning) lottery experiments of the type
    used in behavioral economics research.
    
    Attributes
    ----------
    trial_type_list : list
        A list of the experiment's trial types.  Each element of the list
        is a list of prospect objects representing that trial's available choices.

    Methods
    -------
    make_trials(self)
        Create a time step level dataset for the whole experiment.
    
    FINISH
    
    Notes
    -----
    Rewards are assumed to be monetary, or in some other similar numeric format.
    '''
    def __init__(self, trial_type_list):
        """
        Parameters
        ----------
        trial_type_list : list
            A list of the experiment's trial types.  Each element of the list
            is a list of prospect objects representing that trial's available choices.
        """
        
        # loop through trial types to add information
        (rwd, prob, ambg) = 3*[np.zeros((n_t, n_p, n_o))]
        for t in trial_type_list:
            
            for p in t['prospect_list']:
                
                for o in n_o:
                    
            # FINISH
        
        # create dataset for trial type definitions ('trial_def')
        trial_def = xr.Dataset(data_vars = {'rwd' : (['t', 'p_name', 'o_name'], rwd),
                                            'prob' : (['t', 'p_name', 'o_name'], prob),
                                            'ambg' : (['t', 'p_name', 'o_name'], ambg)},
                               coords = {'t' : range(len(stage)),
                                         't_name' : ('t', t_names),
                                         'prospect' : p_names,
                                         'outcome' : o_names})
        
        # record information in new object ('self')
        self.trial_type_list = trial_type_list
        self.trial_def = trial_def
        self.t_names = t_names
        self.p_names = p_names
        self.o_names = o_names
        self.n_tt = n_tt
        self.n_p = n_p
        self.n_o = n_o
        
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
            Contains time step level data (probabilities, rewards etc. for various prospects).  See
            documentation on the schedule class for more details.
        """
        # FINISH
        
class prospect:
    """
    Attributes
    ----------
    n_o : int
        Number of possible outcomes.
    rwd: list
        Reward (nominal reward, not utility) of each outcome.
    prob : list
        Outcome probabilities.  Must sum to 1.
    ambg : float
        Level of ambiguity.  Defaults to 0.
    """
    def __init__(self, rwd, prob, ambg = 0):
        """
        Parameters
        ----------
        rwd: list
            Reward (nominal reward, not utility) of each outcome.
        prob : list
            Outcome probabilities.  Must sum to 1.
        ambg : float, optional
            Level of ambiguity.  Defaults to 0.
        """
        # check that the arguments given are valid
        assert len(rwds) == len(probs), 'probs and rwds must be of equal length'
        assert np.sum(probs) == 1, 'probs must sum to 1'
        # set up the prospect ('self')
        self.n_o = len(rwds)
        self.rwds = rwds
        self.probs = probs
        if delays is None:
            self.delays = self.n_o*[0]
        else:
            self.delays = delays
        self.ambiguous = ambiguous