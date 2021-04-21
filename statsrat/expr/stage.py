import numpy as np
import pandas as pd

class stage:
    """
    Class to represent experimental stages.
    
    Attributes
    ----------
    name : str or None
        By default this is None until the stage is used
        in the creation of an experiment object.
    n_rep : int
        Number of repetitions (trial blocks).
    n_trial : int
        Number of trials per repetition (trial block).
    n_trial_type : int
        Number of trial types.
    n_t : int
        Total number of time steps in stage.
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
    x_value : series (Pandas)
        Indicates numerical values that cues take when present in the stage,
        e.g. different values of height.
    x_names: list of str
        Names of cues (stimulus attributes).
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
    u_value : series (Pandas) or None, optional
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
    def __init__(self, n_rep, x_pn, x_bg = [], x_value = None, freq = None, u = None, u_psb = None, u_value = None, lrn = True, order_fixed = False, iti = 0):
        """
        Parameters
        ----------
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
        x_value : series (Pandas) or None, optional
            Indicates numerical values that cues take when present in the stage,
            e.g. different values of height.  If None (default), then
            cues are indicated by 1.0 when present.
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
        self.name = None # stages only get a real name attribute when used to create a schedule
        self.n_rep = n_rep
        self.n_trial_type = len(x_pn) # number of trial types
        self.x_pn = x_pn
        self.x_bg = x_bg
        x_names = []
        for xi in x_pn:
            x_names += xi
        x_names += x_bg
        self.x_names = list(np.unique(x_names))
        if x_value is None:
            self.x_value = pd.Series(1.0, index = self.x_names)
        else:
            self.x_value = x_value
        if freq is None:
            self.freq = self.n_trial_type*[1]
        else:
            self.freq = freq
        self.n_t = n_rep*np.sum(self.freq*(1 + iti)) # number of time steps
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