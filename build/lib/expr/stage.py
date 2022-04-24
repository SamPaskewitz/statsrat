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
        Number of trial types (not including intro or outro).
    n_t : int
        Total number of time steps in stage.
    order : list
        Trial order (if order_fixed == True) or else simply
        something used by the 'make_trials' method to determine
        trial order after random reshuffling.
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
    y : list
        Should have one list for each trial type of strings
        specifying the outcomes for each trial type.
    y_psb : list
        Strings specifying outcomes the learner considers
        possible during the stage.
    y_value : series (Pandas) or None, optional
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
    intro_length : int
        The number of time steps with just background stimuli
        before trials begin.  Defaults to 0.
    outro_length : int
        The number of time steps with just background stimuli
        after trials are finished.  Defaults to 0.
    """
    def __init__(self, n_rep, x_pn, x_bg = [], x_value = None, freq = None, y = None, y_psb = None, y_value = None, lrn = True, order = None, order_fixed = False, iti = 0, intro_length = 0, outro_length = 0):
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
            Defaults to None, which produces equal trial frequencies
            (if order is None) or else is determined by the order argument.
        y : list or None, optional
            List of lists: should have one list for each trial type of strings
            specifying the outcomes for each trial type.  Defaults to None,
            which means no US.
        y_psb : list or None, optional
            Strings specifying outcomes the learner considers
            possible during the stage.  Defaults to None, which
            means that all outcomes are considered possible.
        y_value : Series (Pandas) or None, optional
            Indicates the value or intensity of each outcome, e.g.
            varying amounts of money, shock or food.  Series values
            should be floats and the series index should be the names
            of all possible outcomes.  Defaults to None, which produces
            a Series where each outcome has a value of 1.0.
        lrn : logical, optional
            Indicates whether learning is possible during this
            stage.  Typically True except for test stages of human
            tasks.  Defaults to True.
        order : list or None, optional
            If a list, then this specifies trial order.
            If None (default) then trial order is simply
            determined by the freq argument.
        order_fixed : logical, optional
            Indicates whether trial order should be fixed
            or randomized.  Defaults to False.
        iti : int, optional
            The inter-trial interval, i.e. number of time steps
            between outcomes.  Should be 0 if the learner knows
            that outcomes cannot occur during the ITI, as in most
            human category learning experiments.  Defaults to 0.
        intro_length : int, optional
            The number of time steps with just background stimuli
            before trials begin.  Defaults to 0.
        outro_length : int, optional
            The number of time steps with just background stimuli
            after trials are finished.  Defaults to 0.
            
        Notes
        -----
        There are several valid ways to specify the freq, order and order_fixed arguments:
        
        1) Omit all arguments (all have their defaults): each trial type
        is given once per repetition (block) in the order that they are defined.
        
        2) Specify freq, order_fixed = True: each trial type is presented freq[i]
        times in a row.
        
        3) Specify freq, order_fixed = False: trials are presented in random order
        with frequency given by the freq argument.
        
        4) Specify order_fixed = False: trials are presented in random order
        with once each per repetition (block).
        
        5) Specify order (freq = None, order_fixed = False i.e. the defaults): trials
        are presented in the order specified by the order argument.
        
        It is not intended to specify these arguments in any other combination.  Thus,
        the freq and order_fixed arguments will be ignored if the order argument is specified
        (i.e. not None, the default): the freq attribute will be filled out based on order,
        and order_fixed will be set to True.
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
        if order is None:
            if freq is None:
                self.freq = self.n_trial_type*[1]
            else:
                self.freq = freq
            self.order = []
            for j in range(self.n_trial_type):
                self.order += self.freq[j]*[j]
            self.order_fixed = order_fixed
        else:
            self.order = order
            freq_tally = np.zeros(self.n_trial_type)
            for i in range(len(order)):
                freq_tally[order[i]] += 1
            self.freq = list(freq_tally)
            self.order_fixed = True
        self.n_t = intro_length + n_rep*np.sum(self.freq*(1 + iti)) + outro_length # number of time steps
        self.n_trial = len(self.order)
        # set y
        if y is None:
            self.y = self.n_trial_type*[[]]
        else:
            self.y = y
        # set y_psb
        if (y_psb is None) and not (y is None):
            # automatically make all outcomes possible if y_psb is not specified
            y_names = []
            for trial_type in y:
                y_names += trial_type
            self.y_psb = list(np.unique(y_names))
        else:
            # set y_psb as specified by the user
            self.y_psb = list(y_psb)
        # set y_value
        n_y = len(self.y_psb)
        if (y_value is None) and not (y is None):
            # automatically make all outcomes have a value if 1.0 if y_value is not specified
            self.y_value = pd.Series(n_y*[1.0], index = self.y_psb)
        else:
            # set y_value as specified by the user
            self.y_value = y_value
        self.lrn = lrn
        self.iti = iti
        self.intro_length = intro_length
        self.outro_length = outro_length