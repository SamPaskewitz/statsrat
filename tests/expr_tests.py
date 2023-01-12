import unittest
import numpy as np
import pandas as pd
import xarray as xr
from statsrat import expr

class TestMakeTrials(unittest.TestCase):
    def test_make_trials(self):
        """
        Create an experiment object (Pavlovian blocking) and test that it makes trials correctly.
        """
        # make stage objects
        one_cue_stage = expr.stage(x_pn = [['cs1']],
                                   x_bg = ['ctx'],
                                   y = [['us']],
                                   y_psb = ['us'],
                                   order_fixed = False, 
                                   iti = 1,
                                   n_rep = 2)

        two_cue_stage = expr.stage(x_pn = [['cs1', 'cs2']],
                                   x_bg = ['ctx'],
                                   y = [['us']],
                                   y_psb = ['us'],
                                   order_fixed = False, 
                                   iti = 1,
                                   n_rep = 2)

        test_cs2_stage = expr.stage(x_pn = [['cs2']],
                                    x_bg = ['ctx'],
                                    y_psb = ['us'],
                                    order_fixed = False,
                                    iti = 1,
                                    n_rep = 2)

        cs2_score = expr.behav_score(stage = 'test',
                                     trial_pos = ['cs2 -> nothing'],
                                     resp_pos = ['us'])

        # make schedule objects
        blocking = expr.schedule(resp_type = 'exct', stages = {'one_cue': one_cue_stage, 'two_cue': two_cue_stage, 'test': test_cs2_stage})
        
        two_cue = expr.schedule(resp_type = 'exct', stages = {'two_cue': two_cue_stage, 'test': test_cs2_stage})

        # make experiment object
        blocking = expr.experiment(schedules = {'control': two_cue, 'experimental': blocking},
                                   oats = {'blocking': expr.oat(schedule_pos = ['control'],
                                                                schedule_neg = ['experimental'],
                                                                behav_score_pos = cs2_score,
                                                                behav_score_neg = cs2_score)})
        # make trials
        trials = blocking.make_trials(schedule = 'experimental')
        
        # define data in the correct result, for comparison
        comparison = xr.Dataset(data_vars = {'x': (['t', 'x_name'], np.array([[0., 0., 1.],
                                                                             [1., 0., 1.],
                                                                             [0., 0., 1.],
                                                                             [1., 0., 1.],
                                                                             [0., 0., 1.],
                                                                             [1., 1., 1.],
                                                                             [0., 0., 1.],
                                                                             [1., 1., 1.],
                                                                             [0., 0., 1.],
                                                                             [0., 1., 1.],
                                                                             [0., 0., 1.],
                                                                             [0., 1., 1.]])),
                                     'y': (['t', 'y_name'], np.array(4*[0, 1] + 4*[0]).reshape((12, 1))),
                                     'y_psb': (['t', 'y_name'], np.ones((12, 1))),
                                     'y_lrn': (['t', 'y_name'], np.ones((12, 1)))},
                        coords = {'t': range(12),
                                  't_name': ('t', 6*['pre_main', 'main']),
                                  'ex': ('t', 2*['ctx', 'ctx.cs1'] + 2*['ctx', 'ctx.cs1.cs2'] + 2*['ctx', 'ctx.cs2']),
                                  'trial': ('t', [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]),
                                  'trial_name': ('t', 4*['cs1 -> us'] + 4*['cs1.cs2 -> us'] + 4*['cs2 -> nothing']),
                                  'stage': ('t', 4*[0] + 4*[1] + 4*[2]),
                                  'stage_name': ('t', 4*['one_cue'] + 4*['two_cue'] + 4*['test']),
                                  'x_name': ['cs1', 'cs2', 'ctx'],
                                  'y_name': ['us'],
                                  'time': ('t', range(12))},
                       attrs = {'x_ex': pd.DataFrame({'cs1': [0.0, 0.0, 1.0, 1.0], 'cs2': [0.0, 1.0, 0.0, 1.0], 'ctx': [1.0, 1.0, 1.0, 1.0]},
                                                      index = ['ctx', 'ctx.cs2', 'ctx.cs1', 'ctx.cs1.cs2']),
                                'ex_names': ['ctx', 'ctx.cs2', 'ctx.cs1', 'ctx.cs1.cs2'],
                                'resp_type': 'exct',
                                'schedule': 'experimental'})
        
        # test that the trials created by the experiment object are equal to the comparison ones (does not test whether attrs are equal, as that's a bit more of a pain)
        # I'm not sure if this is the right way to use the assertion statements, but the built in assertEqual method doesn't handle xarray datasets.
        self.assertTrue(trials.equals(comparison))

if __name__ == '__main__':
    unittest.main()
