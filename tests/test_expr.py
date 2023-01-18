import unittest
import numpy as np
import pandas as pd
import xarray as xr
from experiments_for_tests import blocking, learned_pred

# https://realpython.com/python-testing
# python -m unittest discover

class TestExperiment(unittest.TestCase):
    def test_make_trials(self):
        """
        Test that an experiment object (simple Pavlovian blocking) makes trials correctly.
        """
        # create trials
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
        # (I'm not sure if this is the right way to use the assertion statements, but the built in assertEqual method doesn't handle xarray datasets)
        self.assertTrue(trials.equals(comparison))
        
    def test_read_csv(self):
        """
        Test that a category learning experiment object can import csv data.
        """
        # read .csv data
        data = learned_pred.read_csv(path = 'simulated data',
                                     x_col = ['left_cue', 'right_cue'],
                                     resp_col = ['response', 'test_response'],
                                     resp_map = {'i': 'cat1', 'ii': 'cat2'})[0]
        
        # read the correct result, for comparison
        comparison = xr.open_dataset('comparison data/imported_trial_data.nc')
        
        # test that they match
        # (I'm not sure if this is the right way to use the assertion statements, but the built in assertEqual method doesn't handle xarray datasets)
        # this does not test whether attrs are the same
        self.assertTrue(data.equals(comparison))

if __name__ == '__main__':
    unittest.main()
