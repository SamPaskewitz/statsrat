import unittest
import xarray as xr
from experiment_for_testing_data_import import learned_pred
from statsrat.expr.predef.pvl_iti import blocking

# https://realpython.com/python-testing
# python -m unittest discover

class TestExperiment(unittest.TestCase):
    def test_make_trials(self):
        """
        Test that an experiment object (simple Pavlovian blocking) makes trials correctly.
        """
        # create trials
        trials = blocking.make_trials('experimental')
        
        # read the correct result, for comparison
        comparison = xr.open_dataset('data/blocking_trials.nc')
        
        # test that the trials created by the experiment object are equal to the comparison ones (does not test whether attrs are equal)
        # (I'm not sure if this is the right way to use the assertion statements, but the built in assertEqual method doesn't handle xarray datasets)
        self.assertTrue(trials.equals(comparison))
        
    def test_read_csv(self):
        """
        Test that a category learning experiment object can import csv data.
        """
        # read .csv data
        data = learned_pred.read_csv(path = 'data/sim data for import test',
                                     x_col = ['left_cue', 'right_cue'],
                                     resp_col = ['response', 'test_response'],
                                     resp_map = {'i': 'cat1', 'ii': 'cat2'})[0]
        
        # read the correct result, for comparison
        comparison = xr.open_dataset('data/imported_trial_data.nc')
        
        # test that they match
        # (I'm not sure if this is the right way to use the assertion statements, but the built in assertEqual method doesn't handle xarray datasets)
        # this only tests the element 0 (the trials dataset), and does not test whether attrs are the same
        self.assertTrue(data.equals(comparison))

if __name__ == '__main__':
    unittest.main()
