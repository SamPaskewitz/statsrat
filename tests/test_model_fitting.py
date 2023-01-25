import unittest
from xarray import open_dataset
from pandas import read_csv
from pandas.testing import assert_frame_equal
from statsrat import fit_indv
from statsrat.rw.predef import CompAct

# import trial by trial data to test model fits
trial_data = open_dataset('data/trial data for model fits.nc')

class TestModelFit(unittest.TestCase):
    """
    Test model fitting functions.
    """
    def test_fit_indv(self):
        # fit the model
        fit = fit_indv(CompAct, trial_data, global_maxeval = 500, local_maxeval = 250)
        
        # import comparison simulation data UPDATE
        comparison = read_csv('data/model fit results/fit_indv.csv', index_col = 'ident')
        
        # test that they are equal
        assert_frame_equal(fit.drop(['global_time_used', 'local_time_used'], axis = 1),
                           comparison.drop(['global_time_used', 'local_time_used'], axis = 1),
                           check_column_type = False,
                           check_exact = False,
                           check_dtype = False)