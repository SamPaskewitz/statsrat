import unittest
import xarray as xr
from statsrat.rw import predef as rw_predef
from statsrat.exemplar import predef as exemplar_predef

# import trials
trials = xr.open_dataset('data/trials_for_simulation_tests.nc')

class TestRW(unittest.TestCase):
    """
    Test selected Rescorla-Wagner family models by simulating them on a pre-defined trial sequence
    (from the inverse base rate effect category learning task) and comparing the simulated output to saved output.
    """
    def test_basic_rw(self):
        # create simulation data
        sim_data = rw_predef.basic.simulate(trials)
        
        # import comparison simulation data
        comparison = xr.open_dataset('data/sim data for comparison/rw/basic.nc')
        
        # test that they are equal
        self.assertTrue(sim_data.equals(comparison))
        
class TestExemplar(unittest.TestCase):
    """
    Test selected exemplar models by simulating them on a pre-defined trial sequence
    (from the inverse base rate effect category learning task) and comparing the simulated output to saved output.
    """
    def test_basic_exemplar(self):
        # create simulation data
        sim_data = exemplar_predef.basic.simulate(trials)
        
        # import comparison simulation data
        comparison = xr.open_dataset('data/sim data for comparison/exemplar/basic.nc')
        
        # test that they are equal
        self.assertTrue(sim_data.equals(comparison))
        
        