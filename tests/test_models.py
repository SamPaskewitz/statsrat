import unittest
import xarray as xr
from statsrat.rw import predef as rw_predef
from statsrat.rw_plus_minus import predef as rw_plus_minus_predef
from statsrat.exemplar import predef as exemplar_predef
from statsrat.bayes_regr import predef as bayes_regr_predef
from statsrat.latent_cause import predef as latent_cause_predef

# import trials
trials = xr.open_dataset('data/trials_for_simulation_tests.nc')

class TestRW(unittest.TestCase):
    """
    Test selected Rescorla-Wagner family models by simulating them on a pre-defined trial sequence
    (from the inverse base rate effect category learning task) and comparing the simulated output to saved output.
    """
    def test_basic(self):
        # create simulation data
        sim_data = rw_predef.basic.simulate(trials)
        
        # import comparison simulation data
        comparison = xr.open_dataset('data/sim data for comparison/rw/basic.nc')
        
        # test that they are equal
        self.assertTrue(sim_data.equals(comparison))
        
    def test_power(self):
        # create simulation data
        sim_data = rw_predef.power.simulate(trials)
        
        # import comparison simulation data
        comparison = xr.open_dataset('data/sim data for comparison/rw/power.nc')
        
        # test that they are equal
        self.assertTrue(sim_data.equals(comparison))
        
    def test_drva(self):
        # create simulation data
        sim_data = rw_predef.drva.simulate(trials)
        
        # import comparison simulation data
        comparison = xr.open_dataset('data/sim data for comparison/rw/drva.nc')
        
        # test that they are equal
        self.assertTrue(sim_data.equals(comparison))
        
    def test_smpr(self):
        # create simulation data
        sim_data = rw_predef.smpr.simulate(trials)
        
        # import comparison simulation data
        comparison = xr.open_dataset('data/sim data for comparison/rw/smpr.nc')
        
        # test that they are equal
        self.assertTrue(sim_data.equals(comparison))
        
    def test_decay(self):
        # create simulation data
        sim_data = rw_predef.decay.simulate(trials)
        
        # import comparison simulation data
        comparison = xr.open_dataset('data/sim data for comparison/rw/decay.nc')
        
        # test that they are equal
        self.assertTrue(sim_data.equals(comparison))
        
    def test_CompAct_cfg2_intercept(self):
        # create simulation data
        sim_data = rw_predef.CompAct_cfg2_intercept.simulate(trials)
        
        # import comparison simulation data
        comparison = xr.open_dataset('data/sim data for comparison/rw/CompAct_cfg2_intercept.nc')
        
        # test that they are equal
        self.assertTrue(sim_data.equals(comparison))
        
    def test_Kalman(self):
        # create simulation data
        sim_data = rw_predef.Kalman.simulate(trials)
        
        # import comparison simulation data
        comparison = xr.open_dataset('data/sim data for comparison/rw/Kalman.nc')
        
        # test that they are equal
        self.assertTrue(sim_data.equals(comparison))

class TestRWPlusMinus(unittest.TestCase):
    """
    Test selected Rescorla-Wagner +/- family models by simulating them on a pre-defined trial sequence
    (from the inverse base rate effect category learning task) and comparing the simulated output to saved output.
    """
    def test_decay_plus_minus(self):
        # create simulation data
        sim_data = rw_plus_minus_predef.decay_plus_minus.simulate(trials)
        
        # import comparison simulation data
        comparison = xr.open_dataset('data/sim data for comparison/rw_plus_minus/decay_plus_minus.nc')
        
        # test that they are equal
        self.assertTrue(sim_data.equals(comparison))
        
    def test_decay_only_minus(self):
        # create simulation data
        sim_data = rw_plus_minus_predef.decay_only_minus.simulate(trials)
        
        # import comparison simulation data
        comparison = xr.open_dataset('data/sim data for comparison/rw_plus_minus/decay_only_minus.nc')
        
        # test that they are equal
        self.assertTrue(sim_data.equals(comparison))
        
class TestExemplar(unittest.TestCase):
    """
    Test selected exemplar models by simulating them on a pre-defined trial sequence
    (from the inverse base rate effect category learning task) and comparing the simulated output to saved output.
    """
    def test_basic(self):
        # create simulation data
        sim_data = exemplar_predef.basic.simulate(trials)
        
        # import comparison simulation data
        comparison = xr.open_dataset('data/sim data for comparison/exemplar/basic.nc')
        
        # test that they are equal
        self.assertTrue(sim_data.equals(comparison))
        
class TestBayesRegr(unittest.TestCase):
    """
    Test selected Bayesian regression models by simulating them on a pre-defined trial sequence
    (from the inverse base rate effect category learning task) and comparing the simulated output to saved output.
    """
    def test_linear_constant(self):
        # create simulation data
        sim_data = bayes_regr_predef.linear_constant.simulate(trials)
        
        # import comparison simulation data
        comparison = xr.open_dataset('data/sim data for comparison/bayes_regr/linear_constant.nc')
        
        # test that they are equal
        self.assertTrue(sim_data.equals(comparison))
        
    def test_probit_constant(self):
        # create simulation data
        sim_data = bayes_regr_predef.probit_constant.simulate(trials)
        
        # import comparison simulation data
        comparison = xr.open_dataset('data/sim data for comparison/bayes_regr/probit_constant.nc')
        
        # test that they are equal
        self.assertTrue(sim_data.equals(comparison))
        
    def test_multinomial_probit_constant(self):
        # create simulation data
        sim_data = bayes_regr_predef.multinomial_probit_constant.simulate(trials)
        
        # import comparison simulation data
        comparison = xr.open_dataset('data/sim data for comparison/bayes_regr/multinomial_probit_constant.nc')
        
        # test that they are equal
        self.assertTrue(sim_data.equals(comparison))      
        
    def test_linear_ard(self):
        # create simulation data
        sim_data = bayes_regr_predef.linear_ard.simulate(trials)
        
        # import comparison simulation data
        comparison = xr.open_dataset('data/sim data for comparison/bayes_regr/linear_ard.nc')
        
        # test that they are equal
        self.assertTrue(sim_data.equals(comparison))
        
    def test_linear_ard_drv_atn(self):
        # create simulation data
        sim_data = bayes_regr_predef.linear_ard_drv_atn.simulate(trials)
        
        # import comparison simulation data
        comparison = xr.open_dataset('data/sim data for comparison/bayes_regr/linear_ard_drv_atn.nc')
        
        # test that they are equal
        self.assertTrue(sim_data.equals(comparison))
        
class TestLatentCause(unittest.TestCase):
    """
    Test selected latent cause models by simulating them on a pre-defined trial sequence
    (from the inverse base rate effect category learning task) and comparing the simulated output to saved output.
    """
    def test_constant(self):
        # create simulation data
        sim_data = latent_cause_predef.constant.simulate(trials)

        # import comparison simulation data
        comparison = xr.open_dataset('data/sim data for comparison/latent_cause/constant.nc')

        # test that they are equal
        self.assertTrue(sim_data.equals(comparison))

    def test_exponential(self):
        # create simulation data
        sim_data = latent_cause_predef.exponential.simulate(trials)

        # import comparison simulation data
        comparison = xr.open_dataset('data/sim data for comparison/latent_cause/exponential.nc')

        # test that they are equal
        self.assertTrue(sim_data.equals(comparison))

    def test_power(self):
        # create simulation data
        sim_data = latent_cause_predef.power.simulate(trials)

        # import comparison simulation data
        comparison = xr.open_dataset('data/sim data for comparison/latent_cause/power.nc')

        # test that they are equal
        self.assertTrue(sim_data.equals(comparison))

    def test_power_asymptote(self):
        # create simulation data
        sim_data = latent_cause_predef.power_asymptote.simulate(trials)

        # import comparison simulation data
        comparison = xr.open_dataset('data/sim data for comparison/latent_cause/power_asymptote.nc')

        # test that they are equal
        self.assertTrue(sim_data.equals(comparison))

    def test_power_clusters(self):
        # create simulation data
        sim_data = latent_cause_predef.power_clusters.simulate(trials)

        # import comparison simulation data
        comparison = xr.open_dataset('data/sim data for comparison/latent_cause/power_clusters.nc')

        # test that they are equal
        self.assertTrue(sim_data.equals(comparison))

    def test_refractory_period(self):
        # create simulation data
        sim_data = latent_cause_predef.refractory_period.simulate(trials)

        # import comparison simulation data
        comparison = xr.open_dataset('data/sim data for comparison/latent_cause/refractory_period.nc')

        # test that they are equal
        self.assertTrue(sim_data.equals(comparison))