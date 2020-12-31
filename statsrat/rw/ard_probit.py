import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats
from scipy.special import softmax
from statsrat.rw import fbase
from statsrat import resp_fun
from numpy.linalg import cond
from scipy.linalg import solve, inv

class model:
    '''
    Class for Bayesian regression models with shrinkage
    (automatic relevance detection).  These don't quite fit
    into the Rescorla-Wagner (rw) family, but are very similar.
    This version uses a version of probit regression (assuming outcomes
    are independent), and hence is suitable for category learning tasks etc.

    Attributes
    ----------
    name : str
        Model name.
    fbase : function
        Base mapping between cues (x) and features (f_x).
    par_names : list
        Names of the model's free parameters (strings).

    Methods
    -------
    simulate(trials, resp_type = 'choice', par_val = None)
        Simulate a trial sequence once with known model parameters.
                
    Notes
    -----
    See the following:
    https://rpubs.com/cakapourani/variational-bayes-bpr
    '''

    def __init__(self, name, fbase):
        """
        Parameters
        ----------
        name : str
            Model name.
        fbase : function
            Base mapping between cues (x) and features (f_x).
        """
        # add data to object ('self')
        self.name = name
        self.fbase = fbase
        self.pars = pars.loc[['prior_tausq_inv_hpar0', 'prior_tausq_inv_hpar1', 'resp_scale']]
 
    def simulate(self, trials, resp_type = 'choice', par_val = None, random_resp = False, ident = 'sim'):
        """
        Simulate a trial sequence once with known model
        parameters.
        
        Parameters
        ----------
        trials : data frame
            Time step level experimental data (cues, outcomes etc.).

        resp_type : str, optional
            Type of behavioral response: one of 'choice', 'exct' or 'supr'.
            Defaults to 'choice'.

        par_val : list, optional
            Learning model parameters (floats or ints).

        random_resp : str, optional
            Whether or not simulated responses should be random.  Defaults
            to false, in which case behavior (b) is identical to expected
            behavior (b_hat); this saves some computation time.  If true
            and resp_type is 'choice', then discrete responses are selected
            using b_hat as choice probabilities.  If true and resp_type is
            'exct' or 'supr' then a small amount of normally distributed
            noise (sd = 0.01) is added to b_hat.

        ident : str, optional
            Individual participant identifier.  Defaults to 'sim'.

        Returns
        -------
        ds : dataset

        Notes
        -----
        Use the response type 'choice' for discrete response options.  This
        produces response probabilities using a softmax function:
        .. math:: \text{resp}_i = \frac{ e^{\phi \hat{u}_i} }{ \sum_j e^{\phi \hat{u}_j} }

        The response type 'exct' is used for excitatory Pavlovian
        conditioning:
        .. math:: \text{resp} = \frac{ e^{\phi \hat{u}_i} }{ e^{\phi \hat{u}_i} + 1 }

        The response type 'supr' (suppression) is used for inhibitory
        Pavlovian conditioning:
        .. math:: \text{resp} = \frac{ e^{-\phi \hat{u}_i} }{ e^{-\phi \hat{u}_i} + 1 }

        Here :math:`\phi` represents the 'resp_scale' parameter.
        """
        # use default parameters unless others are given
        if par_val is None:
            sim_pars = self.pars['default']
        else:
            # check that parameter values are within acceptable limits; if so assemble into a pandas series
            # for some reason, the optimization functions go slightly outside the specified bounds
            abv_min = par_val >= self.pars['min'] - 0.0001
            blw_max = par_val <= self.pars['max'] + 0.0001
            all_ok = np.prod(abv_min & blw_max)
            assert all_ok, 'par_val outside acceptable limits'
            sim_pars = pd.Series(par_val, self.pars.index)

        # various arrays etc.
        x = np.array(trials['x'], dtype = 'float64')
        u = np.array(trials['u'], dtype = 'float64')
        u_psb = np.array(trials['u_psb'], dtype = 'float64')
        u_lrn = np.array(trials['u_lrn'], dtype = 'float64')
        x_names = list(trials.x_name.values)
        u_names = list(trials.u_name.values)
        (f_x, f_names) = self.fbase(x, x_names).values() # features and feature names
        n_t = f_x.shape[0] # number of time points
        n_f = f_x.shape[1] # number of features
        n_u = u.shape[1] # number of outcomes/response options
        u_hat = np.zeros((n_t, n_u)) # outcome predictions
        z_hat = np.zeros((n_t, n_u)) # predicted latent variable (z) before observing outcome (u)
        b_hat = np.zeros((n_t, n_u)) # expected behavior
        shrink_cond = np.zeros((n_t, n_u)) # condition number of a matrix relevant to shrinkage
        # array for sufficient statistics need to estimate w
        sufstat0_w = np.zeros((n_t + 1, n_f, n_u)) # sufficient statistic for w
        sufstat1_w = np.zeros((n_t + 1, n_f, n_f, n_u)) # sufficient statistic for w
        # arrays for variational means and variances
        mean_tausq_inv = np.zeros((n_t, n_f)) # variational mean of 1/tau^2
        initial_mean_tausq_inv = (sim_pars['prior_tausq_inv_hpar1'] + 1)/(-sim_pars['prior_tausq_inv_hpar0'])
        mean_tausq_inv[0, :] = initial_mean_tausq_inv
        mean_w = np.zeros((n_t, n_f, n_u)) # variational mean of w (a.k.a. mu)
        var_w = np.zeros((n_t, n_f, n_u)) # variational variance of w (i.e. the diagonal elements of the covariance matrix)
        var_w[0, :, :] = 1/initial_mean_tausq_inv
        mean_wsq = np.zeros((n_t, n_f, n_u)) # variational mean of w^2
        mean_wsq[0, :, :] = 1/initial_mean_tausq_inv
        mean_z = np.zeros((n_t, n_u)) # variational mean of the latent variable z (after observing u)
        # arrays for variational distribution hyperparameters
        hpar0_tausq_inv = np.array((n_t + 1)*n_f*[sim_pars['prior_tausq_inv_hpar0']], dtype='float').reshape((n_t + 1, n_f)) # hyperparameter for the variational distribution of 1/tausq
        hpar1_tausq_inv = np.array((n_t + 1)*n_f*[sim_pars['prior_tausq_inv_hpar1']], dtype='float').reshape((n_t + 1, n_f)) # hyperparameter for the variational distribution of 1/tausq
        hpar0_w = np.zeros((n_t + 1, n_f, n_u)) # hyperparameter for the variational distribution of w (precision * mean)
        hpar1_w = np.zeros((n_t + 1, n_f, n_f, n_u)) # hyperparameter for the variational distribution of w (precision)
        for j in range(n_u):
            hpar1_w[0, :, :, j] = np.diag(n_f*[initial_mean_tausq_inv])

        # set up response function (depends on response type)
        resp_dict = {'choice': resp_fun.choice,
                     'exct': resp_fun.exct,
                     'supr': resp_fun.supr}
        sim_resp_fun = resp_dict[resp_type]
        
        # loop through time steps
        u_psb_so_far = np.zeros(n_u)
        for t in range(n_t):
            for j in range(n_u):
                if u_psb[t, j] == 1:
                    u_psb_so_far[j] = 1
            # compute hpar_tausq and related quantities using mean_w
            hpar0_tausq_inv[t, :] = sim_pars['prior_tausq_inv_hpar0'] - 0.5*mean_wsq[t, :, :].sum(1)
            hpar1_tausq_inv[t, :] = sim_pars['prior_tausq_inv_hpar1'] + 0.5*u_psb_so_far.sum()
            mean_tausq_inv[t, :] = (hpar1_tausq_inv[t, :] + 1)/(-hpar0_tausq_inv[t, :])
            T = np.diag(mean_tausq_inv[t, :]) # mean prior precision matrix (same for all outcomes)
            # compute hpar_w and related quantities using mean_tausq_inv
            for j in range(n_u):
                calculate = u_psb[t, j] == 1
                if calculate:
                    # compute hpar_w
                    hpar0_w[t:n_t, :, j] = sufstat0_w[t, :, j]
                    hpar1_w[t:n_t, :, :, j] = (T + sufstat1_w[t, :, :, j]) # precision matrix
                    # compute mean_w
                    shrink_cond[t, j] = cond(hpar1_w[t, :, :, j]) # condition number
                    mean_w[t:n_t, :, j] = solve(hpar1_w[t, :, :, j], hpar0_w[t, :, j], assume_a = 'pos')
                    # compute var_w and mean_wsq
                    var_w[t:n_t, :, j] = 1/np.diag(hpar1_w[t, :, :, j])
                    mean_wsq[t:n_t, :, j] = var_w[t, :, j] + mean_w[t, :, j]**2
            # predict u (outcome) and compute b (behavior)
            z_hat[t, :] = u_psb[t, :]*(f_x[t, :]@mean_w[t, :, :]) # predicted value of latent variable (z)
            pred_sd = np.sqrt(1 + np.inner(f_x[t, :]@inv(hpar1_w[t, :, :, j]), f_x[t, :])) # predictive SD for z
            u_hat[t, :] = u_psb[t, :]*stats.norm.cdf(z_hat[t, :]/pred_sd)
            b_hat[t, :] = sim_resp_fun(u_hat[t, :], u_psb[t, :], sim_pars['resp_scale']) # response
            # compute mean_z (the variational mean of the latent variable z after observing u)
            # I've double checked these calculations.
            # From https://rpubs.com/cakapourani/variational-bayes-bpr (section 4.4)
            # See also https://en.wikipedia.org/wiki/Truncated_normal_distribution#One_sided_truncation_(of_lower_tail)[4]
            for j in range(n_u):
                calculate = u_psb[t, j] == 1
                phi = stats.norm.pdf(-z_hat[t, j])
                PHI = stats.norm.cdf(-z_hat[t, j])
                if calculate:
                    if u[t, j] == 1:
                        mean_z[t, j] = z_hat[t, j] + phi/(1 - PHI)
                    else:
                        mean_z[t, j] = z_hat[t, j] - phi/PHI
            # update sufficient statistics of x and u for estimating w
            f = f_x[t, :].squeeze() # for convenience
            for j in range(n_u):
                update = u_lrn[t, j] == 1
                if update:
                    sufstat0_w[(t+1):n_t, :, j] = sufstat0_w[t, :, j] + f*mean_z[t, j] # LOOK HERE FOR ABSENCE OF IBRE
                    # In the linear version, f*u[t, j] = 0 whenever outcome j does not occur.
                    # In this version, f*mean_z[t, j] < 0 whenever outcome j does not occur.
                    # This appears to be the only significant difference between the two versions
                    # when it comes to producing the IBRE.
                    # I should test whether the linear version produces the IBRE when non-occurrence 
                    # is coded as -1 instead of as 0.
                    # Surprisingly, the linear version still produces the IBRE when non-occurence is coded as 1.
                    # Confirming this, if I change the probit version's code to use a consistent 1/-1 coding
                    # instead of mean_z as the feedback signal, it produces the IBRE.
                    
                    # TRY RUNNING A BATCH VERSION OF THIS ALGORITHM
                    # A batch version of the probit algorithm still fails to produce the IBRE.
                    # It does illuminate an important point: the feedback signal is larger on
                    # PC + I -> C trials than on PR + I -> R trials.
                    
                    # It seems that (I haven't checked across all parameter settings) in the linear version,
                    # the final inhibitory PC1 -> R1 association is roughly equal in strength to the 
                    # inhibitory PR1 -> C1 association.  In the probit version, the former is stronger than the
                    # latter.
                    
                    # In the linear version (again, I've only checked some parameter vectors)
                    # tausq_inv increases across learning and is lowest for PR1/PR2.
                    # In the probit version, tausq_inv decreases across learning (if prior_hpar_tausq1_inv is high,
                    # otherwise it increases) and is lowest for PC1/PC2 (although PR1/PR2 may be lower
                    # early in training).  Of course, the linear version produces the IBRE even
                    # without attention learning, so this difference in attention (mean_tausq_inv)
                    # must be an effect rather than a cause.
                    
                    # Let's think in terms of prediction error (delta).  In the linear version
                    # on PR1 + I1 trials early in training, delta is negative for C1 (which does not occur)
                    # and positive for R1 (which does occur).  This will not be different for the probit
                    # version.  I don't currently see how this sort of analysis will lead to understanding why the linear
                    # version produces the IBRE and the probit one does not.
                    sufstat1_w[(t+1):n_t, :, :, j] = sufstat1_w[t, :, :, j] + np.outer(f, f)

        # generate simulated responses
        if random_resp is False:
            b = b_hat
        else:
            rng = np.random.default_rng()
            if resp_type == 'choice':
                b = np.zeros((n_t, n_u))
                for t in range(n_t):
                    choice = rng.choice(n_u, p = b_hat[t, :])
                    b[t, choice] = 1
            else:
                b = b_hat + stats.norm.rvs(loc = 0, scale = 0.01, size = (n_t, n_u))
        
        # put all simulation data into a single xarray dataset
        ds = xr.Dataset(data_vars = {'x' : (['t', 'x_name'], x),
                                     'u' : (['t', 'u_name'], u),
                                     'u_psb' : (['t', 'u_name'], u_psb),
                                     'u_lrn' : (['t', 'u_name'], u_lrn),
                                     'f_x' : (['t', 'f_name'], f_x),
                                     'u_hat' : (['t', 'u_name'], u_hat),
                                     'b_hat' : (['t', 'u_name'], b_hat),
                                     'b' : (['t', 'u_name'], b),
                                     'shrink_cond' : (['t', 'u_name'], shrink_cond),
                                     'mean_tausq_inv' : (['t', 'f_name'], mean_tausq_inv),
                                     'mean_w' : (['t', 'f_name', 'u_name'], mean_w),
                                     'var_w' : (['t', 'f_name', 'u_name'], var_w),
                                     'mean_z' : (['t', 'u_name'], mean_z),
                                     'mean_wsq' : (['t', 'f_name', 'u_name'], mean_wsq[range(n_t), :, :]),
                                     'hpar0_tausq_inv' : (['t', 'f_name'], hpar0_tausq_inv[range(n_t), :]),
                                     'hpar1_tausq_inv' : (['t', 'f_name'], hpar1_tausq_inv[range(n_t), :])},
                        coords = {'t' : range(n_t),
                                  't_name' : ('t', trials.t_name),
                                  'trial' : ('t', trials.trial),
                                  'trial_name' : ('t', trials.trial_name),
                                  'stage' : ('t', trials.stage),
                                  'stage_name' : ('t', trials.stage_name),
                                  'x_name' : x_names,
                                  'f_name' : f_names,
                                  'u_name' : u_names,
                                  'ident' : [ident]},
                        attrs = {'model': self.name,
                                 'model_class' : 'shrinkage',
                                 'schedule' : trials.attrs['schedule'],
                                 'resp_type' : resp_type,
                                 'sim_pars' : sim_pars})
        return ds

########## PARAMETERS ##########    
par_names = ['prior_tausq_inv_hpar0']; par_list = [{'min' : -20.0, 'max' : 0.0, 'default' : -2.0}] # hyperparameter for tausq_inv
par_names += ['prior_tausq_inv_hpar1']; par_list += [{'min' : -1.0, 'max' : 19.0, 'default' : 3.0}] # other hyperparameter for tausq_inv
par_names += ['resp_scale']; par_list += [{'min': 0.0, 'max': 10.0, 'default': 1.0}]
pars = pd.DataFrame(par_list, index = par_names)
del par_names; del par_list