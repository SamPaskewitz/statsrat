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
    Class for Kalman filter models with shrinkage.
    These don't quite fit into the Rescorla-Wagner
    (rw) family, but are very similar.
    FINISH UPDATING.

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
        self.pars = pars.loc[['alpha0', 'beta0', 'u_var', 'drift_var', 'lrate_tausq', 'resp_scale']]
 
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

        # set stuff up
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
        b_hat = np.zeros((n_t, n_u)) # expected behavior
        delta = np.zeros((n_t, n_u))
        alpha = np.zeros((n_t + 1, n_f))
        beta = np.zeros((n_t + 1, n_f))
        alpha[0, :] = sim_pars['alpha0']
        beta[0, :] = sim_pars['beta0']
        tausq0 = sim_pars['beta0']/(sim_pars['alpha0'] - 1)
        tausq = np.zeros((n_t + 1, n_f))
        tausq[0, :] = tausq0
        shrink_cond = np.zeros((n_t, n_u)) # condition number of a matrix relevant to shrinkage
        mu = np.zeros((n_t + 1, n_f, n_u))
        Sigma = np.zeros((n_t + 1, n_u, n_f, n_f)) # weight covariance matrices
        Lambda = np.zeros((n_t + 1, n_u, n_f, n_f)) # weight precision matrices
        Sigma_diag = np.zeros((n_t, n_u, n_f)) # weight variances (covariance matrix diagonals)
        for i in range(n_u):
            Sigma[0, i, :, :] = tausq0*np.identity(n_f) # initialize weight covariance matrix
            Sigma_diag[0, :, :] = tausq0
            Lambda[0, i, :, :] = 1/tausq0*np.identity(n_f) # initialize weight precision matrix
        lrate = np.zeros((n_t, n_f, n_u))
        u_psb_so_far = np.zeros(n_u) # keep track of which outcomes (u) have been encountered as possible so far

        # set up response function (depends on response type)
        resp_dict = {'choice': resp_fun.choice,
                     'exct': resp_fun.exct,
                     'supr': resp_fun.supr}
        sim_resp_fun = resp_dict[resp_type]
        
        # loop through time steps
        for t in range(n_t):
            for j in range(n_u):
                if u_psb[t, j] == 1:
                    u_psb_so_far[j] = 1
            r = np.zeros(n_f)
            for i in range(n_f):
                #alpha[t+1, i] = 0.5*u_psb_so_far.sum() + sim_pars['alpha0']
                #new_alpha = 0.5*u_psb_so_far.sum() + sim_pars['alpha0']
                new_alpha = 0.5*n_u + sim_pars['alpha0']
                alpha[t+1, i] = alpha[t, i] + sim_pars['lrate_tausq']*(new_alpha - alpha[t, i])
                musq = mu[t, i, :]**2
                ssq = Sigma[t, :, i, i]
                #beta[t+1, i] = 0.5*np.sum(musq + ssq) + sim_pars['beta0']
                new_beta = 0.5*np.sum(musq + ssq) + sim_pars['beta0']
                beta[t+1, i] = beta[t, i] + sim_pars['lrate_tausq']*(new_beta - beta[t, i])
                tausq[t+1, i] = beta[t+1, i]/(alpha[t+1, i] - 1)
                r[i] = (tausq[t, i] - tausq[t+1, i])/(tausq[t, i]*tausq[t+1, i])
            rdiag = np.diag(r)
            for j in range(n_u):
                # SIGMA VERSION
                #shrink_mat = rdiag + inv(Sigma[t, j, :, :]) # matrix relevant to shrinkage
                #shrink_cond[t, j] = cond(shrink_mat) # condition number
                #mu[t, :, j] = inv(shrink_mat)@inv(Sigma[t, j, :, :])@mu[t, :, j] # shrink mu (weight means)
                #mu[t, :, j] = solve(shrink_mat @ Sigma[t, j, :, :], mu[t, :, j]) # shrink mu (weight means)
                #Sigma[t, j, :, :] = inv(shrink_mat) # shrink Sigma (covariance matrix)
                #Sigma_diag[t, j, :] = np.diag(Sigma[t, j, :, :])
                
                # LAMBDA VERSION
                mu[t, :, j] = inv(rdiag + Lambda[t, j, :, :])@Lambda[t, j, :, :]@mu[t, :, j] # shrink mu (weight means)
                Lambda[t, j, :, :] = rdiag + Lambda[t, j, :, :] # shrink Lambda (precision matrix)
                shrink_cond[t, j] = cond(Lambda[t, j, :, :])
                Sigma[t, j, :, :] = inv(Lambda[t, j, :, :]) # invert Lambda
                Sigma_diag[t, j, :] = np.diag(Sigma[t, j, :, :])
                
            u_hat[t, :] = u_psb[t, :] * (f_x[t, :] @ mu[t, :, :]) # prediction
            b_hat[t, :] = sim_resp_fun(u_hat[t, :], u_psb[t, :], sim_pars['resp_scale']) # response
            delta[t, :] = u[t, :] - u_hat[t, :] # prediction error
            for j in range(n_u):
                update = u_lrn[t, j] == 1
                #update = True
                if update:
                    # SIGMA VERSION
                    #f = f_x[t, :].reshape((n_f, 1))
                    #drift_mat = sim_pars['drift_var']*np.identity(n_f)
                    #numerator = ((Sigma[t, j, :, :] + drift_mat)@f).squeeze()
                    #denominator = f.transpose()@numerator + sim_pars['u_var']
                    #lrate[t, :, j] = numerator/denominator # learning rates are Kalman gain
                    #new_Sigma = Sigma[t, j, :, :] + drift_mat - lrate[t, :, j].reshape((n_f, 1))@f.transpose()@(Sigma[t, j, :, :] + drift_mat)
                    #Sigma[t+1, j, :, :] = new_Sigma # update Sigma
                    #Lambda[t+1, j, :, :] = inv(new_Sigma) # update Lambda
                    
                    # LAMBDA VERSION
                    f = f_x[t, :].reshape((n_f, 1))
                    drift_mat = sim_pars['drift_var']*np.identity(n_f)
                    numerator = ((Sigma[t, j, :, :] + drift_mat)@f).squeeze()
                    denominator = f.transpose()@numerator + sim_pars['u_var']
                    lrate[t, :, j] = numerator/denominator # learning rates are Kalman gain
                    new_Sigma = Sigma[t, j, :, :] + drift_mat - lrate[t, :, j].reshape((n_f, 1))@f.transpose()@(Sigma[t, j, :, :] + drift_mat)
                    Sigma[t+1, j, :, :] = new_Sigma # update Sigma
                    Lambda[t+1, j, :, :] = inv(new_Sigma) # update Lambda
                    #Lambda[t+1, j, :, :] = solve(new_Sigma, np.identity(n_f), assume_a = 'pos') # update Lambda
                else:
                    Sigma[t+1, j, :, :] = Sigma[t, j, :, :]
                    Lambda[t+1, j, :, :] = Lambda[t, j, :, :]
            mu[t+1, :, :] = mu[t, :, :] + u_lrn[t, :] * lrate[t, :, :]*delta[t, :].reshape((1, n_u)) # update mu

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
                                     'alpha' : (['t', 'f_name'], alpha[range(n_t), :]),
                                     'beta' : (['t', 'f_name'], beta[range(n_t), :]),
                                     'shrink_cond' : (['t', 'u_name'], shrink_cond),
                                     'tausq' : (['t', 'f_name'], tausq[range(n_t), :]),
                                     'mu' : (['t', 'f_name', 'u_name'], mu[range(n_t), :, :]), # remove unnecessary last row
                                     'Sigma' : (['t', 'u_name', 'f_name', 'f_name1'], Sigma[range(n_t), :, :, :]),
                                     'Lambda' : (['t', 'u_name', 'f_name', 'f_name1'], Lambda[range(n_t), :, :, :]),
                                     'Sigma_diag' : (['t', 'u_name', 'f_name'], Sigma_diag),
                                     'delta' : (['t', 'u_name'], delta),
                                     'lrate' : (['t', 'f_name', 'u_name'], lrate)},
                        coords = {'t' : range(n_t),
                                  't_name' : ('t', trials.t_name),
                                  'trial' : ('t', trials.trial),
                                  'trial_name' : ('t', trials.trial_name),
                                  'stage' : ('t', trials.stage),
                                  'stage_name' : ('t', trials.stage_name),
                                  'x_name' : x_names,
                                  'f_name' : f_names,
                                  'f_name1' : f_names,
                                  'u_name' : u_names,
                                  'ident' : [ident]},
                        attrs = {'model': self.name,
                                 'model_class' : 'shrinkage',
                                 'schedule' : trials.attrs['schedule'],
                                 'resp_type' : resp_type,
                                 'sim_pars' : sim_pars})
        return ds

########## PARAMETERS ##########
    
par_names = ['alpha0']; par_list = [{'min' : 1.0, 'max' : 10.0, 'default' : 2.0}] # hyperparameter for tausq
par_names += ['beta0']; par_list += [{'min' : 0.0, 'max' : 10.0, 'default' : 2.0}] # other hyperparameter for tausq
par_names += ['u_var']; par_list += [{'min' : 0.0, 'max' : 5.0, 'default' : 0.1}] # outcome variance for Kalman filter
par_names += ['drift_var']; par_list += [{'min' : 0.0, 'max' : 2.0, 'default' : 0.01}] # drift variance for Kalman filter
par_names += ['lrate_tausq']; par_list += [{'min': 0.0, 'max': 0.5, 'default': 0.25}]
par_names += ['resp_scale']; par_list += [{'min': 0.0, 'max': 10.0, 'default': 1.0}]
pars = pd.DataFrame(par_list, index = par_names)
del par_names; del par_list