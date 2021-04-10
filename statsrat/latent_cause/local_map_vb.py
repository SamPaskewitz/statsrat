import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats
from scipy.special import digamma
from statsrat import resp_fun
from . import kernel

# https://github.com/LukasNeugebauer/LCM_python/blob/master/LCM.py
# https://github.com/sjgershm/LCM

class model:
    '''
    Local MAP approximation/variational Bayes hybrid (not mathematically justified or thought out).
    The local MAP approximation is used for two purposes: approximating the prior on latent causes,
    and deciding when to add a new latent cause.  Everything else is done via streaming variational Bayes.
    
    Notes
    -----
    mu | sigma^2 ~ N(tau1/n, sigma^2/n)
    1/sigma^2 ~ Gamma((n + 3)/2, (n tau2 - tau1^2)/(2 n))
    '''
    def __init__(self, name, kernel):
        '''
        Parameters
        ----------
        '''
        self.name = name
        self.kernel = kernel
        # determine the model's parameter space
        self.par_names = kernel.par_names + ['prior_tau2_x', 'prior_nu_x', 'prior_tau2_y', 'prior_nu_y', 'stick', 'alpha', 'resp_scale']
        self.pars = pars.loc[self.par_names]
        
    def simulate(self, trials, par_val = None, random_resp = False, ident = 'sim'):
        '''
        Simulate a trial sequence once with known model parameters.
        
        Parameters
        ----------
        trials: dataset (xarray)
            Time step level experimental data (cues, outcomes etc.).

        par_val: list, optional
            Learning model parameters (floats or ints).

        random_resp: str, optional
            Whether or not simulated responses should be random.  Defaults
            to false, in which case behavior (b) is identical to expected
            behavior (b_hat); this saves some computation time.  If true
            and resp_type is 'choice', then discrete responses are selected
            using b_hat as choice probabilities.  If true and resp_type is
            'exct' or 'supr' then a small amount of normally distributed
            noise (sd = 0.01) is added to b_hat.

        ident: str, optional
            Individual participant identifier.  Defaults to 'sim'.

        Returns
        -------
        ds: dataset
            Simulation data.

        '''        
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
        n_t = x.shape[0] # number of time points
        n_x = x.shape[1] # number of stimulus attributes
        n_u = u.shape[1] # number of outcomes/response options
        u_hat = np.zeros((n_t, n_u)) # outcome predictions
        b_hat = np.zeros((n_t, n_u)) # expected behavior
        time = trials['time'].values # real world time (in arbitrary units, starting at 0)
        
        # prior for x parameters
        tau1_x = np.zeros((n_t + 1, n_t + 1, n_x))
        tau2_x = sim_pars['prior_tau2_x']*np.ones((n_t + 1, n_t + 1, n_x))
        nu_x = sim_pars['prior_nu_x']*np.ones((n_t + 1, n_t + 1, n_x))
        # prior for y parameters
        tau1_y = np.zeros((n_t + 1, n_t + 1, n_u))
        tau2_y = sim_pars['prior_tau2_y']*np.ones((n_t + 1, n_t + 1, n_u))
        nu_y = sim_pars['prior_nu_y']*np.ones((n_t + 1, n_t + 1, n_u))
        
        E_log_prior = np.zeros((n_t, n_t))
        E_log_lik_x = np.zeros((n_t, n_t))
        E_log_lik_y = np.zeros((n_t, n_t))
        est_mu_x = np.zeros((n_t, n_t, n_x))
        prior_E_eta1_x = -(sim_pars['prior_nu_x']*(sim_pars['prior_nu_x'] + 3))/(2*sim_pars['prior_nu_x']*sim_pars['prior_tau2_x'])
        est_sigma_x = (1/np.sqrt(-2*prior_E_eta1_x))*np.ones((n_t, n_t, n_x))
        est_mu_y = np.zeros((n_t, n_t, n_u))
        E_eta1_y = -(sim_pars['prior_nu_y']*(sim_pars['prior_nu_y'] + 3))/(2*sim_pars['prior_nu_y']*sim_pars['prior_tau2_y'])
        est_sigma_y = (1/np.sqrt(-2*E_eta1_y))*np.ones((n_t, n_t, n_u))
        z = np.zeros((n_t), dtype = int) # hard latent cause assignments
        z_onehot = np.zeros((n_t, n_t + 1)) # one hot representation of z, i.e. winner is 1 and all others are 0
        n = np.zeros((n_t + 1, n_t + 1)) # estimated number of observations assigned to each latent cause
        N = np.zeros(n_t + 1, dtype=int) # estimated number of latent causes
        N[[0, 1]] = 1
        phi_x = np.zeros((n_t, n_t + 1)) # posterior of latent causes after observing x, but before observing u
        phi = np.zeros((n_t, n_t + 1)) # posterior of latent causes after observing both x and y (i.e. u)
        x_sofar = np.zeros(n_x) # keep track of cues (x) observed so far
                         
        # set up response function (depends on response type)
        resp_dict = {'choice': resp_fun.choice,
                     'exct': resp_fun.exct,
                     'supr': resp_fun.supr}
        sim_resp_fun = resp_dict[trials.resp_type]
        
        # run calculations for first time step
        x_sofar[x[0, :] > 0] = 1 # keep track of cues observed so far
        b_hat[0, :] = sim_resp_fun(u_hat[0, :], u_psb[0, :], sim_pars['resp_scale']) # response
        phi_x[0, 0] = 1
        phi[0, 0] = 1
        tau1_x[1, 0, :] = tau1_x[0, 0, :] + x_sofar*x[0, :]
        tau2_x[1, 0, :] = tau2_x[0, 0, :] + x_sofar*(x[0, :]**2)
        tau1_y[1, 0, :] = tau1_y[0, 0, :] + u_psb[0, :]*u[0, :]
        tau2_y[1, 0, :] = tau2_y[0, 0, :] + u_psb[0, :]*(u[0, :]**2)
        n[1, 0] = n[0, 0] + 1
        z[0] = 0
        z_onehot[0, 0] = 1
        N[0] = 1
               
        # loop through time steps
        for t in range(1, n_t):
            # preliminary stuff
            x_sofar[x[t, :] > 0] = 1 # keep track of cues observed so far
            ind_n = range(N[t]) # index for latent causes
            ind_n1 = range(N[t] + 1) # index latent causes
            
            # compute Eq[log p(x_n | z_n = t, eta)] (expected log-likelihood of x)
            E_eta0_x = (nu_x[t, ind_n1, :] + 3)*tau1_x[t, ind_n1, :]/(nu_x[t, ind_n1, :]*tau2_x[t, ind_n1, :] - tau1_x[t, ind_n1, :]**2)
            E_eta1_x = -(nu_x[t, ind_n1, :]*(nu_x[t, ind_n1, :] + 3))/(2*(nu_x[t, ind_n1, :]*tau2_x[t, ind_n1, :] - tau1_x[t, ind_n1, :]**2))
            est_mu_x[t, ind_n1, :] = -E_eta0_x/(2*E_eta1_x)
            est_sigma_x[t, ind_n1, :] = 1/np.sqrt(-2*E_eta1_x)
            Ell_cues = stats.norm.logpdf(x[t, :], est_mu_x[t, ind_n1], est_sigma_x[t, ind_n1])
            E_log_lik_x[t, ind_n1] = np.sum(x_sofar*Ell_cues, axis = 1) # assumed independent -> add log_lik across cues
            
            # approximate Eq[log p(z_n = t | z_1, ..., z_{n-1})] (expected log-prior)
            K = self.kernel(t, time, sim_pars).reshape((t, 1)) # temporal kernel (i.e. decay function for latent causes)
            r = np.sum(K*z_onehot[0:t, ind_n], axis = 0) # recency
            sum_r = np.sum(r)
            log_denominator = np.log(sum_r + sim_pars['stick'] + sim_pars['alpha'])
            E_log_prior[t, ind_n] = np.log(r[ind_n]) - log_denominator # old clusters
            E_log_prior[t, z[t-1]] += sim_pars['stick'] # add stickiness to most recent cluster
            E_log_prior[t, N[t]] = np.log(sim_pars['alpha']) - log_denominator # new cluster

            # compute E_log_lik_phi based on x
            s = np.exp(E_log_lik_x[t, ind_n1] + E_log_prior[t, ind_n1])
            new_phi_x = s/s.sum()
            phi_x[t, ind_n] = new_phi_x[ind_n]
                                           
            # predict y (recall that 'y' = 'u')
            E_eta0_y = (nu_y[t, ind_n1, :] + 3)*tau1_y[t, ind_n1, :]/(nu_y[t, ind_n1, :]*tau2_y[t, ind_n1, :] - tau1_y[t, ind_n1, :]**2)
            E_eta1_y = -(nu_y[t, ind_n1, :]*(nu_y[t, ind_n1, :] + 3))/(2*(nu_y[t, ind_n1, :]*tau2_y[t, ind_n1, :] - tau1_y[t, ind_n1, :]**2))
            est_mu_y[t, ind_n1, :] = -E_eta0_y/(2*E_eta1_y)
            u_hat[t, :] = u_psb[t, :]*np.sum(new_phi_x.reshape((N[t] + 1, 1))*est_mu_y[t, ind_n1], axis = 0)
            b_hat[t, :] = sim_resp_fun(u_hat[t, :], u_psb[t, :], sim_pars['resp_scale']) # response

            # compute Eq[log p(y_n | z_n = t, eta)] (expected log-likelihood of y)
            est_sigma_y[t, ind_n1, :] = 1/np.sqrt(-2*E_eta1_y)
            Ell_outcomes = stats.norm.logpdf(u[t, :], est_mu_y[t, ind_n1], est_sigma_y[t, ind_n1])
            E_log_lik_y[t, ind_n1] = np.sum(u_psb[t, :]*Ell_outcomes, axis = 1) # assumed independent -> add log_lik across outcomes

            # compute phi
            s_xy = np.exp(E_log_lik_x[t, ind_n1] + E_log_lik_y[t, ind_n1] + E_log_prior[t, ind_n1])
            phi[t, ind_n1] = s_xy/s_xy.sum()
                
            # hard latent cause assignment
            z[t] = np.argmax(phi[t, :]) # winning (most probable) cluster
            z_onehot[t, z[t]] = 1
            if z[t] == N[t]:
                phi_learn = phi[t, :]
                N[t + 1] = N[t] + 1 # increase number of latent causes
            else:
                phi_learn = np.zeros(n_t + 1)
                phi_learn[ind_n] = phi[t, ind_n]/phi[t, ind_n].sum() # drop new cause and re-normalize over old latent causes
                N[t + 1] = N[t]

            # learning (update hyperparameters)
            tau1_x[t + 1, :, :] = tau1_x[t, :, :] + x_sofar*np.outer(phi_learn, x[t, :])
            tau2_x[t + 1, :, :] = tau2_x[t, :, :] + x_sofar*np.outer(phi_learn, x[t, :]**2)
            nu_x[t + 1, :, :] = nu_x[t, :, :] + np.outer(phi_learn, x_sofar)
            tau1_y[t + 1, :, :] = tau1_y[t, :, :] + u_psb[t, :]*np.outer(phi_learn, u[t, :])
            tau2_y[t + 1, :, :] = tau2_y[t, :, :] + u_psb[t, :]*np.outer(phi_learn, u[t, :]**2)
            nu_y[t + 1, :, :] = nu_y[t, :, :] + np.outer(phi_learn, u_psb[t, :])
            n[t + 1, :] = n[t, :] + phi_learn
            
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
        ds = trials.copy(deep = True)
        ds = ds.assign_coords({'z_name' : np.array(range(n_t), dtype = str), 'ident' : [ident]})
        ds = ds.assign({'u_psb' : (['t', 'u_name'], u_psb),
                        'u_lrn' : (['t', 'u_name'], u_lrn),
                        'u_hat' : (['t', 'u_name'], u_hat),
                        'b_hat' : (['t', 'u_name'], b_hat),
                        'b' : (['t', 'u_name'], b),
                        'est_mu_x' : (['t', 'z_name', 'x_name'], est_mu_x),
                        'est_sigma_x' : (['t', 'z_name', 'x_name'], est_sigma_x),
                        'est_precision_x' : (['t', 'z_name', 'x_name'], 1/est_sigma_x**2),
                        'est_mu_y' : (['t', 'z_name', 'u_name'], est_mu_y),
                        'est_sigma_y' : (['t', 'z_name', 'u_name'], est_sigma_y),
                        'est_precision_y' : (['t', 'z_name', 'u_name'], 1/est_sigma_y**2),
                        'n' : (['t', 'z_name'], n[0:-1, 0:-1]),
                        'z' : (['t'], z),
                        'phi_x' : (['t', 'z_name'], phi_x[:, 0:-1]),
                        'phi' : (['t', 'z_name'], phi[:, 0:-1]),
                        'N' : (['t'], N[0:(t+1)]),
                        'E_log_prior': (['t', 'z_name'], E_log_prior),
                        'E_log_lik_x': (['t', 'z_name'], E_log_lik_x),
                        'E_log_lik_y': (['t', 'z_name'], E_log_lik_y)})
        ds = ds.assign_attrs({'model': self.name,
                              'model_class': 'latent_cause',
                              'sim_pars': sim_pars})
        return ds

########## PARAMETERS ##########
# Note: allowing prior_a to be close to 1 seems to cause problems.
par_names = []; par_list = []                         
par_names += ['gamma']; par_list += [{'min': 0.0, 'max': 5.0, 'default': 0.5, 'description': 'decay rate for exponential SCRP; higher -> favors more recent latent causes'}] 
par_names += ['alpha']; par_list += [{'min': 0.0, 'max': 15.0, 'default': 1.0, 'description': 'concentration parameter; higher -> tend to infer more latent causes'}]
par_names += ['prior_tau2_x']; par_list += [{'min': 0.01, 'max': 5.0, 'default': 1.0, 'description': 'prior hyperparameter for eta for x'}]
par_names += ['prior_nu_x']; par_list += [{'min': 0.01, 'max': 10.0, 'default': 5.0, 'description': 'prior hyperparameter for eta for x'}]
par_names += ['prior_tau2_y']; par_list += [{'min': 0.01, 'max': 5.0, 'default': 1.0, 'description': 'prior hyperparameter for eta for y'}]
par_names += ['prior_nu_y']; par_list += [{'min': 0.01, 'max': 10.0, 'default': 5.0, 'description': 'prior hyperparameter for eta for y'}]
par_names += ['stick']; par_list += [{'min': 0.0, 'max': 5.0, 'default': 1.0, 'description': 'stickiness for CRP prior'}]
par_names += ['resp_scale']; par_list += [{'min': 0.0, 'max': 10.0, 'default': 1.0, 'description': 'scales softmax/logistic response functions'}]

pars = pd.DataFrame(par_list, index = par_names)
del par_names; del par_list