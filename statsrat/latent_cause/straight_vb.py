import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats
from scipy.special import digamma
from statsrat import resp_fun
from . import kernel

class model:
    '''
    Class for models using straight variational Bayes (no streaming).  Frankly, this could be an alternative "simulate"
    method for the existing class, but I'm just going to make it a separate class at least for now in order to keep things simple.

    '''
    def __init__(self, kernel):
        self.name = 'basic (Bernoulli)'
        self.kernel = kernel
        # determine the model's parameter space
        par_names = kernel.par_names + ['prior_a_x', 'prior_b_x', 'prior_a_y', 'prior_b_y']
        self.pars = pars.loc[par_names + ['alpha', 'resp_scale']]
        
    def simulate(self, trials, par_val = None, n_iter = 5, batch_size = 5, random_resp = False, ident = 'sim'):     
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
        
        # prior for parameters for x
        tilde_tau_x = sim_pars['prior_a_x'] - 1
        tilde_n_x = sim_pars['prior_a_x'] + sim_pars['prior_b_x'] - 2
        tau_x = tilde_tau_x*np.ones((n_t + 1, n_x)) # natural hyperparameters of outcome distribution
        # prior for parameters for y
        tilde_tau_y = sim_pars['prior_a_y'] - 1
        tilde_n_y = sim_pars['prior_a_y'] + sim_pars['prior_b_y'] - 2
        tau_y = tilde_tau_y*np.ones((n_t + 1, n_u)) # natural hyperparameters of stimulus distribution
        
        n = np.zeros((n_t + 1)) # estimated number of observations assigned to each latent cause
        phi_x = np.zeros((n_t, n_t + 1)) # posterior of latent causes after observing x, but before observing u
        phi = np.zeros((n_t, n_t + 1)) # posterior of latent causes after observing both x and y (i.e. u)
        E_r = np.zeros((n_t, n_t)) # mean recency
        V_r = np.zeros((n_t, n_t)) # variance of recency
        T_x = x # sufficient statistic (T(x))
        T_y = u_psb*u # sufficient statistic (T(y))
                         
        # set up response function (depends on response type)
        resp_dict = {'choice': resp_fun.choice,
                     'exct': resp_fun.exct,
                     'supr': resp_fun.supr}
        sim_resp_fun = resp_dict[trials.resp_type]
        
        # initialize
        K = 1
        phi[:, 0] = 1
        
        # run coordinate ascent iterations
        for i in range(n_iter):
            # variables for indexing
            rk1 = range(K + 1)
            rk = range(K)
            
            # variational distribution for theta
            tau_x[rk, :] = tilde_tau_x + phi[:, rk].transpose()@T_x # omit x_sofar for simplicity
            tau_y[rk, :] = tilde_tau_y + phi[:, rk].transpose()@T_y
            n[rk] = np.sum(phi[:, rk], axis = 0)
            n_for_x = np.repeat(n[rk1] + tilde_n_x, n_x).reshape((K + 1, n_x))
            n_for_y = np.repeat(n[rk1] + tilde_n_y, n_u).reshape((K + 1, n_u))
            
            # variational distribution for z
            for t in range(n_t):
                if t > 0:
                    # approximate Eq[log p(z_n = t | z_1, ..., z_{n-1})] (expected log-prior)
                    kernel = self.kernel(t, time, sim_pars) # temporal kernel (i.e. decay function for latent causes)
                    E_r[t, rk1] = np.sum(kernel.reshape((t, 1))*phi[0:t, rk1], axis = 0)
                    V_r[t, rk1] = np.sum((kernel**2).reshape((t, 1))*phi[0:t, rk1]*(1 - phi[0:t, rk1]), axis = 0)
                    sum_r = np.sum(kernel)
                    E_log_prior = np.zeros(K + 1)
                    E_log_prior[rk] = np.log(E_r[t, rk]) - 0.5*V_r[t, rk]/(E_r[t, rk]**2) - np.log(sum_r + sim_pars['alpha'])
                    E_log_prior[K] = np.log(sim_pars['alpha']) - np.log(sum_r + sim_pars['alpha'])
                else:
                    E_log_prior = np.zeros(K + 1)
                
                # compute Eq[log p(x_n | z_n = t, eta)] (expected log-likelihood of x)
                E_eta_x = digamma(tau_x[rk1, :]) - digamma(n_for_x - tau_x[rk1, :] + 1) # expected natural parameter (eta)
                E_a_eta_x = digamma(n_for_x - tau_x[rk1, :] + 1) - digamma(n_for_x + 2) # expected log partition function (a(eta))
                b_x = 0 # log base measure (b(x))
                Ell_cues = E_eta_x*T_x[t, :] - E_a_eta_x - b_x # expected log likelihood for each cue
                E_log_lik_x = np.sum(Ell_cues, axis = 1) # assumed independent -> add log_lik across cues (omit x_sofar)
            
                # compute Eq[log p(y_n | z_n = t, eta)] (expected log-likelihood of y)
                E_eta_y = digamma(tau_y[rk1, :]) - digamma(n_for_y - tau_y[rk1, :] + 1) # expected natural parameter
                E_a_eta_y = digamma(n_for_y - tau_y[rk1, :] + 1) - digamma(n_for_y + 2) # expected log partition function
                b_y = 0 # log base measure (b(y))
                Ell_outcomes = E_eta_y*T_y[t, :] - E_a_eta_y - b_y # expected log likelihood for each outcome
                E_log_lik_y = np.sum(u_psb[t, :]*Ell_outcomes, axis = 1) # assumed independent -> add log_lik across outcomes
            
                # compute phi
                s = np.exp(E_log_lik_x + E_log_lik_y + E_log_prior)
                phi[t, range(K + 1)] = s/s.sum()
                
            # decide whether to expand latent causes
            print()
            print('----- miau -----')
            print(np.sum(phi[:, K]))
            #print(tau_x[rk1, :])
            #print(E_log_lik_x)
            #print(E_log_lik_y)
            #print(E_log_prior)
            #print(K)
            #if (n[K] > 1) and (i > 0):
            if (np.sum(phi[:, K]) > 1) and (i > 0):
                K += 1
            
        # loop through time steps to predict y (recall that 'y' = 'u')
        rk = range(K)
        n_for_x = np.repeat(n[rk] + tilde_n_x, n_x).reshape((K, n_x))
        for t in range(n_t):
            if t > 0:
                # approximate Eq[log p(z_n = t | z_1, ..., z_{n-1})] (expected log-prior)
                kernel = self.kernel(t, time, sim_pars) # temporal kernel (i.e. decay function for latent causes)
                E_r[t, rk] = np.sum(kernel.reshape((t, 1))*phi[0:t, rk], axis = 0)
                V_r[t, rk] = np.sum((kernel**2).reshape((t, 1))*phi[0:t, rk]*(1 - phi[0:t, rk]), axis = 0)
                sum_r = np.sum(kernel)
                E_log_prior = np.zeros(K)
                E_log_prior[range(K)] = np.log(E_r[t, rk]) - 0.5*V_r[t, rk]/(E_r[t, rk]**2) - np.log(sum_r + sim_pars['alpha'])
            else:
                E_log_prior = np.zeros(K)

            # compute Eq[log p(x_n | z_n = t, eta)] (expected log-likelihood of x)
            E_eta_x = digamma(tau_x[rk, :]) - digamma(n_for_x - tau_x[rk, :] + 1) # expected natural parameter (eta)
            E_a_eta_x = digamma(n_for_x - tau_x[rk, :] + 1) - digamma(n_for_x + 2) # expected log partition function (a(eta))
            b_x = 0 # log base measure (b(x))
            Ell_cues = E_eta_x*T_x[t, :] - E_a_eta_x - b_x # expected log likelihood for each cue
            E_log_lik_x = np.sum(Ell_cues, axis = 1) # assumed independent -> add log_lik across cues (omit x_sofar)

            # compute phi_x
            s = np.exp(E_log_lik_x + E_log_prior)
            phi_x[t, rk] = s/s.sum()

            # predict y
            n_for_calc = n[rk].reshape((K, 1)) + tilde_n_y
            E_post_pred = (tau_y[rk, :] + 1)/(n_for_calc + 2) # mean of posterior predictive
            u_hat[t, :] = u_psb[t, :]*np.sum(phi_x[t, rk].reshape((K, 1))*E_post_pred, axis = 0) # predicted outcome (u)
            b_hat[t, :] = sim_resp_fun(u_hat[t, :], u_psb[t, :], sim_pars['resp_scale']) # response
            
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
        ds = ds.assign_coords({'z_name' : np.array(range(K), dtype = str), 'ident' : [ident]})
        ds = ds.assign({'u_psb' : (['t', 'u_name'], u_psb),
                        'u_lrn' : (['t', 'u_name'], u_lrn),
                        'u_hat' : (['t', 'u_name'], u_hat),
                        'b_hat' : (['t', 'u_name'], b_hat),
                        'b' : (['t', 'u_name'], b),
                        'tau_x' : (['z_name', 'x_name'], tau_x[rk, :]),
                        'tau_y' : (['z_name', 'u_name'], tau_y[rk, :]),
                        'n' : (['z_name'], n[rk]),
                        'E_r' : (['t', 'z_name'], E_r[:, rk]),
                        'V_r' : (['t', 'z_name'], V_r[:, rk]),
                        'phi_x' : (['t', 'z_name'], phi_x[:, rk]),
                        'phi' : (['t', 'z_name'], phi[:, rk])})
        ds = ds.assign_attrs({'model': self.name,
                              'model_class': 'latent_cause',
                              'sim_pars': sim_pars})
        # add in conventional parameter estimates to dataset
        a_x = tau_x[rk, :] + 1
        b_x = np.repeat(n[rk] + tilde_n_x, n_x).reshape((K, n_x)) - tau_x[rk, :] + 1
        a_y = tau_y[rk, :] + 1
        b_y = np.repeat(n[rk] + tilde_n_y, n_u).reshape((K, n_u)) - tau_y[rk, :] + 1
        E_theta_x = a_x/(a_x + b_x) # mean conventional parameter (probability) for x
        E_theta_y = a_y/(a_y + b_y) # mean conventional parameter (probability) for y
        ds = ds.assign({'E_theta_x': (['z_name', 'x_name'], E_theta_x), 'E_theta_y': (['z_name', 'y_name'], E_theta_y)})
        
        return ds

########## PARAMETERS ##########
# Note: allowing prior_a to be close to 1 seems to cause problems.
par_names = []; par_list = []                         
par_names += ['gamma']; par_list += [{'min': 0.0, 'max': 5.0, 'default': 0.5, 'description': 'decay rate for exponential SCRP; higher -> favors more recent latent causes'}] 
par_names += ['alpha']; par_list += [{'min': 0.0, 'max': 20.0, 'default': 0.5, 'description': 'concentration parameter; higher -> tend to infer more latent causes'}]
par_names += ['prior_a_x']; par_list += [{'min': 1.0, 'max': 40.0, 'default': 2.0, 'description': 'prior hyperparameter for eta for x (log-odds in Bernoulli likelihood)'}]
par_names += ['prior_b_x']; par_list += [{'min': 1.0, 'max': 40.0, 'default': 5.0, 'description': 'prior hyperparameter for eta for x (log-odds in Bernoulli likelihood)'}]
par_names += ['prior_a_y']; par_list += [{'min': 1.0, 'max': 40.0, 'default': 2.0, 'description': 'prior hyperparameter for eta for y (log-odds in Bernoulli likelihood)'}]
par_names += ['prior_b_y']; par_list += [{'min': 1.0, 'max': 40.0, 'default': 5.0, 'description': 'prior hyperparameter for eta for y (log-odds in Bernoulli likelihood)'}]
par_names += ['resp_scale']; par_list += [{'min': 0.0, 'max': 10.0, 'default': 1.0, 'description': 'scales softmax/logistic response functions'}]

pars = pd.DataFrame(par_list, index = par_names)
del par_names; del par_list