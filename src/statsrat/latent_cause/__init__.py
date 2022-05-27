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
    Class for Bayesian latent cause learning models (Anderson, 1991; Gershman, Blei & Niv, 2010).
    
    Attributes
    ----------
    name: str
        Model name.
    kernel: function
        Temporal kernel for distance dependent prior on latent causes.
    par_names: list
        Names of the model's free parameters (strings).
    pars: dict
        Information about model parameters (min, max, default, description).
    
    Methods
    -------
    simulate(self, trials, par_val = None, n_z = 10, n_p = 50, random_resp = False, ident = 'sim', sim_type = 'local_vb')
        Simulate a trial sequence once with known model parameters using
        either the .local_vb() or .particle() method.  This is just a wrapper
        for those simulation methods.
        
    local_vb(self, trials, par_val = None, n_z = 10, random_resp = False, ident = 'sim')     
        Simulate the model using a combination of local MAP and variational Bayes.
    
    particle_filter(self, trials, par_val = None, n_z = 10, n_p = 50, random_resp = False, ident = 'sim')
        Simulate the model using a particle filter algorithm.
    
    Notes
    -----
    The local_vb method uses a local MAP approximation for two purposes:
    approximating the prior on latent causes, and deciding when to add a 
    new latent cause.  Everything else is done via streaming variational Bayes.
    
    Currently both outcomes (y) and predictor stimuli (x) are drawn from independent
    normal distributions with the following hyperpriors:
    mu | sigma^2 ~ N(tau1/n, sigma^2/n)
    1/sigma^2 ~ Gamma((n + 3)/2, (n tau2 - tau1^2)/(2 n))
    
    In the future I may add other distribution options, e.g. multinomial or Bernoulli.
    
    *** MODIFY THE MINIMUM FOR THE NUMERATOR TO THE PARTICLE FILTER METHOD ***
    *** EXPLAIN THE NORMAL DISTRIBUTION PARAMETERIZATION ***
    
    Relevant Papers
    ---------------
    Anderson, J. R. (1991).
    The adaptive nature of human categorization.
    Psychological Review, 98(3), 409.
    
    Broderick, T., Boyd, N., Wibisono, A., Wilson, A. C., & Jordan, M. I. (2013).
    Streaming variational Bayes.
    ArXiv Preprint ArXiv:1307.6769.
    
    Gershman, S. J., Blei, D. M., & Niv, Y. (2010).
    Context, learning, and extinction.
    Psychological Review, 117(1), 197–209.
    
    Gershman, S. J., & Niv, Y. (2012).
    Exploring a latent cause theory of classical conditioning.
    Learning & Behavior, 40(3), 255–268.
    
    Huynh, V., & Phung, D. (2017). Streaming clustering with Bayesian nonparametric models.
    Neurocomputing, 258, 52–62.
    
    Sanborn, A. N., Griffiths, T. L., & Navarro, D. J. (2010).
    Rational approximations to rational models: Alternative algorithms
    for category learning. Psychological Review, 117(4), 1144–1167.
    
    Zhu, X., Ghahramani, Z., & Lafferty, J. (n.d.).
    Time-Sensitive Dirichlet Process Mixture Models.    
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
        
    def simulate(self, trials, par_val = None, n_z = 10, n_p = 50, random_resp = False, ident = 'sim', sim_type = 'local_vb'):
        '''
        Simulate a trial sequence once with known model parameters using
        either the .local_vb() or .particle() method.
        
        Parameters
        ----------
        trials: dataset (xarray)
            Time step level experimental data (cues, outcomes etc.).

        par_val: list, optional
            Learning model parameters (floats or ints).
            
        n_z: int, optional
            Maximum number of latent causes.  Defaults to 10.
            
        n_p: int, optional
            Number of particles.  Defaults to 50.  Only relevant if using
            the .particle() simulation methods (i.e. sim_type = 'particle').
            
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
            
        sim_type: str, optional
            Determines what kind of simulation to perform.  The options are
            'local_vb' (combination of local MAP and mean field variational
            Bayes updates) and 'particle' (particle filter).  Defaults to
            'local_vb'.

        Returns
        -------
        ds: dataset
            Simulation data.
        
        Notes
        -----
        The .simulate() method is just a wrapper for the .local_vb() and
        .particle() methods, with the choice between these method indicated
        by the sim_type argument.  The .local_vb() and .particle() methods
        can also be used on their own, without using the .simulate() method
        as a wrapper.  The .simulate() method is only present in latent cause
        models in order to interface with the rest of the Statrat package
        (e.g. functions for performing model fitting and OATs).
        
        Note on terminology: "cluster" = "latent cause"
        '''
        method_dict = {'local_vb': lambda par_val: self.local_vb(trials, par_val, n_z, random_resp, ident),
                       'particle': lambda par_val: self.particle_filter(trials, par_val, n_z, n_p, random_resp, ident)}
        return method_dict[sim_type](par_val)
        
    def local_vb(self, trials, par_val = None, n_z = 10, random_resp = False, ident = 'sim'):        
        '''
        Simulate the model using a combination of local MAP and variational Bayes.
        
        Parameters
        ----------
        trials: dataset (xarray)
            Time step level experimental data (cues, outcomes etc.).

        par_val: list, optional
            Learning model parameters (floats or ints).
            
        n_z: int, optional
            Maximum number of latent causes.  Defaults to 10.
            
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
            
        sim_type: str, optional
            Determines what kind of simulation to perform.  The options are
            'local_vb' (combination of local MAP and mean field variational
            Bayes updates) and 'particle' (particle filter).  Defaults to
            'local_vb'.

        Returns
        -------
        ds: dataset
            Simulation data.
            
        Explanation of variables in ds
        ------------------------------
        y_psb: indicator vector for outcomes (y) that are possible on the trial (from the learner's perspective)
        y_lrn: indicator vector for outcomes (y) for which there is feedback and hence learning will occur
        y_hat: outcome predictions
        b_hat: expected value of behavioral response
        b: vector representing actual behavioral response (identical to b_hat unless the random_resp argument is set to True)
        est_mu_x: estimated mean of x
        est_sigma_x: estimated standard deviation of x
        est_precision_x: estimated precision of x
        est_mu_y: estimated mean of y
        est_sigma_y: estimated standard deviation of y
        est_precision_y: estimated precision of y
        n: estimated number of observations assigned to each latent cause
        z: hard latent cause assignments
        phi_x: posterior of latent causes after observing x, but before observing y
        phi: posterior of latent causes after observing both x and y
        N: estimated number of latent causes
        E_log_prior: expected log-prior for latent causes
        E_log_lik_x: expected log-likelihood of x for latent causes
        E_log_lik_y: expected log-likelihood of y for latent causes
        b_index: index of behavioral response (only present if response type is 'choice' and random_resp is True)
        b_name: name of behavioral response (only present if response type is 'choice' and random_resp is True)
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
        y = np.array(trials['y'], dtype = 'float64')
        y_psb = np.array(trials['y_psb'], dtype = 'float64')
        y_lrn = np.array(trials['y_lrn'], dtype = 'float64')
        x_names = list(trials.x_name.values)
        y_names = list(trials.y_name.values)
        n_t = x.shape[0] # number of time points
        n_x = x.shape[1] # number of stimulus attributes
        n_y = y.shape[1] # number of outcomes/response options
        y_hat = np.zeros((n_t, n_y)) # outcome predictions
        b_hat = np.zeros((n_t, n_y)) # expected behavior
        time = trials['time'].values # real world time (in arbitrary units, starting at 0)
        x_sofar = np.zeros(n_x) # keep track of cues (x) observed so far
        
        # prior for x parameters
        tau1_x = np.zeros((n_t + 1, n_z, n_x))
        tau2_x = sim_pars['prior_tau2_x']*np.ones((n_t + 1, n_z, n_x))
        nu_x = sim_pars['prior_nu_x']*np.ones((n_t + 1, n_z, n_x))
        # prior for y parameters
        tau1_y = np.zeros((n_t + 1, n_z, n_y))
        tau2_y = sim_pars['prior_tau2_y']*np.ones((n_t + 1, n_z, n_y))
        nu_y = sim_pars['prior_nu_y']*np.ones((n_t + 1, n_z, n_y))
        
        E_log_prior = np.zeros((n_t, n_z))
        E_log_lik_x = np.zeros((n_t, n_z))
        E_log_lik_y = np.zeros((n_t, n_z))
        est_mu_x = np.zeros((n_t, n_z, n_x))
        prior_E_eta2_x = -(sim_pars['prior_nu_x']*(sim_pars['prior_nu_x'] + 3))/(2*sim_pars['prior_nu_x']*sim_pars['prior_tau2_x'])
        est_sigma_x = (1/np.sqrt(-2*prior_E_eta2_x))*np.ones((n_t, n_z, n_x))
        est_mu_y = np.zeros((n_t, n_z, n_y))
        E_eta2_y = -(sim_pars['prior_nu_y']*(sim_pars['prior_nu_y'] + 3))/(2*sim_pars['prior_nu_y']*sim_pars['prior_tau2_y'])
        est_sigma_y = (1/np.sqrt(-2*E_eta2_y))*np.ones((n_t, n_z, n_y))
        z = np.zeros((n_t), dtype = int) # hard latent cause assignments
        z_onehot = np.zeros((n_t, n_z)) # one hot representation of z, i.e. winner is 1 and all others are 0
        n = np.zeros((n_t + 1, n_z)) # estimated number of observations assigned to each latent cause
        N = np.zeros(n_t + 1, dtype=int) # estimated number of latent causes
        N[[0, 1]] = 1
        phi_x = np.zeros((n_t, n_z)) # posterior of latent causes after observing x, but before observing y
        phi = np.zeros((n_t, n_z)) # posterior of latent causes after observing both x and y
                         
        # set up response function (depends on response type)
        resp_dict = {'choice': resp_fun.choice,
                     'exct': resp_fun.exct,
                     'supr': resp_fun.supr}
        sim_resp_fun = resp_dict[trials.resp_type]
        
        # run calculations for first time step
        x_sofar[x[0, :] > 0] = 1 # keep track of cues observed so far
        b_hat[0, :] = sim_resp_fun(y_hat[0, :], y_psb[0, :], sim_pars['resp_scale']) # response
        phi_x[0, 0] = 1
        phi[0, 0] = 1
        tau1_x[1, 0, :] = tau1_x[0, 0, :] + x_sofar*x[0, :]
        tau2_x[1, 0, :] = tau2_x[0, 0, :] + x_sofar*(x[0, :]**2)
        tau1_y[1, 0, :] = tau1_y[0, 0, :] + y_psb[0, :]*y[0, :]
        tau2_y[1, 0, :] = tau2_y[0, 0, :] + y_psb[0, :]*(y[0, :]**2)
        n[1, 0] = n[0, 0] + 1
        z[0] = 0
        z_onehot[0, 0] = 1
        N[0] = 1
               
        # loop through time steps
        for t in range(1, n_t):
            # preliminary stuff
            x_sofar[x[t, :] > 0] = 1 # keep track of cues observed so far
            if N[t] < n_z:
                N_zt = N[t] + 1 # maximum number of latent causes considered this time step
                ind_n = range(N[t])
                ind_n1 = range(N[t] + 1)
            else:
                N_zt = n_z
                ind_n = range(N[t])
                ind_n1 = ind_n
            
            # compute Eq[log p(x_n | z_n = t, eta)] (expected log-likelihood of x)
            E_eta1_x = (nu_x[t, ind_n1, :] + 3)*tau1_x[t, ind_n1, :]/(nu_x[t, ind_n1, :]*tau2_x[t, ind_n1, :] - tau1_x[t, ind_n1, :]**2)
            E_eta2_x = -(nu_x[t, ind_n1, :]*(nu_x[t, ind_n1, :] + 3))/(2*(nu_x[t, ind_n1, :]*tau2_x[t, ind_n1, :] - tau1_x[t, ind_n1, :]**2))
            est_mu_x[t, ind_n1, :] = -E_eta1_x/(2*E_eta2_x)
            est_sigma_x[t, ind_n1, :] = 1/np.sqrt(-2*E_eta2_x)
            Ell_cues = stats.norm.logpdf(x[t, :], est_mu_x[t, ind_n1], est_sigma_x[t, ind_n1])
            E_log_lik_x[t, ind_n1] = np.sum(x_sofar*Ell_cues, axis = 1) # assumed independent -> add log_lik across cues
            
            # approximate Eq[log p(z_n = t | z_1, ..., z_{n-1})] (expected log-prior on cluster membership)
            K = self.kernel(t, N[t], time, sim_pars) # temporal kernel (i.e. decay function for latent causes)
            r = np.sum(K*z_onehot[0:t, ind_n], axis = 0) # recency
            num_old = r[ind_n] # numerator of prior for old clusters
            num_old[z[t-1]] += sim_pars['stick'] # add stickiness to most recent cluster
            num_old[z[t-1]] = np.max([num_old[z[t-1]], 0.0001]) # minimum numerator is 0.0001 (needed sometimes with negative stickiness parameter)
            log_denominator = np.log(np.sum(num_old) + sim_pars['alpha'])
            E_log_prior[t, ind_n] = np.log(num_old) - log_denominator # expected log prior for old clusters
            if N[t] < n_z:
                E_log_prior[t, N[t]] = np.log(sim_pars['alpha']) - log_denominator # new cluster

            # compute E_log_lik_phi based on x
            s = np.exp(E_log_lik_x[t, ind_n1] + E_log_prior[t, ind_n1])
            new_phi_x = s/s.sum()
            phi_x[t, ind_n] = new_phi_x[ind_n]
                                           
            # predict y
            E_eta1_y = (nu_y[t, ind_n1, :] + 3)*tau1_y[t, ind_n1, :]/(nu_y[t, ind_n1, :]*tau2_y[t, ind_n1, :] - tau1_y[t, ind_n1, :]**2)
            E_eta2_y = -(nu_y[t, ind_n1, :]*(nu_y[t, ind_n1, :] + 3))/(2*(nu_y[t, ind_n1, :]*tau2_y[t, ind_n1, :] - tau1_y[t, ind_n1, :]**2))
            est_mu_y[t, ind_n1, :] = -E_eta1_y/(2*E_eta2_y)
            y_hat[t, :] = y_psb[t, :]*np.sum(new_phi_x.reshape((N_zt, 1))*est_mu_y[t, ind_n1], axis = 0)
            b_hat[t, :] = sim_resp_fun(y_hat[t, :], y_psb[t, :], sim_pars['resp_scale']) # response

            # compute Eq[log p(y_n | z_n = t, eta)] (expected log-likelihood of y)
            est_sigma_y[t, ind_n1, :] = 1/np.sqrt(-2*E_eta2_y)
            Ell_outcomes = stats.norm.logpdf(y[t, :], est_mu_y[t, ind_n1], est_sigma_y[t, ind_n1])
            E_log_lik_y[t, ind_n1] = np.sum(y_psb[t, :]*Ell_outcomes, axis = 1) # assumed independent -> add log_lik across outcomes

            # compute phi (approximate posterior on cluster membership)
            s_xy = np.exp(E_log_lik_x[t, ind_n1] + E_log_lik_y[t, ind_n1] + E_log_prior[t, ind_n1])
            phi[t, ind_n1] = s_xy/s_xy.sum()
                
            # hard latent cause assignment
            z[t] = np.argmax(phi[t, :]) # winning (most probable) cluster
            z_onehot[t, z[t]] = 1
            if (z[t] == N[t]) and (N[t] < n_z):
                phi_learn = phi[t, :]
                N[t + 1] = N[t] + 1 # increase number of latent causes
            else:
                phi_learn = np.zeros(n_z)
                phi_learn[ind_n] = phi[t, ind_n]/phi[t, ind_n].sum() # drop new cause and re-normalize over old latent causes
                N[t + 1] = N[t]
                
            # learning (update hyperparameters)
            tau1_x[t + 1, :, :] = tau1_x[t, :, :] + x_sofar*np.outer(phi_learn, x[t, :])
            tau2_x[t + 1, :, :] = tau2_x[t, :, :] + x_sofar*np.outer(phi_learn, x[t, :]**2)
            nu_x[t + 1, :, :] = nu_x[t, :, :] + np.outer(phi_learn, x_sofar)
            tau1_y[t + 1, :, :] = tau1_y[t, :, :] + y_psb[t, :]*np.outer(phi_learn, y[t, :])
            tau2_y[t + 1, :, :] = tau2_y[t, :, :] + y_psb[t, :]*np.outer(phi_learn, y[t, :]**2)
            nu_y[t + 1, :, :] = nu_y[t, :, :] + np.outer(phi_learn, y_psb[t, :])
            n[t + 1, :] = n[t, :] + phi_learn
            
        # generate simulated responses
        (b, b_index) = resp_fun.generate_responses(b_hat, random_resp, trials.resp_type)
        
        # put all simulation data into a single xarray dataset
        ds = trials.copy(deep = True)
        ds = ds.assign_coords({'z_name' : np.array(range(n_z), dtype = str), 'ident' : [ident]})
        ds = ds.assign({'y_psb' : (['t', 'y_name'], y_psb),
                        'y_lrn' : (['t', 'y_name'], y_lrn),
                        'y_hat' : (['t', 'y_name'], y_hat),
                        'b_hat' : (['t', 'y_name'], b_hat),
                        'b' : (['t', 'y_name'], b),
                        'est_mu_x' : (['t', 'z_name', 'x_name'], est_mu_x),
                        'est_sigma_x' : (['t', 'z_name', 'x_name'], est_sigma_x),
                        'est_precision_x' : (['t', 'z_name', 'x_name'], 1/est_sigma_x**2),
                        'est_mu_y' : (['t', 'z_name', 'y_name'], est_mu_y),
                        'est_sigma_y' : (['t', 'z_name', 'y_name'], est_sigma_y),
                        'est_precision_y' : (['t', 'z_name', 'y_name'], 1/est_sigma_y**2),
                        'n' : (['t', 'z_name'], n[0:-1, :]),
                        'z' : (['t'], z),
                        'phi_x' : (['t', 'z_name'], phi_x),
                        'phi' : (['t', 'z_name'], phi),
                        'N' : (['t'], N[0:(t+1)]),
                        'E_log_prior': (['t', 'z_name'], E_log_prior),
                        'E_log_lik_x': (['t', 'z_name'], E_log_lik_x),
                        'E_log_lik_y': (['t', 'z_name'], E_log_lik_y)})
        ds = ds.assign_attrs({'model': self.name,
                              'model_class': 'latent_cause',
                              'sim_pars': sim_pars,
                              'n_z': n_z})
        return ds
    
    def particle_filter(self, trials, par_val = None, n_z = 10, n_p = 50, random_resp = False, ident = 'sim'):
        '''
        Simulate the model using a particle filter algorithm.
        
        Parameters
        ----------
        trials: dataset (xarray)
            Time step level experimental data (cues, outcomes etc.).

        par_val: list, optional
            Learning model parameters (floats or ints)
            
        n_z: int, optional
            Maximum number of latent causes.  Defaults to 10.
        
        n_p: int, optional
            Number of particles.  Defaults to 50.
            
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
            
        Explanation of variables in ds
        ------------------------------
        y_psb: indicator vector for outcomes (y) that are possible on the trial (from the learner's perspective)
        y_lrn: indicator vector for outcomes (y) for which there is feedback and hence learning will occur
        y_hat: outcome predictions
        b_hat: expected value of behavioral response
        b: vector representing actual behavioral response (identical to b_hat unless the random_resp argument is set to True)
        mean_N: mean number of latent causes per particle
        sd_N: standard deviation of number of latent causes per particle
        mean_ineq: mean of a rough measure of 'inequality' among latent causes (see code for details)
        b_index: index of behavioral response (only present if response type is 'choice' and random_resp is True)
        b_name: name of behavioral response (only present if response type is 'choice' and random_resp is True)
        
        Notes
        -----
        The particle filter algorithm is based on Gershman, Blei and Niv (2010); see the appendix of that paper.
        
        The marginal likelihood/posterior predictive calculation is based on
        https://en.wikipedia.org/wiki/Conjugate_prior#When_likelihood_function_is_a_continuous_distribution
        
        *** I should double check that this is correct. ***
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
        y = np.array(trials['y'], dtype = 'float64')
        y_psb = np.array(trials['y_psb'], dtype = 'float64')
        y_lrn = np.array(trials['y_lrn'], dtype = 'float64')
        x_names = list(trials.x_name.values)
        y_names = list(trials.y_name.values)
        n_t = x.shape[0] # number of time points
        n_x = x.shape[1] # number of stimulus attributes
        n_y = y.shape[1] # number of outcomes/response options
        y_hat = np.zeros((n_t, n_y)) # outcome predictions
        b_hat = np.zeros((n_t, n_y)) # expected behavior
        time = trials['time'].values # real world time (in arbitrary units, starting at 0)
        x_sofar = np.zeros(n_x) # keep track of cues (x) observed so far
        
        # data for particles (hyperparameters not tracked across time, i.e. those arrays represent current time point only)
        
        # prior for x parameters
        tau1_x = np.zeros((n_p, n_z, n_x))
        tau2_x = sim_pars['prior_tau2_x']*np.ones((n_p, n_z, n_x))
        nu_x = sim_pars['prior_nu_x']*np.ones((n_p, n_z, n_x))
        # prior for y parameters
        tau1_y = np.zeros((n_p, n_z, n_y))
        tau2_y = sim_pars['prior_tau2_y']*np.ones((n_p, n_z, n_y))
        nu_y = sim_pars['prior_nu_y']*np.ones((n_p, n_z, n_y))
        # other data
        z = np.zeros(n_p, dtype=int) # latent cause assigments
        z_onehot = np.zeros((n_p, n_t, n_z)) # one hot representation of z, i.e. winner is 1 and all others are 0
        n = np.zeros((n_p, n_z)) # estimated number of observations assigned to each latent cause
        N = np.ones(n_p, dtype=int) # estimated number of latent causes
        
        # summary statistics about latent causes across particles
        mean_N = np.zeros(n_t)
        sd_N = np.zeros(n_t)
        mean_ineq = np.zeros(n_t)
        
        # rng object (for resampling particles)
        rng = np.random.default_rng()
        
        # set up response function (depends on response type)
        resp_dict = {'choice': resp_fun.choice,
                     'exct': resp_fun.exct,
                     'supr': resp_fun.supr}
        sim_resp_fun = resp_dict[trials.resp_type]
        
        # run calculations for first time step
        x_sofar[x[0, :] > 0] = 1 # keep track of cues observed so far
        b_hat[0, :] = sim_resp_fun(y_hat[0, :], y_psb[0, :], sim_pars['resp_scale']) # response (y_hat initially is always 0)
        tau1_x[:, 0, :] += x_sofar*x[0, :]
        tau2_x[:, 0, :] += x_sofar*x[0, :]**2
        nu_x[:, 0, :] += x_sofar
        tau1_y[:, 0, :] += y_psb[0, :]*y[0, :]
        tau2_y[:, 0, :] += y_psb[0, :]*y[0, :]**2
        nu_y[:, 0, :] += y_psb[0, :]
        n[:, 0] += 1
        z_onehot[:, 0, 0] = 1
        z[:] = 0
               
        # loop through time steps
        for t in range(1, n_t):
            # preliminary stuff
            old_z = z # previous latent cause
            z = np.zeros(n_p, dtype=int) # current latent cause
            x_sofar[x[t, :] > 0] = 1 # keep track of cues observed so far
            y_hat_p = np.zeros((n_p, n_y)) # y predictions for each particle
            lik_x = np.zeros(n_p)
            lik_y = np.zeros(n_p)
            ineq = np.zeros(n_p) # rough measure of 'inequality' among latent causes
                        
            # loop through particles
            for p in range(n_p):
                # indices for latent causes, etc.
                if N[p] < n_z:
                    N_zt = N[p] + 1 # maximum number of latent causes considered this time step
                    ind_n = range(N[p])
                    ind_n1 = range(N[p] + 1)
                else:
                    N_zt = n_z
                    ind_n = range(N[p])
                    ind_n1 = ind_n
                ineq[p] = np.max(n[p, :])/(t + 1) # proportion of assignments to the latent cause that is active most often
                    
                # sample latent cause for upcoming time step
                K = self.kernel(t, N[p], time, sim_pars).reshape((N[p], t)) # temporal kernel (i.e. decay function for latent causes)
                r = np.sum(K*z_onehot[p, 0:t, ind_n], axis = 1) # recency
                num_prior = np.zeros(N_zt) # numerator of prior on latent causes
                num_prior[ind_n] = r[ind_n]
                num_prior[old_z[p]] += sim_pars['stick'] # add stickiness to most recent cluster
                if N[p] < n_z:
                    num_prior[N[p]] = sim_pars['alpha']
                prior = num_prior/num_prior.sum()
                z[p] = rng.choice(N_zt, p = prior)
                z_onehot[p, t, z[p]] = 1
                if (z[p] == N[p]) and (N[p] < n_z):
                    N[p] += 1
                
                # compute p(x_n | z_n = t, eta) (likelihood of x)
                df_x = nu_x[p, z[p], :] + 3
                my_x = tau1_x[p, z[p], :]/nu_x[p, z[p], :]
                beta_x = (nu_x[p, z[p], :]*tau2_x[p, z[p], :] - tau1_x[p, z[p], :]**2)/(2*nu_x[p, z[p], :])
                sigma_x = np.sqrt(2*beta_x/df_x)
                ll_x = stats.t.logpdf(x[t, :], df_x, my_x, sigma_x)
                lik_x[p] = np.exp(np.sum(x_sofar*ll_x)) # assumed independent -> add log_lik across cues

                # predict y
                my_y = tau1_y[p, z[p], :]/nu_y[p, z[p], :]
                y_hat_p[p, :] = y_psb[t, :]*my_y

                # compute p(y_n | z_n = t, eta) (likelihood of y)
                df_y = nu_y[p, z[p], :] + 3
                beta_y = (nu_y[p, z[p], :]*tau2_y[p, z[p], :] - tau1_y[p, z[p], :]**2)/(2*nu_y[p, z[p], :])
                sigma_y = np.sqrt(2*beta_y/df_y)
                ll_y = stats.t.logpdf(y[t, :], df_y, my_y, sigma_y)
                lik_y[p] = np.exp(np.sum(y_psb[t, :]*ll_y)) # assumed independent -> add log_lik across outcomes
                
                # learning (update hyperparameters)
                tau1_x[p, z[p], :] += x_sofar*x[t, :]
                tau2_x[p, z[p], :] += x_sofar*x[t, :]**2
                nu_x[p, z[p], :] += x_sofar
                tau1_y[p, z[p], :] += y_psb[t, :]*y[t, :]
                tau2_y[p, z[p], :] += y_psb[t, :]*y[t, :]**2
                nu_y[p, z[p], :] += y_psb[t, :]
                n[p, z[p]] += 1
                
            # after looping through particles, average their predictions together and compute b_hat
            pred_weights = lik_x/lik_x.sum()
            y_hat[t, :] = np.mean(y_hat_p*np.repeat(pred_weights, n_y).reshape((n_p, n_y)), axis = 0)
            b_hat[t, :] = sim_resp_fun(y_hat[t, :], y_psb[t, :], sim_pars['resp_scale']) # response
            
            # record summary statistics about latent causes across particles
            mean_N[t] = np.mean(N) # mean number of latent causes per particle
            sd_N[t] = np.std(N) # standard deviation of number of latent causes per particle
            mean_ineq[t] = np.mean(ineq) # mean of a rough measure of 'inequality' among latent causes
            
            # prior to next time step, resample particles according to the x, y likelihood
            sample_weights = lik_x*lik_y/np.sum(lik_x*lik_y)
            new_p = rng.choice(n_p, size = n_p, replace = True, p = sample_weights)
            z = z[new_p]
            z_onehot = z_onehot[new_p, :, :]
            N = N[new_p]
            n = n[new_p, :]
            tau1_x = tau1_x[new_p, :, :]
            tau2_x = tau2_x[new_p, :, :]
            nu_x = nu_x[new_p, :, :]
            tau1_y = tau1_y[new_p, :, :]
            tau2_y = tau2_y[new_p, :, :]
            nu_y = nu_y[new_p, :, :]
            
        # generate simulated responses
        (b, b_index) = resp_fun.generate_responses(b_hat, random_resp, trials.resp_type)
        
        # put all simulation data into a single xarray dataset
        ds = trials.copy(deep = True)
        ds = ds.assign_coords({'z_name' : np.array(range(n_t), dtype = str), 'ident' : [ident]})
        ds = ds.assign({'y_psb' : (['t', 'y_name'], y_psb),
                        'y_lrn' : (['t', 'y_name'], y_lrn),
                        'y_hat' : (['t', 'y_name'], y_hat),
                        'b_hat' : (['t', 'y_name'], b_hat),
                        'b' : (['t', 'y_name'], b),
                        'mean_N': ('t', mean_N),
                        'sd_N': ('t', sd_N),
                        'mean_ineq': ('t', mean_ineq)})
        ds = ds.assign_attrs({'model': self.name,
                              'model_class': 'latent_cause',
                              'sim_pars': sim_pars,
                              'n_z': n_z,
                              'n_p': n_p})
        return ds

########## PARAMETERS ##########
# Note: allowing prior_a to be close to 1 seems to cause problems.
par_names = []; par_list = []                         
par_names += ['gamma']; par_list += [{'min': 0.0, 'max': 5.0, 'default': 1.0, 'description': 'decay rate for exponential SCRP; higher -> favors more recent latent causes'}]
par_names += ['power']; par_list += [{'min': 0.0, 'max': 5.0, 'default': 1.0, 'description': 'decay rate for power law SCRP; higher -> favors more recent latent causes'}]
par_names += ['alpha']; par_list += [{'min': 0.0, 'max': 15.0, 'default': 1.0, 'description': 'concentration parameter; higher -> tend to infer more latent causes'}]
par_names += ['prior_tau2_x']; par_list += [{'min': 0.01, 'max': 10.0, 'default': 1.0, 'description': 'prior hyperparameter for eta for x'}]
par_names += ['prior_nu_x']; par_list += [{'min': 1.0, 'max': 10.0, 'default': 5.0, 'description': 'prior hyperparameter for eta for x'}]
par_names += ['prior_tau2_y']; par_list += [{'min': 0.01, 'max': 10.0, 'default': 1.0, 'description': 'prior hyperparameter for eta for y'}]
par_names += ['prior_nu_y']; par_list += [{'min': 1.0, 'max': 10.0, 'default': 5.0, 'description': 'prior hyperparameter for eta for y'}]
par_names += ['stick']; par_list += [{'min': -5.0, 'max': 5.0, 'default': 1.0, 'description': 'stickiness for CRP prior'}]
par_names += ['window']; par_list += [{'min': 0.0, 'max': 1000.0, 'default': 100.0, 'description': 'window determining refractory period for kernel'}]
par_names += ['kernel_asymptote']; par_list += [{'min': 0.0, 'max': 2.0, 'default': 0.5, 'description': 'asymptote for kernel'}]
par_names += ['resp_scale']; par_list += [{'min': 0.0, 'max': 10.0, 'default': 1.0, 'description': 'scales softmax/logistic response functions'}]

pars = pd.DataFrame(par_list, index = par_names)
del par_names; del par_list