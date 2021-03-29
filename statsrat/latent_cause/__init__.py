import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats
from scipy.special import digamma
from statsrat import resp_fun
from . import kernel

class model:
    '''
    Class for latent cause models.
    
    Attributes
    ----------
    name: str
        Model name.
    par_names: list
        Names of the model's free parameters (strings).
          
    Methods
    -------
    simulate(trials, resp_type = 'choice', par_val = None, random_resp = False, ident = 'sim', max_z = 15)
        Simulate a trial sequence once with known model parameters.
        
    Notes
    -----
    To start with (temporarily), this assumes that both x and u have Bernoulli likelihoods.
    UPDATE THESE NOTES
    
    Both the outcome vector (u) and stimulus vector (x) are assumed to be
    determined by a latent cause (z).  Only one latent cause is active on each
    trial.  Instead of local MAP approximation (Anderson, 1991; Gerhsman et al, 2017)
    or a particle filter (Gershman et al, 2010) to simulate latent cause models, statsrat
    uses a streaming variational Bayes algorithm (Blei & Jordan, 2006; Broderick et al)

    Anderson, J. R. (1991). The adaptive nature of human categorization. Psychological Review, 98(3), 409.
    Blei, D. M., & Jordan, M. I. (2006). Variational inference for Dirichlet process mixtures. Bayesian Analysis, 1(1), 121–143. https://doi.org/10.1214/06-BA104
    Broderick, T., Boyd, N., Wibisono, A., Wilson, A. C., & Jordan, M. I. (n.d.). Streaming Variational Bayes.
    Gershman, S. J., Blei, D. M., & Niv, Y. (2010). Context, learning, and extinction. Psychological Review, 117(1), 197–209.
    Gershman, S. J., Monfils, M.-H., Norman, K. A., & Niv, Y. (2017). The computational nature of memory modification. Elife, 6, e23763.

    '''
    def __init__(self, kernel):
        '''
        Parameters
        ----------
        '''
        self.name = 'basic (Bernoulli)'
        self.kernel = kernel
        # determine the model's parameter space
        par_names = kernel.par_names + ['prior_a_x', 'prior_b_x', 'prior_a_y', 'prior_b_y']
        self.pars = pars.loc[par_names + ['alpha', 'resp_scale']]
        
    def simulate(self, trials, par_val = None, n_iter = 5, random_resp = False, ident = 'sim'):
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
        
        UPDATE THESE NOTES IF NEEDED
        
        Likelihood computations only use stimulus attributes (elements of x) and 
        outcomes (elements of u) that the learner has observed up to that point in 
        the learning process.  That way, model computations cannot be affected by 
        future events (stimulus attributes that haven't yet been encountered).  This
        is important for between groups simulations in which each group's schedule has
        a different set of stimulus attributes or possible outcomes.
        
        Similarly, likelihood computations only use outcomes (elements of u) that
        the learner believes are possible at a given stage.
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
        
        # prior for parameters for x
        tilde_tau_x = sim_pars['prior_a_x'] - 1
        tilde_n_x = sim_pars['prior_a_x'] + sim_pars['prior_b_x'] - 2
        tau_x = tilde_tau_x*np.ones((n_t + 1, n_t + 1, n_x)) # natural hyperparameters of outcome distribution
        # prior for parameters for y
        tilde_tau_y = sim_pars['prior_a_y'] - 1
        tilde_n_y = sim_pars['prior_a_y'] + sim_pars['prior_b_y'] - 2
        tau_y = tilde_tau_y*np.ones((n_t + 1, n_t + 1, n_u)) # natural hyperparameters of stimulus distribution
        
        n = np.zeros((n_t + 1, n_t + 1)) # estimated number of observations assigned to each latent cause
        N = np.zeros(n_t + 1, dtype=int) # estimated number of latent causes
        N[[0, 1]] = 1
        phi_x = np.zeros((n_t, n_t + 1)) # posterior of latent causes after observing x, but before observing u
        phi = np.zeros((n_t, n_t + 1)) # posterior of latent causes after observing both x and y (i.e. u)
        x_sofar = np.zeros(n_x) # keep track of cues (x) observed so far
        
        E_r = np.zeros((n_t, n_t)) # mean recency
        V_r = np.zeros((n_t, n_t)) # variance of recency
        sum_r = 0 # sum of recencies across latent causes
                         
        # set up response function (depends on response type)
        resp_dict = {'choice': resp_fun.choice,
                     'exct': resp_fun.exct,
                     'supr': resp_fun.supr}
        sim_resp_fun = resp_dict[trials.resp_type]
        
        # run calculations for first time step
        x_sofar[x[0, :] > 0] = 1 # keep track of cues observed so far
        E_post_pred = (tau_y[0, 0, :] + 1)/(tilde_n_y + 2) # mean of posterior predictive
        u_hat[0, :] = u_psb[0, :]*E_post_pred # predicted outcome (u)
        b_hat[0, :] = sim_resp_fun(u_hat[0, :], u_psb[0, :], sim_pars['resp_scale']) # response
        phi_x[0, 0] = 1
        phi[0, 0] = 1
        T_x = x[0, :] # sufficient statistic (T(x))
        tau_x[1, 0, :] = tau_x[0, 0, :] + x_sofar*T_x
        T_y = u[0, :] # sufficient statistic (T(y))
        tau_y[1, 0, :] = tau_y[0, 0, :] + u_psb[0, :]*T_y
        n[1, 0] = n[0, 0] + 1
               
        # loop through time steps
        for t in range(1, n_t):
            # preliminary stuff
            x_sofar[x[t, :] > 0] = 1 # keep track of cues observed so far
            ind_n = range(N[t]) # index for latent causes
            ind_n1 = range(N[t] + 1) # index latent causes
            n_for_x = np.repeat(n[t, ind_n1] + tilde_n_x, n_x).reshape((N[t] + 1, n_x)) # used in computations involving x
            n_for_y = np.repeat(n[t, ind_n1] + tilde_n_y, n_u).reshape((N[t] + 1, n_u)) # used in computations involving y (i.e. u)
            
            # compute Eq[log p(x_n | z_n = t, eta)] (expected log-likelihood of x)
            E_eta_x = digamma(tau_x[t, ind_n1, :]) - digamma(n_for_x - tau_x[t, ind_n1, :] + 1) # expected natural parameter (eta)
            E_a_eta_x = digamma(n_for_x - tau_x[t, ind_n1, :] + 1) - digamma(n_for_x + 2) # expected log partition function (a(eta))
            b_x = 0 # log base measure (b(x))
            T_x = x[t, :] # sufficient statistic (T(x))
            Ell_cues = E_eta_x*T_x - E_a_eta_x - b_x # expected log likelihood for each cue
            E_log_lik_x = np.sum(x_sofar*Ell_cues, axis = 1) # assumed independent -> add log_lik across cues
            
            # approximate Eq[log p(z_n = t | z_1, ..., z_{n-1})] (expected log-prior)
            K = self.kernel(t, time, sim_pars) # temporal kernel (i.e. decay function for latent causes)
            E_r[t, ind_n] = np.sum(K.reshape((t, 1))*phi[0:t, ind_n], axis = 0)
            V_r[t, ind_n] = np.sum((K**2).reshape((t, 1))*phi[0:t, ind_n]*(1 - phi[0:t, ind_n]), axis = 0)
            sum_r = np.sum(K)
            E_log_prior = np.zeros(N[t] + 1)
            E_log_prior[ind_n] = np.log(E_r[t, ind_n]) - 0.5*V_r[t, ind_n]/(E_r[t, ind_n]**2) - np.log(sum_r + sim_pars['alpha'])
            E_log_prior[N[t]] = np.log(sim_pars['alpha']) - np.log(sum_r + sim_pars['alpha'])

            # compute E_log_lik_phi based on x
            s = np.exp(E_log_lik_x + E_log_prior)
            new_phi_x = s/s.sum()
            phi_x[t, ind_n] = new_phi_x[ind_n]
                                           
            # predict y (recall that 'y' = 'u')
            n_for_calc = n[t, ind_n1].reshape((N[t] + 1, 1)) + tilde_n_y
            E_post_pred = (tau_y[t, ind_n1, :] + 1)/(n_for_calc + 2) # mean of posterior predictive
            u_hat[t, :] = u_psb[t, :]*np.sum(new_phi_x.reshape((N[t] + 1, 1))*E_post_pred, axis = 0) # predicted outcome (u)
            b_hat[t, :] = sim_resp_fun(u_hat[t, :], u_psb[t, :], sim_pars['resp_scale']) # response
            
            # loop through iterations
            new_tau_x = tau_x[t, ind_n1, :]
            new_tau_y = tau_y[t, ind_n1, :]
            for i in range(n_iter):
                # compute Eq[log p(x_n | z_n = t, eta)] (expected log-likelihood of x)
                E_eta_x = digamma(tau_x[t, ind_n1, :]) - digamma(n_for_x - new_tau_x + 1) # expected natural parameter (eta)
                E_a_eta_x = digamma(n_for_x - new_tau_x + 1) - digamma(n_for_x + 2) # expected log partition function (a(eta))
                b_x = 0 # log base measure (b(x))
                T_x = x[t, :] # sufficient statistic (T(x))
                Ell_cues = E_eta_x*T_x - E_a_eta_x - b_x # expected log likelihood for each cue
                E_log_lik_x = np.sum(x_sofar*Ell_cues, axis = 1) # assumed independent -> add log_lik across cues
            
                # compute Eq[log p(y_n | z_n = t, eta)] (expected log-likelihood of y)
                E_eta_y = digamma(new_tau_y) - digamma(n_for_y - new_tau_y + 1) # expected natural parameter
                E_a_eta_y = digamma(n_for_y - new_tau_y + 1) - digamma(n_for_y + 2) # expected log partition function
                b_y = 0 # log base measure (b(y))
                T_y = u[t, :] # sufficient statistic (T(y))
                Ell_outcomes = E_eta_y*T_y - E_a_eta_y - b_y # expected log likelihood for each outcome
                E_log_lik_y = np.sum(u_psb[t, :]*Ell_outcomes, axis = 1) # assumed independent -> add log_lik across outcomes
            
                # compute phi
                s = np.exp(E_log_lik_x + E_log_lik_y + E_log_prior)
                new_phi = s/s.sum()

                # learning (update hyperparameters)
                new_tau_x = tau_x[t, ind_n1, :] + x_sofar*np.outer(new_phi, T_x)
                new_tau_y = tau_y[t, ind_n1, :] + u_psb[t, :]*np.outer(new_phi, T_y)
                new_n = n[t, ind_n1] + new_phi
                n_for_x = np.repeat(new_n + tilde_n_x, n_x).reshape((N[t] + 1, n_x))
                n_for_y = np.repeat(new_n + tilde_n_y, n_u).reshape((N[t] + 1, n_u))
            
            # add latent cause (expand N) if needed
            if new_phi[N[t]] == new_phi.max():
                N[t + 1] = N[t] + 1
            else:
                N[t + 1] = N[t]
            
            # perform permanent updates/record computations
            ind_lrn = range(N[t + 1])
            phi[t, ind_lrn] = new_phi[ind_lrn]
            tau_x[t + 1, ind_lrn, :] = new_tau_x[ind_lrn, :]
            tau_y[t + 1, ind_lrn, :] = new_tau_y[ind_lrn, :]
            n[t + 1, ind_lrn] = new_n[ind_lrn]
            #tau_x[t + 1, ind_lrn, :] = tau_x[t, ind_lrn, :] + x_sofar*np.outer(phi[t, ind_lrn], T_x)
            #tau_y[t + 1, ind_lrn, :] = tau_y[t, ind_lrn, :] + u_psb[t, :]*np.outer(phi[t, ind_lrn], T_y)
            #n[t + 1, ind_lrn] = n[t, ind_lrn] + phi[t, ind_lrn]
            
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
                        'tau_x' : (['t', 'z_name', 'x_name'], tau_x[0:-1, 0:-1, :]),
                        'tau_y' : (['t', 'z_name', 'u_name'], tau_y[0:-1, 0:-1, :]),
                        'n' : (['t', 'z_name'], n[0:-1, 0:-1]),
                        'E_r' : (['t', 'z_name'], E_r),
                        'V_r' : (['t', 'z_name'], V_r),
                        'phi_x' : (['t', 'z_name'], phi_x[:, 0:-1]),
                        'phi' : (['t', 'z_name'], phi[:, 0:-1]),
                        'N' : (['t'], N[0:(t+1)])})
        ds = ds.assign_attrs({'model': self.name,
                              'model_class': 'latent_cause',
                              'sim_pars': sim_pars})
        # add in conventional parameter estimates to dataset
        a_x = tau_x[0:-1, 0:-1, :] + 1
        b_x = np.repeat(n[0:-1, 0:-1] + tilde_n_x, n_x).reshape((n_t, n_t, n_x)) - tau_x[0:-1, 0:-1, :] + 1
        a_y = tau_y[0:-1, 0:-1, :] + 1
        b_y = np.repeat(n[0:-1, 0:-1] + tilde_n_y, n_u).reshape((n_t, n_t, n_u)) - tau_y[0:-1, 0:-1, :] + 1
        E_theta_x = a_x/(a_x + b_x) # mean conventional parameter (probability) for x
        E_theta_y = a_y/(a_y + b_y) # mean conventional parameter (probability) for y
        ds = ds.assign({'E_theta_x': (['t', 'z_name', 'x_name'], E_theta_x), 'E_theta_y': (['t', 'z_name', 'y_name'], E_theta_y)})
        
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