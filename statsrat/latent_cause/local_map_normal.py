import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats
from statsrat import resp_fun
from . import kernel

# https://github.com/LukasNeugebauer/LCM_python/blob/master/LCM.py
# https://github.com/sjgershm/LCM

class model:
    '''
    Local MAP approximation with normal distribution.
    '''
    def __init__(self, name, kernel):
        '''
        Parameters
        ----------
        '''
        self.name = name
        self.kernel = kernel
        # determine the model's parameter space
        self.par_names = kernel.par_names + ['prior_tau1_x', 'prior_nu_x', 'prior_tau1_y', 'prior_nu_y', 'stick', 'alpha', 'resp_scale']
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
        
        # prior for x parameters
        tilde_nu_x = sim_pars['prior_nu_x']
        tau0_x = np.zeros((n_t + 1, n_t + 1, n_x)) # corresponds to kappa (mean of normal-gamma) times n
        tau1_x = sim_pars['prior_tau1_x']*np.ones((n_t + 1, n_t + 1, n_x)) # corresponds to beta/2
        # prior for y parameters
        tilde_nu_y = sim_pars['prior_nu_y']
        tau0_y = np.zeros((n_t + 1, n_t + 1, n_u)) # corresponds to kappa (mean of normal-gamma) times n
        tau1_y = sim_pars['prior_tau1_y']*np.ones((n_t + 1, n_t + 1, n_u)) # corresponds to beta/2
        
        log_prior = np.zeros((n_t, n_t))
        log_lik_x = np.zeros((n_t, n_t))
        log_lik_y = np.zeros((n_t, n_t))
        E_x = np.zeros((n_t, n_t, n_x))
        V_x = np.zeros((n_t, n_t, n_x))
        E_y = np.zeros((n_t, n_t, n_u))
        V_y = np.zeros((n_t, n_t, n_u))
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
        tau0_x[1, 0, :] = tau0_x[0, 0, :] + x_sofar*x[0, :]
        tau1_x[1, 0, :] = tau1_x[0, 0, :] + x_sofar*(x[0, :]**2)
        tau0_y[1, 0, :] = tau0_y[0, 0, :] + u_psb[0, :]*u[0, :]
        tau1_y[1, 0, :] = tau1_y[0, 0, :] + u_psb[0, :]*(u[0, :]**2)
        n[1, 0] = n[0, 0] + 1
        z[0] = 0
        z_onehot[0, 0] = 1
        N[0] = 1
               
        # loop through time steps
        for t in range(1, n_t):
            # I AM NOT CONFIDENT THAT THE POSTERIOR PREDICTIVE (I.E. MARGINAL LIKELIHOOD) FORMULA IS CORRECT.
            
            # preliminary stuff
            x_sofar[x[t, :] > 0] = 1 # keep track of cues observed so far
            ind_n = range(N[t]) # index for latent causes
            ind_n1 = range(N[t] + 1) # index latent causes
            nu_x = np.repeat(n[t, ind_n1] + tilde_nu_x, n_x).reshape((N[t] + 1, n_x)) # used in computations involving x
            nu_y = np.repeat(n[t, ind_n1] + tilde_nu_y, n_u).reshape((N[t] + 1, n_u)) # used in computations involving y
            
            # compute log p(x_n | z_n = t, eta) (log-likelihood of x)
            E_x[t, ind_n1, :] = tau0_x[t, ind_n1, :]/nu_x
            alpha_x = nu_x/2 + 1
            beta_x = tau1_x[t, ind_n1, :]/2
            df_x = 2*alpha_x # posterior predictive degrees of freedom
            V_x[t, ind_n1, :] = (beta_x*(nu_x + 1))/(alpha_x*nu_x) # posterior predictive variance
            ll_cues = stats.t.logpdf(x[t, :], df_x, E_x[t, ind_n1, :], np.sqrt(V_x[t, ind_n1, :]))
            log_lik_x[t, ind_n1] = np.sum(x_sofar*ll_cues, axis = 1) # assumed independent -> add log_lik across cues
            
            # compute log p(z_n = t | z_1, ..., z_{n-1}) (log-prior)
            K = self.kernel(t, time, sim_pars).reshape((t, 1)) # temporal kernel (i.e. decay function for latent causes)
            r = np.sum(K*z_onehot[0:t, ind_n], axis = 0) # recency
            sum_r = np.sum(r)
            numerator = r[ind_n]
            numerator[z[t-1]] += sim_pars['stick'] # add stickiness to most recent cluster
            log_denominator = np.log(sum_r + sim_pars['stick'] + sim_pars['alpha'])
            log_prior[t, ind_n] = np.log(numerator) - log_denominator # old clusters
            log_prior[t, N[t]] = np.log(sim_pars['alpha']) - log_denominator # new cluster

            # compute log_lik_phi based on x
            s = np.exp(log_lik_x[t, ind_n1] + log_prior[t, ind_n1])
            new_phi_x = s/s.sum()
            phi_x[t, ind_n] = new_phi_x[ind_n]
                                           
            # predict y (recall that 'y' = 'u')
            E_y[t, ind_n1, :] = tau0_y[t, ind_n1, :]/nu_y # posterior predictive mean (refer to my notes) # CHECK THAT THIS IS CORRECT.
            u_hat[t, :] = u_psb[t, :]*np.sum(new_phi_x.reshape((N[t] + 1, 1))*E_y[t, ind_n1, :], axis = 0) # predicted outcome (u)
            b_hat[t, :] = sim_resp_fun(u_hat[t, :], u_psb[t, :], sim_pars['resp_scale']) # response

            # compute log p(y_n | z_n = t, eta) (log-likelihood of y)
            alpha_y = nu_y/2 + 1
            beta_y = tau1_y[t, ind_n1, :]/2
            df_y = 2*alpha_y # posterior predictive degrees of freedom
            V_y[t, ind_n1, :] = (beta_y*(nu_y + 1))/(alpha_y*nu_y) # posterior predictive variance
            ll_outcomes = stats.t.logpdf(u[t, :], df_y, E_y[t, ind_n1, :], np.sqrt(V_y[t, ind_n1, :]))
            log_lik_y[t, ind_n1] = np.sum(u_psb[t, :]*ll_outcomes, axis = 1) # assumed independent -> add log_lik across outcomes

            # compute phi
            s_xy = np.exp(log_lik_x[t, ind_n1] + log_lik_y[t, ind_n1] + log_prior[t, ind_n1])
            phi[t, ind_n1] = s_xy/s_xy.sum()
                
            # hard latent cause assignment
            z[t] = np.argmax(phi[t, :]) # winning (most probable) cluster
            z_onehot[t, z[t]] = 1
            if z[t] == N[t]:
                N[t + 1] = N[t] + 1 # increase number of latent causes
            else:
                N[t + 1] = N[t]

            # learning (update hyperparameters for winning cluster)
            tau0_x[t + 1, :, :] = tau0_x[t, :, :]
            tau1_x[t + 1, :, :] = tau1_x[t, :, :]
            tau0_y[t + 1, :, :] = tau0_y[t, :, :]
            tau1_y[t + 1, :, :] = tau1_y[t, :, :]
            n[t + 1, :] = n[t, :]
            tau0_x[t + 1, z[t], :] += x_sofar*x[t, :]
            tau1_x[t + 1, z[t], :] += x_sofar*(x[t, :]**2)
            tau0_y[t + 1, z[t], :] += u_psb[t, :]*u[t, :]
            tau1_y[t + 1, z[t], :] += u_psb[t, :]*(u[t, :]**2)
            n[t + 1, z[t]] += 1
            
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
                        'tau0_x' : (['t', 'z_name', 'x_name'], tau0_x[0:-1, 0:-1, :]),
                        'tau1_x' : (['t', 'z_name', 'x_name'], tau1_x[0:-1, 0:-1, :]),
                        'tau0_y' : (['t', 'z_name', 'u_name'], tau0_y[0:-1, 0:-1, :]),
                        'tau1_y' : (['t', 'z_name', 'u_name'], tau1_y[0:-1, 0:-1, :]),
                        'n' : (['t', 'z_name'], n[0:-1, 0:-1]),
                        'z' : (['t'], z),
                        'E_x' : (['t', 'z_name', 'x_name'], E_x),
                        'V_x' : (['t', 'z_name', 'x_name'], V_x),
                        'E_y' : (['t', 'z_name', 'u_name'], E_y),
                        'V_y' : (['t', 'z_name', 'u_name'], V_y),
                        'phi_x' : (['t', 'z_name'], phi_x[:, 0:-1]),
                        'phi' : (['t', 'z_name'], phi[:, 0:-1]),
                        'N' : (['t'], N[0:(t+1)]),
                        'log_prior': (['t', 'z_name'], log_prior),
                        'log_lik_x': (['t', 'z_name'], log_lik_x),
                        'log_lik_y': (['t', 'z_name'], log_lik_y)})
        ds = ds.assign_attrs({'model': self.name,
                              'model_class': 'latent_cause',
                              'sim_pars': sim_pars})        
        return ds

########## PARAMETERS ##########
# Note: allowing prior_a to be close to 1 seems to cause problems.
par_names = []; par_list = []                         
par_names += ['gamma']; par_list += [{'min': 0.0, 'max': 5.0, 'default': 0.5, 'description': 'decay rate for exponential SCRP; higher -> favors more recent latent causes'}] 
par_names += ['alpha']; par_list += [{'min': 0.0, 'max': 15.0, 'default': 1.0, 'description': 'concentration parameter; higher -> tend to infer more latent causes'}]
par_names += ['prior_tau1_x']; par_list += [{'min': 0.01, 'max': 0.5, 'default': 0.2, 'description': 'prior hyperparameter for eta for x'}]
par_names += ['prior_nu_x']; par_list += [{'min': 0.01, 'max': 10.0, 'default': 5.0, 'description': 'prior hyperparameter for eta for x'}]
par_names += ['prior_tau1_y']; par_list += [{'min': 0.01, 'max': 0.5, 'default': 0.2, 'description': 'prior hyperparameter for eta for y'}]
par_names += ['prior_nu_y']; par_list += [{'min': 0.01, 'max': 10.0, 'default': 5.0, 'description': 'prior hyperparameter for eta for y'}]
par_names += ['stick']; par_list += [{'min': 0.0, 'max': 5.0, 'default': 1.0, 'description': 'stickiness for CRP prior'}]
par_names += ['resp_scale']; par_list += [{'min': 0.0, 'max': 10.0, 'default': 1.0, 'description': 'scales softmax/logistic response functions'}]

pars = pd.DataFrame(par_list, index = par_names)
del par_names; del par_list