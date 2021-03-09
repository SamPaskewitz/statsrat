import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats
from scipy.special import digamma
from statsrat import resp_fun

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
    def __init__(self):
        '''
        Parameters
        ----------
        '''
        self.name = 'basic (Bernoulli)'
        # determine the model's parameter space
        par_names = [''] # SPECIFY BETA HYPERPARAMETERS
        self.pars = pars.loc[par_names + ['gamma', 'alpha', 'resp_scale']]
        
    def simulate(self, trials, resp_type = 'choice', par_val = None, random_resp = False, ident = 'sim'):
        '''
        Simulate a trial sequence once with known model parameters.
        
        Parameters
        ----------
        trials: dataset (xarray)
            Time step level experimental data (cues, outcomes etc.).

        resp_type: str, optional
            Type of behavioral response: one of 'choice', 'exct' or 'supr'.
            Defaults to 'choice'.

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
        max_z = n_t # maximum number of latent causes is equal to number of time points
        n_x = x.shape[1] # number of stimulus attributes
        n_u = u.shape[1] # number of outcomes/response options
        u_hat = np.zeros((n_t, n_u)) # outcome predictions
        b_hat = np.zeros((n_t, n_u)) # expected behavior
        
        tau_u = np.zeros((n_t, n_x, max_z) # natural hyperparameters of outcome distribution
        tau_x = np.zeros((n_t, n_u, max_z)) # natural hyperparameters of stimulus distribution
        n = np.zeros((n_t, max_z)) # estimated number of observations assigned to each latent cause
        N = 1 # estimated number of latent causes
                         
        E_log_lik_x = np.zeros((n_t, max_z)) # expected log-likelihood of x
        E_log_lik_y = np.zeros((n_t, max_z)) # expected log-likelihood of y (i.e. of u)
        E_log_prior = np.zeros((n_t, max_z)) # expected log-prior on z
        
        phi_x = np.zeros((n_t, max_z)) # posterior of latent causes after observing x, but before observing u
        phi = np.zeros((n_t, max_z)) # posterior of latent causes after observing both x and y (i.e. u)
        
        # FIGURE OUT INITIALIZATION
        E_r = # mean recency
        V_r = # variance of recency
        sum_r = # sum of recencies across latent causes
        
        # set up response function (depends on response type)
        # UPDATE THIS TO READ FROM TRIALS OBJECT
        resp_dict = {'choice': resp_fun.choice,
                     'exct': resp_fun.exct,
                     'supr': resp_fun.supr}
        sim_resp_fun = resp_dict[resp_type]
        
        # loop through time steps
        for t in range(n_t):
            # compute Eq[log p(x_n | z_n = t, eta)] (expected log-likelihood of x)
            x_sofar[x[t, :]] = 1 # keep track of cues observed so far
            E_eta_x = digamma(tau_x[t, :, :]) - digamma(n[t, :] - tau_x[t, :, :] + 1) # expected natural parameter (eta)
            E_a_eta_x = digamma(n[t, :] - tau_x[t, :, :] + 1) - digamma(n[t, :] + 2) # expected log partition function (a(eta))
            b_x = 0 # log base measure (b(x))
            T_x = x[t, :] # sufficient statistic (T(x))
            foo = E_eta_x*T_x - E_a_eta_x - b_x
            E_log_lik_x[t, range(N + 1)] = np.sum(foo, axis = 0) # cues assumed independent -> add log_lik across cues
            
            # approximate Eq[log p(z_n = t | z_1, ..., z_{n-1})] (expected log-prior)
            E_log_prior[t, range(N)] = np.log(E_r[range(N)] - V_r[range(N)]/(2*(E_r[range(N)])**2)) - np.log(sum_r + sim_pars['alpha'])
            E_log_prior[t, N + 1] = np.log(sim_pars['alpha'] - np.log(sum_r + sim_pars['alpha'])
            
            # compute phi based on x
            s = np.exp(E_log_lik_x + E_log_prior)
            phi_x[t, :] = s/s.sum()
                                           
            # predict y ('y' = 'u') 
            E_post_pred = (tau[t, :] + 1)/(n[t, :] + 2) # mean of posterior predictive
            u_hat[t, :] = np.sum(phi_x[t, :]*E_post_pred*u_psb[t, :], axis = 0) # predicted outcome (u)
            b_hat[t, :] = sim_resp_fun(u_hat[t, :], u_psb[t, :], sim_pars['resp_scale']) # response
                                           
            # compute Eq[log p(y_n | z_n = t, eta)] (expected log-likelihood of y)
            E_eta_y = digamma(tau_y[t, :, :]) - digamma(n[t, :] - tau_y[t, :, :] + 1) # expected natural parameter (eta)
            E_a_eta_y = digamma(n[t, :] - tau_y[t, :] + 1) - digamma(n[t, :] + 2) # expected log partition function (a(eta))
            b_y = 0 # log base measure (b(y))
            T_y = y[t, :] # sufficient statistic (T(y))
            bar = E_eta_y*T_y - E_a_eta_y - b_y
            E_log_lik_y[t, range(N + 1)] = np.sum(u_psb[t, :]*bar, axis = 0) # outcomes assumed independent -> add log_lik across outcomes
                                           
            # update phi based on y
            s *= np.exp(E_log_lik_y)
            phi[t, :] = s/s.sum()
                                           
            # learning (update hyperparameters)                        
            tau_x[t, :, :] = tau_x[t, :, :] + phi[t, :]*T_x
            tau_y[t, :, :] = tau_y[t, :, :] + phi[t, :]*T_y
            n[t, :] = n[t, :] + phi[t, :]
                                           
            # add latent cause (expand N) if needed
            if n[N + 1] > 1:
                N += 1
                # INITIALIZE STUFF AS APPROPRIATE
           
            # update E_r, V_r and sum_r
            E_r = np.exp(-sim_pars['gamma'])*(phi[t, :] + E_r)
            V_r = np.exp(-2*sim_pars['gamma'])*(phi[t, :]*(1 - phi[t, :]) + V_r)
            sum_r = np.exp(-sim_pars['gamma'])*(1 + sum_r)
                                                       
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
        # UPDATE THIS
        ds = xr.Dataset(data_vars = {'x' : (['t', 'x_name'], x),
                                     'u' : (['t', 'u_name'], u),
                                     'u_psb' : (['t', 'u_name'], u_psb),
                                     'u_lrn' : (['t', 'u_name'], u_lrn),
                                     'u_hat' : (['t', 'u_name'], u_hat),
                                     'b_hat' : (['t', 'u_name'], b_hat),
                                     'b' : (['t', 'u_name'], b),
                                     'prior' : (['t', 'z_name'], prior),
                                     'post_x' : (['t', 'z_name'], post_x),
                                     'post_xu' : (['t', 'z_name'], post_xu),
                                     'x_lik' : (['t', 'z_name'], x_lik),
                                     'u_lik' : (['t', 'z_name'], u_lik),
                                     'z' : (['t', 'z_name'], z),
                                     'z_counts' : (['z_name'], z_counts)},
                        coords = {'t' : range(n_t),
                                  't_name' : ('t', trials.t_name),
                                  'trial' : ('t', trials.trial),
                                  'trial_name' : ('t', trials.trial_name),
                                  'stage' : ('t', trials.stage),
                                  'stage_name' : ('t', trials.stage_name),
                                  'x_name' : x_names,
                                  'u_name' : u_names,
                                  'z_name' : np.array(range(max_z), dtype = 'str'),
                                  'ident' : [ident]},
                        attrs = {'model': self.name,
                                 'model_class' : 'lc',
                                 'schedule' : trials.attrs['schedule'],
                                 'resp_type' : resp_type,
                                 'sim_pars' : sim_pars})
        return ds

########## PARAMETERS ##########
                         
# ADD HYPERPARAMETERS FOR BETA PRIOR
par_names = ['resp_scale']; par_list = [{'min': 0.0, 'max': 10.0, 'default': 1.0, 'description': 'scales softmax/logistic response functions'}]
par_names += ['gamma']; par_list += [{'min': 0.0, 'max': 10.0, 'default': 2.0, 'description': 'decay rate for exponential SCRP; higher -> favors more recent latent causes'}] 
par_names += ['alpha']; par_list += [{'min': 0.0, 'max': 40.0, 'default': 2.0, 'description': 'concentration parameter; higher -> tend to infer more latent causes'}]                                

pars = pd.DataFrame(par_list, index = par_names)
del par_names; del par_list