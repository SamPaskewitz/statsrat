import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats
from statsrat import resp_fun
from . import u_dist, x_dist, prior

class model:
    '''
    Class for latent cause models.
    
    Attributes
    ----------
    name: str
        Model name.
    u_dist: object
        Distribution of the outcome vector.
    x_dist: object
        Distribution of the stimulus vector.
    prior: function
        Prior on latent causes.
    par_names: list
        Names of the model's free parameters (strings).
          
    Methods
    -------
    simulate(trials, resp_type = 'choice', par_val = None, random_resp = False, ident = 'sim', max_z = 15)
        Simulate a trial sequence once with known model parameters.
        
    Notes
    -----
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
    def __init__(self, name, u_dist, x_dist, prior):
        '''
        Parameters
        ----------
        name: str
            Model name.
        u_dist: object
            Distribution of the outcome vector.
        x_dist: object
            Distribution of the stimulus vector.
        prior: function
            Prior on latent causes.
        '''
        self.name = name
        self.u_dist = u_dist
        self.x_dist = x_dist
        self.prior = prior
        # determine the model's parameter space
        par_names = list(np.unique(u_dist.par_names + x_dist.par_names + prior.par_names))
        self.pars = pars.loc[par_names + ['resp_scale']]
        
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
        delta = np.zeros((n_t, n_u)) # prediction error
        u_dist = self.u_dist(sim_pars, n_t, n_u, max_z) # outcome distribution
        x_dist = self.x_dist(sim_pars, n_t, n_x, max_z) # stimulus distribution
        tau_u = np.zeros((n_t, max_z, u_dist.n_tau)) # natural hyperparameters of outcome distribution
        tau_x = np.zeros((n_t, max_z, x_dist.n_tau)) # natural hyperparameters of stimulus distribution
        prior = np.zeros((n_t, max_z)) # prior on latent causes
        post_x = np.zeros((n_t, max_z)) # posterior of latent causes after observing x, but before observing u
        post_xu = np.zeros((n_t, max_z)) # posterior of latent causes after observing both x and u
        x_lik = np.zeros((n_t, max_z)) # likelihood of x
        u_lik = np.zeros((n_t, max_z)) # likelihood of u
        z_counts = np.zeros(max_z) # estimated number of observations assigned to each latent cause
        x_obs_sofar = np.zeros(n_x) # whether or not each stimulus attribute (x) has been observed so far
        u_obs_sofar = np.zeros(n_u) # whether or not each outcome (u) has been observed so far
        
        # set up response function (depends on response type)
        resp_dict = {'choice': resp_fun.choice,
                     'exct': resp_fun.exct,
                     'supr': resp_fun.supr}
        sim_resp_fun = resp_dict[resp_type]
        
        # loop through time steps
        for t in range(n_t):
            # calculate posterior on latent causes before observing outcome (u)
            prior[t, :] = self.prior(sim_pars, t, z_est, z_counts, max_z) # prior on latent causes
            x_obs_sofar[x[t, :] > 0] = 1 # keep track of stimulus attributes (x) observed so far
            #x_lik[t, :] = x_dist.lik(obs = x[t, :], use = x_obs_sofar) # likelihood of x (based on attributes observed so far)
            
            x_lik_array = np.zeros(n_x)
            for i in range(n_x):
                x_lik_array[i] = np.exp(np.inner(x_dist.suf_stat(x[t, :]), x_dist.expected_eta)) # FIX!!!
            x_lik[t, :] = np.prod(x_lik_array) # unnormalized likelihood of x using variational means of parameters
            numerator = prior[t, :]*x_lik[t, :]
            post_x[t, :] = numerator/numerator.sum() # posterior on latent causes after observing only x (using variational means)
            
            # reward prediction, before feedback
            #u_hat[t, :] = u_dist.predict(sim_pars, t, x[t, :], u_psb[t, :], post_x[t, :]) # prediction
            u_hat_z = u_psb[t, :]*u_dist.predict_mean() # predictions from each latent cause
            u_hat[t, :] = post_x[t, :]@u_hat_z # prediction (probability weighted average across latent causes)
            b_hat[t, :] = sim_resp_fun(u_hat[t, :], u_psb[t, :], sim_pars['resp_scale']) # response
                         
            # calculate posterior on latent causes after observing outcome (u)
            u_obs_sofar[u[t, :] > 0] = 1 # keep track of outcomes (u) observed so far
            #u_lik[t, :] = u_dist.lik(obs = u[t, :], use = u_psb[t, :]*u_obs_sofar) # likelihood of u
            u_lik[t, :] = np.exp() # unnormalized likelihood of u using variational means of parameters
            new_numerator = numerator*u_lik[t, :]
            post_xu[t, :] = new_numerator/new_numerator.sum() # posterior on latent causes after observing both x and u (using variational means)
                         
            # update x and u distribution parameters
            # ADD EFFECT OF u_lrn
            tau_u[t, :, :] += post_xu[t, :]@(u_lrn[t, :]*u_dist.suf_stat(u[t, :]))
            tau_x[t, :, :] += post_xu[t, :]@(x_obs_sofar*x_dist.suf_stat(x[t, :]))
            #x_dist.update(x[t, :], post_xu[t, :])
            #u_dist.update(u[t, :], post_xu[t, :])
            
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
                         
par_names = ['resp_scale']; par_list = [{'min': 0.0, 'max': 10.0, 'default': 1.0}]
par_names += ['beta0']; par_list += [{'min': 0.0, 'max': 50.0, 'default': 10}] # prior strength, i.e. half of "sample size" (discrete likelihood)
par_names += ['alpha']; par_list += [{'min': 0.0, 'max': 40.0, 'default': 2}] # concentration parameter (for prior); higher -> tend to infer more latent causes
par_names += ['lmb0']; par_list += [{'min': 0.0, 'max': 20.0, 'default': 0.1}] # confidence in the prior on mu (normal likelihood)
par_names += ['a0']; par_list += [{'min': 0.0, 'max': 20.0, 'default': 1.0}] # confidence in the prior on sigma (normal likelihood)
pars = pd.DataFrame(par_list, index = par_names)
del par_names; del par_list