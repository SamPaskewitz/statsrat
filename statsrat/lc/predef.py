from statsrat import lc

# discrete (Bernoulli) likelihood - updating based on hard assignment (c.f. Anderson, 1991) - with coupling prob prior
discrete_cpl = lc.model(name = 'discrete_cpl',
                        u_dist = lc.u_dist.discrete_hard,
                        x_dist = lc.x_dist.discrete_hard,
                        prior = lc.prior.coupling_prob)

# normal likelihood (hard assignment) with coupling prob prior
normal_cpl = lc.model(name = 'normal_cpl',
                      u_dist = lc.u_dist.normal_hard,
                      x_dist = lc.x_dist.normal_hard,
                      prior = lc.prior.coupling_prob)

# discrete (Bernoulli) likelihood (hard assignment) with Chinese restaurant process prior
discrete_crp = lc.model(name = 'discrete_crp',
                        u_dist = lc.u_dist.discrete_hard,
                        x_dist = lc.x_dist.discrete_hard,
                        prior = lc.prior.Chinese_rest)

# normal likelihood (hard assignment) with Chinese restaurant process prior
normal_crp = lc.model(name = 'normal_crp',
                      u_dist = lc.u_dist.normal_hard,
                      x_dist = lc.x_dist.normal_hard,
                      prior = lc.prior.Chinese_rest)

# discrete (Bernoulli)  likelihood (hard assignment) with power law prior
discrete_power = lc.model(name = 'power',
                          u_dist = lc.u_dist.discrete_hard,
                          x_dist = lc.x_dist.discrete_hard,
                          prior = lc.prior.power_law)

# normal likelihood (hard assignment) with power law prior
normal_power = lc.model(name = 'power',
                        u_dist = lc.u_dist.normal_hard,
                        x_dist = lc.x_dist.normal_hard,
                        prior = lc.prior.power_law)