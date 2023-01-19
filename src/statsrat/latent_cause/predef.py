from statsrat import latent_cause as lc

constant = lc.model(name = 'constant',
                    kernel = lc.kernel.constant)

exponential = lc.model(name = 'exponential',
                       kernel = lc.kernel.exponential)

power = lc.model(name = 'power',
                 kernel = lc.kernel.power)

power_asymptote = lc.model(name = 'power_asymptote',
                           kernel = lc.kernel.power_asymptote)

power_clusters = lc.model(name = 'power_clusters',
                          kernel = lc.kernel.power_clusters)

refractory_period = lc.model(name = 'refractory_period',
                             kernel = lc.kernel.refractory_period)