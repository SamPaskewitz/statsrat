from statsrat import exemplar

basic = exemplar.model(name = 'basic',
                       sim = exemplar.sim.Gaussian,
                       rtrv = exemplar.rtrv.normalized_sim,
                       atn_update = exemplar.atn_update.null,
                       y_ex_update = exemplar.y_ex_update.ex_mean)