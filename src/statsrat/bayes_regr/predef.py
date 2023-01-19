from statsrat import bayes_regr as br
from statsrat.rw.fbase import elem

linear_constant = br.model(name = 'linear_constant', 
                           fbase = elem, 
                           link = br.link.linear, 
                           tausq_inv_dist = br.tausq_inv_dist.constant)

probit_constant = br.model(name = 'probit_constant', 
                           fbase = elem, 
                           link = br.link.probit, 
                           tausq_inv_dist = br.tausq_inv_dist.constant)

multinomial_probit_constant = br.model(name = 'multinomial_probit_constant', 
                                       fbase = elem, 
                                       link = br.link.multinomial_probit, 
                                       tausq_inv_dist = br.tausq_inv_dist.constant)

linear_ard = br.model(name = 'linear_ard', 
                      fbase = elem, 
                      link = br.link.linear, 
                      tausq_inv_dist = br.tausq_inv_dist.ard)

linear_ard_drv_atn = br.model(name = 'linear_ard_drv_atn', 
                              fbase = elem, 
                              link = br.link.linear, 
                              tausq_inv_dist = br.tausq_inv_dist.ard_drv_atn)