from statsrat import rw, rw_plus_minus

decay_plus_minus = rw_plus_minus.model(name = 'decay_plus_minus',
                                       pred = rw.pred.rectified,
                                       fbase = rw.fbase.elem,
                                       fweight = rw.fweight.none,
                                       lrate = rw.lrate.cnst,
                                       aux = rw.aux.basic)