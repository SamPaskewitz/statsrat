from statsrat import rw, rw_plus_minus

decay_plus_minus = rw_plus_minus.model(name = 'decay_plus_minus',
                                       pred = rw.pred.rectified,
                                       fbase = rw.fbase.elem,
                                       fweight = rw.fweight.none,
                                       lrate = rw.lrate.cnst,
                                       drate_plus = rw.drate.cnst,
                                       drate_minus = rw.drate.cnst,
                                       aux = rw.aux.basic)

decay_only_minus = rw_plus_minus.model(name = 'decay_only_minus',
                                       pred = rw.pred.rectified,
                                       fbase = rw.fbase.elem,
                                       fweight = rw.fweight.none,
                                       lrate = rw.lrate.cnst,
                                       drate_plus = rw.drate.zero,
                                       drate_minus = rw.drate.cnst,
                                       aux = rw.aux.basic)