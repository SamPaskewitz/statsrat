from statsrat import rw

'''
Pre-defined Rescorla-Wagner family models.

basic: Basic Rescorla-Wagner model with elemental features.

decay: Basic Rescorla-Wagner model with elemental features and constant weight decay.

decay_intercept: Basic Rescorla-Wagner model with intercept and constant weight decay.

intercept: Rescorla-Wagner model with elemental features plus intercept.

cfg2: Rescorla-Wagner model with binary configural features.

cfg2_intercept: Rescorla-Wagner model with binary configural features and intercept.

power: Rescorla-Wagner model with decaying learning rate.

power_intercept: Rescorla-Wagner model with decaying learning rate and intercept.

power_cfg2: Rescorla-Wagner model with decaying learning rate and binary configural features.

drva: The derived attention model from Le Pelley, Mitchell, Beesley, George and Wills (2016).

smpr: Simple predictiveness model (with only elemental features); Model 2 from Paskewitz and Jones (2020).

CompAct: CompAct (with only elemental features); Model 4 from Paskewitz and Jones (2020).

CompAct_intercept: CompAct (with intercept term).

CompAct_cfg2_intercept:: CompAct (with intercept term and configural features).

Kalman: Kalman filter Rescorla-Wagner (with only elemental features); Dayan and Kakade (2001), Gershman and Diedrichsen (2015).

Kalman_intercept: Kalman filter Rescorla-Wagner (with intercept term).
'''


# Basic Rescorla-Wagner model with elemental features.
basic = rw.model(name = 'basic',
                 pred = rw.pred.identity,
                 fbase = rw.fbase.elem,
                 fweight = rw.fweight.none, 
                 lrate = rw.lrate.cnst,
                 drate = rw.drate.zero,
                 aux = rw.aux.basic)

# Basic Rescorla-Wagner model with elemental features and constant weight decay.
decay = rw.model(name = 'decay',
                 pred = rw.pred.identity,
                 fbase = rw.fbase.elem,
                 fweight = rw.fweight.none, 
                 lrate = rw.lrate.cnst,
                 drate = rw.drate.cnst,
                 aux = rw.aux.basic)

# Basic Rescorla-Wagner model with intercept and constant weight decay.
decay_intercept = rw.model(name = 'decay_intercept',
                           pred = rw.pred.identity,
                           fbase = rw.fbase.elem_intercept,
                           fweight = rw.fweight.none, 
                           lrate = rw.lrate.cnst,
                           drate = rw.drate.cnst,
                           aux = rw.aux.basic)

# Rescorla-Wagner model with elemental features plus intercept.
intercept = rw.model(name = 'intercept',
                     pred = rw.pred.identity,
                     fbase = rw.fbase.elem_intercept,
                     fweight = rw.fweight.none, 
                     lrate = rw.lrate.cnst,
                     drate = rw.drate.zero,
                     aux = rw.aux.basic)

# Rescorla-Wagner model with binary configural features.
cfg2 = rw.model(name = 'cfg2',
                pred = rw.pred.identity,
                fbase = rw.fbase.cfg2,
                fweight = rw.fweight.none,
                lrate = rw.lrate.cnst,
                drate = rw.drate.zero,
                aux = rw.aux.basic)

# Rescorla-Wagner model with binary configural features and intercept.
cfg2_intercept = rw.model(name = 'cfg2_intercept',
                          pred = rw.pred.identity,
                          fbase = rw.fbase.cfg2_intercept,
                          fweight = rw.fweight.none,
                          lrate = rw.lrate.cnst,
                          drate = rw.drate.zero,
                          aux = rw.aux.basic)

# Rescorla-Wagner model with decaying learning rate.
power = rw.model(name = 'power',
                pred = rw.pred.identity,
                fbase = rw.fbase.elem,
                fweight = rw.fweight.none,
                lrate = rw.lrate.power,
                drate = rw.drate.zero,
                aux = rw.aux.basic)

# Rescorla-Wagner model with decaying learning rate and intercept.
power_intercept = rw.model(name = 'power_intercept',
                          pred = rw.pred.identity,
                          fbase = rw.fbase.elem_intercept,
                          fweight = rw.fweight.none,
                          lrate = rw.lrate.power,
                          drate = rw.drate.zero,
                          aux = rw.aux.basic)

# Rescorla-Wagner model with decaying learning rate and binary configural features.
power_cfg2 = rw.model(name = 'power_cfg2',
                    pred = rw.pred.identity,
                    fbase = rw.fbase.cfg2,
                    fweight = rw.fweight.none,
                    lrate = rw.lrate.power,
                    drate = rw.drate.zero,
                    aux = rw.aux.basic)

# The derived attention model from Le Pelley, Mitchell, Beesley, George and Wills (2016).
drva = rw.model(name = 'drva',
                pred = rw.pred.identity,
                fbase = rw.fbase.elem,
                fweight = rw.fweight.none,
                lrate = rw.lrate.from_aux_feature,
                drate = rw.drate.zero,
                aux = rw.aux.drva)

# Simple predictiveness model (with only elemental features); Model 2 from Paskewitz and Jones (2020).
smpr = rw.model(name = 'smpr',
                pred = rw.pred.identity,
                fbase = rw.fbase.elem,
                fweight = rw.fweight.from_aux_feature,
                lrate = rw.lrate.from_aux_feature,
                drate = rw.drate.zero,
                aux = rw.aux.grad)

# CompAct (with only elemental features); Model 4 from Paskewitz and Jones (2020).
CompAct = rw.model(name = 'CompAct',
                   pred = rw.pred.identity,
                   fbase = rw.fbase.elem,
                   fweight = rw.fweight.from_aux_norm,
                   lrate = rw.lrate.from_aux_norm,
                   drate = rw.drate.zero,
                   aux = rw.aux.gradcomp)

# CompAct (with intercept term).
CompAct_intercept = rw.model(name = 'CompAct_intercept',
                             pred = rw.pred.identity,
                             fbase = rw.fbase.elem_intercept,
                             fweight = rw.fweight.from_aux_norm,
                             lrate = rw.lrate.from_aux_norm,
                             drate = rw.drate.zero,
                             aux = rw.aux.gradcomp)

# CompAct (with intercept term and configural features).
CompAct_cfg2_intercept = rw.model(name = 'CompAct_cfg2_intercept',
                                  pred = rw.pred.identity,
                                  fbase = rw.fbase.cfg2_intercept,
                                  fweight = rw.fweight.from_aux_norm,
                                  lrate = rw.lrate.from_aux_norm,
                                  drate = rw.drate.zero,
                                  aux = rw.aux.gradcomp)

# Kalman filter Rescorla-Wagner (with only elemental features); Dayan and Kakade (2001), Gershman and Diedrichsen (2015).
Kalman = rw.model(name = 'Kalman',
                  pred = rw.pred.identity,
                  fbase = rw.fbase.elem,
                  fweight = rw.fweight.none,
                  lrate = rw.lrate.from_aux_direct,
                  drate = rw.drate.zero,
                  aux = rw.aux.Kalman)

# Kalman filter Rescorla-Wagner (with intercept term).
Kalman_intercept = rw.model(name = 'Kalman_intercept',
                            pred = rw.pred.identity,
                            fbase = rw.fbase.elem_intercept,
                            fweight = rw.fweight.none,
                            lrate = rw.lrate.from_aux_direct,
                            drate = rw.drate.zero,
                            aux = rw.aux.Kalman)