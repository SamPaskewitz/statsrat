from statsrat import rw

########## EXAMPLES OF RESCORLA-WAGNER FAMILY MODELS ##########

# Basic Rescorla-Wagner model with elemental features.
basic = rw.model(name = 'basic',
                 fbase = rw.fbase.elem,
                 fweight = rw.fweight.none, 
                 lrate = rw.lrate.cnst,
                 drate = rw.drate.zero,
                 aux = rw.aux.basic)

# Basic Rescorla-Wagner model with elemental features and constant weight decay.
decay = rw.model(name = 'decay',
                 fbase = rw.fbase.elem,
                 fweight = rw.fweight.none, 
                 lrate = rw.lrate.cnst,
                 drate = rw.drate.cnst,
                 aux = rw.aux.basic)

# Rescorla-Wagner model with elemental features plus intercept.
intercept = rw.model(name = 'intercept',
                     fbase = rw.fbase.elem_intercept,
                     fweight = rw.fweight.none, 
                     lrate = rw.lrate.cnst,
                     drate = rw.drate.zero,
                     aux = rw.aux.basic)

# Rescorla-Wagner model with added binary configural features.
cfg2 = rw.model(name = 'cfg2',
                fbase = rw.fbase.cfg2,
                fweight = rw.fweight.none,
                lrate = rw.lrate.cnst,
                drate = rw.drate.zero,
                aux = rw.aux.basic)

# Rescorla-Wagner model with harmonically decaying learning rate.
hrmn = rw.model(name = 'hrmn',
                fbase = rw.fbase.elem,
                fweight = rw.fweight.none,
                lrate = rw.lrate.hrmn,
                drate = rw.drate.zero,
                aux = rw.aux.basic)

# Rescorla-Wagner model with harmonically decaying learning rate and intercept.
hrmn_intercept = rw.model(name = 'hrmn_intercept',
                          fbase = rw.fbase.elem_intercept,
                          fweight = rw.fweight.none,
                          lrate = rw.lrate.hrmn,
                          drate = rw.drate.zero,
                          aux = rw.aux.basic)

# Rescorla-Wagner model with harmonically decaying learning rate and binary configural features.
hrmncfg2 = rw.model(name = 'hrmncfg2',
                    fbase = rw.fbase.cfg2,
                    fweight = rw.fweight.none,
                    lrate = rw.lrate.hrmn,
                    drate = rw.drate.zero,
                    aux = rw.aux.basic)

# Derived attention model.
# Le Pelley, Mitchell, Beesley, George and Wills (2016)
drva = rw.model(name = 'drva',
                fbase = rw.fbase.elem,
                fweight = rw.fweight.none,
                lrate = rw.lrate.from_aux_feature,
                drate = rw.drate.zero,
                aux = rw.aux.drva)

# Simple predictiveness model (with only elemental features).
# Paskewitz and Jones (2020): Model 2
smpr = rw.model(name = 'smpr',
                fbase = rw.fbase.elem,
                fweight = rw.fweight.from_aux_feature,
                lrate = rw.lrate.from_aux_feature,
                drate = rw.drate.zero,
                aux = rw.aux.grad)

# CompAct (with only elemental features).
# Paskewitz and Jones (2020): Model 4
CompAct = rw.model(name = 'CompAct',
                   fbase = rw.fbase.elem,
                   fweight = rw.fweight.from_aux_norm,
                   lrate = rw.lrate.from_aux_norm,
                   drate = rw.drate.zero,
                   aux = rw.aux.gradcomp)

# CompAct (with intercept term).
CompAct_intercept = rw.model(name = 'CompAct_intercept',
                             fbase = rw.fbase.elem_intercept,
                             fweight = rw.fweight.from_aux_norm,
                             lrate = rw.lrate.from_aux_norm,
                             drate = rw.drate.zero,
                             aux = rw.aux.gradcomp)

# Kalman filter Rescorla-Wagner (with only elemental features).
# Dayan and Kakade (2001), Gershman and Diedrichsen (2015)
Kalman = rw.model(name = 'Kalman',
                  fbase = rw.fbase.elem,
                  fweight = rw.fweight.none,
                  lrate = rw.lrate.from_aux_direct,
                  drate = rw.drate.zero,
                  aux = rw.aux.Kalman)

# Kalman filter Rescorla-Wagner (with intercept term).
Kalman_intercept = rw.model(name = 'Kalman_intercept',
                            fbase = rw.fbase.elem_intercept,
                            fweight = rw.fweight.none,
                            lrate = rw.lrate.from_aux_direct,
                            drate = rw.drate.zero,
                            aux = rw.aux.Kalman)