from statsrat import expr

'''
These are very simple examples of Pavlovian conditioning designs.

They are not based on any particular study, but are similar to many.

For simplicity, these are all two group, between subjects designs
with only a single punctate CS.

The OAT score for each schedule is simply response to CS.

Each OAT is defined to be positive if it corresponds to published results.

This version includes inter-trial intervals (ITIs).
'''

##### DEFINE SCHEDULES #####

# no conditioning (as a very basic control)
no_cond = expr.schedule(name = 'no_cond',
                    stage_list = [
                        expr.stage(name = 'intro',
                              x_pn = [[]],
                              x_bg = ['ctx_a'],
                              u_psb = ['us'],
                              order_fixed = True,
                              iti = 5,
                              n_rep = 5),
                        expr.stage(name = 'test',
                              x_pn = [['cs']],
                              x_bg = ['ctx'],
                              u_psb = ['us'],
                              order_fixed = True,
                              iti = 5,
                              n_rep = 5)
                    ])

# basic conditioning
cond = expr.schedule(name = 'cond',
                stage_list = [
                   expr.stage(name = 'intro',
                              x_pn = [[]],
                              x_bg = ['ctx'],
                              u_psb = ['us'],
                              order_fixed = True,
                              iti = 5,
                              n_rep = 5),
                   expr.stage(name = 'training',
                         x_pn = [['cs']],
                         x_bg = ['ctx'],
                         u = [['us']],
                         u_psb = ['us'],
                         order_fixed = True, 
                         iti = 5,
                              n_rep = 5),
                    expr.stage(name = 'test',
                          x_pn = [['cs']],
                          x_bg = ['ctx'],
                          u_psb = ['us'],
                          order_fixed = True,
                          iti = 5,
                               n_rep = 5)
                ])

# blocking
blocking = expr.schedule(name = 'blocking',
                    stage_list = [
                        expr.stage(name = 'intro',
                              x_pn = [[]],
                              x_bg = ['ctx'],
                              u_psb = ['us'],
                              order_fixed = True,
                              iti = 5,
                              n_rep = 5),
                        expr.stage(name = 'first_cue',
                              x_pn = [['cs1']],
                              x_bg = ['ctx'],
                              u = [['us']],
                              u_psb = ['us'],
                              order_fixed = True,
                              iti = 5,
                                   n_rep = 5),
                        expr.stage(name = 'training',
                             x_pn = [['cs1', 'cs2']],
                             x_bg = ['ctx'],
                             u = [['us']],
                             u_psb = ['us'],
                             order_fixed = False, 
                             iti = 5,
                                   n_rep = 5),
                        expr.stage(name = 'test',
                              x_pn = [['cs1'], ['cs2']],
                              x_bg = ['ctx'],
                              u_psb = ['us'],
                              order_fixed = False,
                              iti = 5,
                                   n_rep = 5)
                    ])

# two cue conditioning (e.g. overshadowing)
two_cue = expr.schedule(name = 'two_cue',
                   stage_list = [
                       expr.stage(name = 'intro',
                              x_pn = [[]],
                              x_bg = ['ctx'],
                              u_psb = ['us'],
                              order_fixed = True,
                              iti = 5,
                              n_rep = 5),
                       expr.stage(name = 'training',
                             x_pn = [['cs1', 'cs2']],
                             x_bg = ['ctx'],
                             u = [['us']],
                             u_psb = ['us'],
                             order_fixed = False, 
                             iti = 5,
                                  n_rep = 5),
                        expr.stage(name = 'test',
                              x_pn = [['cs1'], ['cs2']],
                              x_bg = ['ctx'],
                              u_psb = ['us'],
                              order_fixed = False,
                              iti = 5,
                                   n_rep = 5)
                   ])

# pre-exposure (for latent inhibition)
pre_exp = expr.schedule(name = 'pre_exp',
                   stage_list = [
                       expr.stage(name = 'intro',
                              x_pn = [[]],
                              x_bg = ['ctx'],
                              u_psb = ['us'],
                              order_fixed = True,
                              iti = 5,
                              n_rep = 5),
                       expr.stage(name = 'pre_exp',
                             x_pn = [['cs']],
                             x_bg = ['ctx'],
                             u_psb = ['us'],
                             order_fixed = True,
                             iti = 5,
                                  n_rep = 5),
                       expr.stage(name = 'training',
                             x_pn = [['cs']],
                             x_bg = ['ctx'],
                             u = [['us']],
                             u_psb = ['us'],
                             order_fixed = True,
                             iti = 5,
                                  n_rep = 5),
                        expr.stage(name = 'test',
                              x_pn = [['cs']],
                              x_bg = ['ctx'],
                              u_psb = ['us'],
                              order_fixed = True,
                              iti = 5,
                                   n_rep = 5)
                   ])

# pre-exposure (ABC)
pre_exp_abc = expr.schedule(name = 'pre_exp_abc',
                       stage_list = [
                       expr.stage(name = 'pre_exp',
                             x_pn = [['cs']],
                             x_bg = ['ctx_a'],
                             u_psb = ['us'],
                             order_fixed = True,
                             iti = 5,
                                  n_rep = 5),
                       expr.stage(name = 'intro',
                          x_pn = [[]],
                          x_bg = ['ctx_b'],
                          u_psb = ['us'],
                          order_fixed = True,
                          iti = 5,
                          n_rep = 5),
                       expr.stage(name = 'training',
                             x_pn = [['cs']],
                             x_bg = ['ctx_b'],
                             u = [['us']],
                             u_psb = ['us'],
                             order_fixed = True,
                             iti = 5,
                                  n_rep = 5),
                       expr.stage(name = 'test',
                             x_pn = [['cs']],
                             x_bg = ['ctx_c'],
                             u_psb = ['us'],
                             order_fixed = True,
                             iti = 5,
                                  n_rep = 5)
                       ])

# extinction
extn = expr.schedule(name = 'extinction',
                   stage_list = [
                       expr.stage(name = 'intro',
                              x_pn = [[]],
                              x_bg = ['ctx'],
                              u_psb = ['us'],
                              order_fixed = True,
                              iti = 5,
                              n_rep = 5),
                       expr.stage(name = 'training',
                             x_pn = [['cs']],
                             x_bg = ['ctx'],
                             u = [['us']],
                             u_psb = ['us'],
                             order_fixed = True, 
                             iti = 5,
                                  n_rep = 5),
                       expr.stage(name = 'extinction',
                             x_pn = [['cs']],
                             x_bg = ['ctx'],
                             u_psb = ['us'],
                             order_fixed = True,
                             iti = 5,
                                  n_rep = 5),
                       expr.stage(name = 'test',
                             x_pn = [['cs']],
                             x_bg = ['ctx'],
                             u_psb = ['us'],
                             order_fixed = True,
                             iti = 5,
                                  n_rep = 5)
                   ])

# extinction (ABC)
extn_abc = expr.schedule(name = 'extinction_abc',
                   stage_list = [
                       expr.stage(name = 'intro',
                              x_pn = [[]],
                              x_bg = ['ctx_a'],
                              u_psb = ['us'],
                              order_fixed = True,
                              iti = 5,
                              n_rep = 5),
                       expr.stage(name = 'training',
                             x_pn = [['cs']],
                             x_bg = ['ctx_a'],
                             u = [['us']],
                             u_psb = ['us'],
                             order_fixed = True, 
                             iti = 5,
                             n_rep = 5),
                       expr.stage(name = 'extinction',
                             x_pn = [['cs']],
                             x_bg = ['ctx_b'],
                             u_psb = ['us'],
                             order_fixed = True,
                             iti = 5,
                             n_rep = 5),
                       expr.stage(name = 'test',
                                  x_pn = [['cs']],
                                  x_bg = ['ctx_c'],
                                  u_psb = ['us'],
                                  order_fixed = True,
                                  iti = 5,
                                  n_rep = 5)
                   ])

# extinction with extra US (for reinstatement)
extn_extra_us = expr.schedule(name = 'extinction_extra_us',
                               stage_list = [
                                   expr.stage(name = 'intro',
                                          x_pn = [[]],
                                          x_bg = ['ctx'],
                                          u_psb = ['us'],
                                          order_fixed = True,
                                          iti = 5,
                                          n_rep = 5),
                                   expr.stage(name = 'training',
                                         x_pn = [['cs']],
                                         x_bg = ['ctx'],
                                         u = [['us']],
                                         u_psb = ['us'],
                                         order_fixed = True, 
                                         iti = 5,
                                         n_rep = 5),
                                   expr.stage(name = 'extinction',
                                         x_pn = [['cs']],
                                         x_bg = ['ctx'],
                                         u_psb = ['us'],
                                         order_fixed = True,
                                         iti = 5,
                                         n_rep = 5),
                                   expr.stage(name = 'extra_us',
                                         x_pn = [[]],
                                         x_bg = ['ctx'],
                                         u = [['us']],
                                         u_psb = ['us'],
                                         order_fixed = True,
                                         iti = 5,
                                         n_rep = 5),
                                   expr.stage(name = 'test',
                                         x_pn = [['cs']],
                                         x_bg = ['ctx'],
                                         u_psb = ['us'],
                                         order_fixed = True,
                                         iti = 5,
                                         n_rep = 5)
                               ])

# extinction with extra time in the conditioning/extinction context before test
extn_extra_time = expr.schedule(name = 'extinction_extra_time',
                                   stage_list = [
                                       expr.stage(name = 'intro',
                                              x_pn = [[]],
                                              x_bg = ['ctx'],
                                              u_psb = ['us'],
                                              order_fixed = True,
                                              iti = 5,
                                              n_rep = 5),
                                       expr.stage(name = 'training',
                                             x_pn = [['cs']],
                                             x_bg = ['ctx'],
                                             u = [['us']],
                                             u_psb = ['us'],
                                             order_fixed = True, 
                                             iti = 5,
                                             n_rep = 5),
                                       expr.stage(name = 'extinction',
                                             x_pn = [['cs']],
                                             x_bg = ['ctx'],
                                             u_psb = ['us'],
                                             order_fixed = True,
                                             iti = 5,
                                             n_rep = 5),
                                       expr.stage(name = 'extra_time',
                                             x_pn = [[]],
                                             x_bg = ['ctx'],
                                             u_psb = ['us'],
                                             order_fixed = True,
                                             iti = 5,
                                             n_rep = 5),
                                       expr.stage(name = 'test',
                                             x_pn = [['cs']],
                                             x_bg = ['ctx'],
                                             u_psb = ['us'],
                                             order_fixed = True,
                                             iti = 5,
                                             n_rep = 5)
                                   ])

# extinction with delay (in the home cage, i.e. a separate context) before test
extn_delay = expr.schedule(name = 'extinction_delay',
                                   stage_list = [
                                       expr.stage(name = 'intro',
                                              x_pn = [[]],
                                              x_bg = ['ctx'],
                                              u_psb = ['us'],
                                              order_fixed = True,
                                              iti = 5,
                                              n_rep = 5),
                                       expr.stage(name = 'training',
                                             x_pn = [['cs']],
                                             x_bg = ['ctx'],
                                             u = [['us']],
                                             u_psb = ['us'],
                                             order_fixed = True, 
                                             iti = 5,
                                             n_rep = 5),
                                       expr.stage(name = 'extinction',
                                             x_pn = [['cs']],
                                             x_bg = ['ctx'],
                                             u_psb = ['us'],
                                             order_fixed = True,
                                             iti = 5,
                                             n_rep = 5),
                                       expr.stage(name = 'delay',
                                             x_pn = [[]],
                                             x_bg = ['home_cage'],
                                             u_psb = ['us'],
                                             order_fixed = True,
                                             iti = 0,
                                             n_rep = 50),
                                       expr.stage(name = 'test',
                                             x_pn = [['cs']],
                                             x_bg = ['ctx'],
                                             u_psb = ['us'],
                                             order_fixed = True,
                                             iti = 5,
                                             n_rep = 5)
                                   ])

##### DEFINE BEHAVIORAL SCORE #####
test_score = expr.behav_score(stage = 'test',
                               trial_pos = ['cs -> nothing', 'cs1 -> nothing'],
                               resp_pos = 2*['us'])


##### DEFINE OATS AND EXPERIMENTS #####

# basic conditioning, i.e. acquistion of a conditioned response
conditioning = expr.experiment(resp_type = 'exct',
                               schedules = {'control': no_cond, 'experimental': cond},
                               oats = {'acquistion_of_cr': expr.oat(schedule_pos = ['experimental'],
                                                                    schedule_neg = ['control'],
                                                                    behav_score_pos = test_score)
})

# basic extinction of a conditioned response
extinction = expr.experiment(resp_type = 'exct',
                             schedules = {'control': cond, 'experimental': extn},
                             oats = {'extinction_of_cr': expr.oat(schedule_pos = ['control'],
                                                                  schedule_neg = ['experimental'],
                                                                  behav_score_pos = test_score)
})

# context dependence of extinction (i.e. ABC renewal)
renewal = expr.experiment(resp_type = 'exct',
                          schedules = {'experimental': extn_abc, 'control': extn},
                          oats = {'renewal': expr.oat(schedule_pos = ['experimental'],
                                                      schedule_neg = ['control'],
                                                      behav_score_pos = test_score)
})

# reinstatement
reinstatement = expr.experiment(resp_type = 'exct',
                                schedules = {'experimental': extn_extra_us, 'control': extn_extra_time},
                                oats = {'reinstatement': expr.oat(schedule_pos = ['experimental'],
                                                                  schedule_neg = ['control'],
                                                                  behav_score_pos = test_score)
})

# basic latent inhibition
latent_inhib = expr.experiment(resp_type = 'exct',
                               schedules = {'control': cond, 'experimental': pre_exp},
                               oats = {'latent_inhibition': expr.oat(schedule_pos = ['control'],
                                                                     schedule_neg = ['experimental'],
                                                                     behav_score_pos = test_score)
})

# context dependence of latent inhibition
ctx_li = expr.experiment(resp_type = 'exct',
                         schedules = {'experimental': pre_exp_abc, 'control': pre_exp},
                         oats = {'ctx_dependence_li': expr.oat(schedule_pos = ['experimental'],
                                                               schedule_neg = ['control'],
                                                               behav_score_pos = test_score)
})

# overshadowing
overshadowing = expr.experiment(resp_type = 'exct',
                                schedules = {'control': cond, 'experimental': two_cue},
                                oats = {'overshadowing': expr.oat(schedule_pos = ['control'],
                                                                  schedule_neg = ['experimental'],
                                                                  behav_score_pos = test_score)
})

# blocking
blocking = expr.experiment(resp_type = 'exct',
                           schedules = {'control': two_cue, 'experimental': blocking},
                           oats = {'blocking': expr.oat(schedule_pos = ['control'],
                                                          schedule_neg = ['experimental'],
                                                          behav_score_pos = test_score)
})

# spontaneous recovery
spont_rec = expr.experiment(resp_type = 'exct',
                            schedules = {'experimental': extn_delay, 'control': extn},
                            oats = {'spontaneous_recovery': expr.oat(schedule_pos = ['experimental'],
                                                                       schedule_neg = ['control'],
                                                                       behav_score_pos = test_score)
})