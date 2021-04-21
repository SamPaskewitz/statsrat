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

iti = 5
n_rep_train = 5
n_rep_extn = 5
n_rep_test = 2
n_rep_pre_exp = 5
n_rep_no_stim = 10

##### DEFINE STAGES #####

no_stim_stage = expr.stage(name = 'no_stim',
                           x_pn = [[]],
                           x_bg = ['ctx'],
                           u_psb = ['us'],
                           order_fixed = True,
                           iti = 0,
                           n_rep = n_rep_no_stim)

no_stim_ctx_a_stage = expr.stage(name = 'no_stim',
                                 x_pn = [[]],
                                 x_bg = ['ctx_a'],
                                 u_psb = ['us'],
                                 order_fixed = True,
                                 iti = 0,
                                 n_rep = n_rep_no_stim)

test_stage = expr.stage(name = 'test',
                        x_pn = [['cs']],
                        x_bg = ['ctx'],
                        u_psb = ['us'],
                        order_fixed = True,
                        iti = iti,
                        n_rep = n_rep_test)

test_cs2_stage = expr.stage(name = 'test',
                            x_pn = [['cs2']],
                            x_bg = ['ctx'],
                            u_psb = ['us'],
                            order_fixed = False,
                            iti = iti,
                            n_rep = n_rep_test)

test_ctx_a_stage = expr.stage(name = 'test',
                              x_pn = [['cs']],
                              x_bg = ['ctx_a'],
                              u_psb = ['us'],
                              order_fixed = True,
                              iti = iti,
                              n_rep = n_rep_test)

test_ctx_b_stage = expr.stage(name = 'test',
                              x_pn = [['cs']],
                              x_bg = ['ctx_b'],
                              u_psb = ['us'],
                              order_fixed = True,
                              iti = iti,
                              n_rep = n_rep_test)

test_ctx_c_stage = expr.stage(name = 'test',
                              x_pn = [['cs']],
                              x_bg = ['ctx_c'],
                              u_psb = ['us'],
                              order_fixed = True,
                              iti = iti,
                              n_rep = n_rep_test)

delay_stage = expr.stage(name = 'delay',
                         x_pn = [[]],
                         x_bg = ['home_cage'],
                         u_psb = ['us'],
                         order_fixed = True,
                         iti = 0,
                         n_rep = 400)

one_cue_stage = expr.stage(name = 'one_cue',
                           x_pn = [['cs1']],
                           x_bg = ['ctx'],
                           u = [['us']],
                           u_psb = ['us'],
                           order_fixed = False, 
                           iti = iti,
                           n_rep = n_rep_train)

two_cue_stage = expr.stage(name = 'two_cue',
                           x_pn = [['cs1', 'cs2']],
                           x_bg = ['ctx'],
                           u = [['us']],
                           u_psb = ['us'],
                           order_fixed = False, 
                           iti = iti,
                           n_rep = n_rep_train)

pre_exp_stage = expr.stage(name = 'pre_exposure',
                           x_pn = [['cs']],
                           x_bg = ['ctx'],
                           u_psb = ['us'],
                           order_fixed = True,
                           iti = iti,
                           n_rep = n_rep_pre_exp)

pre_exp_ctx_a_stage = expr.stage(name = 'pre_exposure',
                                 x_pn = [['cs']],
                                 x_bg = ['ctx_a'],
                                 u_psb = ['us'],
                                 order_fixed = True,
                                 iti = iti,
                                 n_rep = n_rep_pre_exp)

training_stage = expr.stage(name = 'training',
                            x_pn = [['cs']],
                            x_bg = ['ctx'],
                            u = [['us']],
                            u_psb = ['us'],
                            order_fixed = True,
                            iti = iti,
                            n_rep = n_rep_train)

training_ctx_a_stage = expr.stage(name = 'training',
                                  x_pn = [['cs']],
                                  x_bg = ['ctx_a'],
                                  u = [['us']],
                                  u_psb = ['us'],
                                  order_fixed = True,
                                  iti = iti,
                                  n_rep = n_rep_train)

training_ctx_b_stage = expr.stage(name = 'training',
                                  x_pn = [['cs']],
                                  x_bg = ['ctx_b'],
                                  u = [['us']],
                                  u_psb = ['us'],
                                  order_fixed = True,
                                  iti = iti,
                                  n_rep = n_rep_train)

extn_stage = expr.stage(name = 'extinction',
                        x_pn = [['cs']],
                        x_bg = ['ctx'],
                        u_psb = ['us'],
                        order_fixed = True,
                        iti = iti,
                        n_rep = n_rep_extn)

extn_ctx_a_stage = expr.stage(name = 'extinction',
                              x_pn = [['cs']],
                              x_bg = ['ctx_a'],
                              u_psb = ['us'],
                              order_fixed = True,
                              iti = iti,
                              n_rep = n_rep_extn)

extn_ctx_b_stage = expr.stage(name = 'extinction',
                              x_pn = [['cs']],
                              x_bg = ['ctx_b'],
                              u_psb = ['us'],
                              order_fixed = True,
                              iti = iti,
                              n_rep = n_rep_extn)

extra_us_stage = expr.stage(name = 'extra_us',
                            x_pn = [[]],
                            x_bg = ['ctx'],
                            u = [['us']],
                            u_psb = ['us'],
                            order_fixed = True,
                            iti = iti,
                            n_rep = n_rep_train)

extra_us_ctl_stage = expr.stage(name = 'extra_us',
                                x_pn = [[]],
                                x_bg = ['ctx'],
                                u_psb = ['us'],
                                order_fixed = True,
                                iti = 0,
                                n_rep = (iti + 1)*n_rep_train)

##### DEFINE SCHEDULES #####

# no conditioning (as a very basic control)
no_cond = expr.schedule(name = 'no_conditioning', resp_type = 'exct', stage_list = [test_stage])

# basic conditioning
cond = expr.schedule(name = 'conditioning', resp_type = 'exct', stage_list = [training_stage, test_stage])

# blocking
blocking = expr.schedule(name = 'blocking', resp_type = 'exct', stage_list = [one_cue_stage, two_cue_stage, test_cs2_stage])

# two cue conditioning (e.g. overshadowing)
two_cue = expr.schedule(name = 'two_cue', resp_type = 'exct', stage_list = [two_cue_stage, test_cs2_stage])

# pre-exposure (for latent inhibition)
pre_exp = expr.schedule(name = 'pre_exposure', resp_type = 'exct', stage_list = [pre_exp_stage, training_stage, test_stage])

# pre-exposure (ABC)
pre_exp_abc = expr.schedule(name = 'pre_exposure_abc', resp_type = 'exct', stage_list = [pre_exp_ctx_a_stage, training_ctx_b_stage, test_ctx_c_stage])

# extinction
extn = expr.schedule(name = 'extinction', resp_type = 'exct', stage_list = [training_stage, extn_stage, test_stage])

# extinction (AAA)
extn_aaa = expr.schedule(name = 'extinction_aaa', resp_type = 'exct', stage_list = [training_ctx_a_stage, extn_ctx_a_stage, test_ctx_a_stage])

# extinction (ABA)
extn_aba = expr.schedule(name = 'extinction_aba', resp_type = 'exct', stage_list = [training_ctx_a_stage, extn_ctx_b_stage, test_ctx_a_stage])

# extinction (ABC)
extn_abc = expr.schedule(name = 'extinction_abc', resp_type = 'exct', stage_list = [training_ctx_a_stage, extn_ctx_b_stage, test_ctx_c_stage])

# extinction (AAB)
extn_aab = expr.schedule(name = 'extinction_aab', resp_type = 'exct', stage_list = [training_ctx_a_stage, extn_ctx_a_stage, test_ctx_b_stage])

# extinction (ABB)
extn_abb = expr.schedule(name = 'extinction_abb', resp_type = 'exct', stage_list = [training_ctx_a_stage, extn_ctx_b_stage, test_ctx_b_stage])

# extinction with extra US (for reinstatement)
extn_extra_us = expr.schedule(name = 'extra_us', resp_type = 'exct', stage_list = [training_stage, extn_stage, extra_us_stage, test_stage])

# control condition for extinction with US (for reinstatement)
extn_extra_us_ctl = expr.schedule(name = 'extra_us_control', resp_type = 'exct', stage_list = [training_stage, extn_stage, extra_us_ctl_stage, test_stage])

# extinction with delay before test (explicit)
extn_delay_explicit = expr.schedule(name = 'extinction_delay', resp_type = 'exct', stage_list = [training_stage, extn_stage, delay_stage, test_stage])

# extinction with delay before test (implicit)
extn_delay_implicit = expr.schedule(name = 'extinction_delay', resp_type = 'exct', stage_list = [training_stage, extn_stage, test_stage], delays = [0, 100])

##### DEFINE BEHAVIORAL SCORES #####

cs_score = expr.behav_score(stage = 'test',
                            trial_pos = ['cs -> nothing'],
                            resp_pos = ['us'])

cs1_score = expr.behav_score(stage = 'test',
                             trial_pos = ['cs1 -> nothing'],
                             resp_pos = ['us'])

cs2_score = expr.behav_score(stage = 'test',
                             trial_pos = ['cs2 -> nothing'],
                             resp_pos = ['us'])


##### DEFINE OATS AND EXPERIMENTS #####

# basic conditioning, i.e. acquistion of a conditioned response
conditioning = expr.experiment(schedules = {'control': no_cond, 'experimental': cond},
                               oats = {'acquistion': expr.oat(schedule_pos = ['experimental'],
                                                              schedule_neg = ['control'],
                                                              behav_score_pos = cs_score,
                                                              behav_score_neg = cs_score)
})

# basic extinction of a conditioned response
extinction = expr.experiment(schedules = {'control': cond, 'experimental': extn},
                             oats = {'extinction': expr.oat(schedule_pos = ['control'],
                                                            schedule_neg = ['experimental'],
                                                            behav_score_pos = cs_score,
                                                            behav_score_neg = cs_score)
})

# ABA renewal
aba_renewal = expr.experiment(schedules = {'experimental': extn_aba, 'control': extn_aaa},
                              oats = {'renewal': expr.oat(schedule_pos = ['experimental'],
                                                          schedule_neg = ['control'],
                                                          behav_score_pos = cs_score,
                                                          behav_score_neg = cs_score)
})

# ABC renewal
abc_renewal = expr.experiment(schedules = {'experimental': extn_abc, 'control': extn_abb},
                              oats = {'renewal': expr.oat(schedule_pos = ['experimental'],
                                                          schedule_neg = ['control'],
                                                          behav_score_pos = cs_score,
                                                          behav_score_neg = cs_score)
})

# AAB renewal
aab_renewal = expr.experiment(schedules = {'experimental': extn_aab, 'control': extn_aaa},
                              oats = {'renewal': expr.oat(schedule_pos = ['experimental'],
                                                          schedule_neg = ['control'],
                                                          behav_score_pos = cs_score,
                                                          behav_score_neg = cs_score)
})

# reinstatement
reinstatement = expr.experiment(schedules = {'experimental': extn_extra_us, 'control': extn_extra_us_ctl},
                                oats = {'reinstatement': expr.oat(schedule_pos = ['experimental'],
                                                                  schedule_neg = ['control'],
                                                                  behav_score_pos = cs_score,
                                                                  behav_score_neg = cs_score)
})

# basic latent inhibition
latent_inhib = expr.experiment(schedules = {'control': cond, 'experimental': pre_exp},
                               oats = {'latent_inhibition': expr.oat(schedule_pos = ['control'],
                                                                     schedule_neg = ['experimental'],
                                                                     behav_score_pos = cs_score,
                                                                     behav_score_neg = cs_score)
})

# context dependence of latent inhibition
ctx_li = expr.experiment(schedules = {'experimental': pre_exp_abc, 'control': pre_exp},
                         oats = {'ctx_dependence_li': expr.oat(schedule_pos = ['experimental'],
                                                               schedule_neg = ['control'],
                                                               behav_score_pos = cs_score,
                                                               behav_score_neg = cs_score)
})

# overshadowing
overshadowing = expr.experiment(schedules = {'control': cond, 'experimental': two_cue},
                                oats = {'overshadowing': expr.oat(schedule_pos = ['control'],
                                                                  schedule_neg = ['experimental'],
                                                                  behav_score_pos = cs1_score,
                                                                  behav_score_neg = cs1_score)
})

# blocking
blocking = expr.experiment(schedules = {'control': two_cue, 'experimental': blocking},
                           oats = {'blocking': expr.oat(schedule_pos = ['control'],
                                                        schedule_neg = ['experimental'],
                                                        behav_score_pos = cs2_score,
                                                        behav_score_neg = cs2_score)
})

# spontaneous recovery (explicit)
spont_rec_exp = expr.experiment(schedules = {'experimental': extn_delay_explicit, 'control': extn},
                                oats = {'spontaneous_recovery': expr.oat(schedule_pos = ['experimental'],
                                                                         schedule_neg = ['control'],
                                                                         behav_score_pos = cs_score,
                                                                         behav_score_neg = cs_score)
})

# spontaneous recovery (implicit)
spont_rec_imp = expr.experiment(schedules = {'experimental': extn_delay_implicit, 'control': extn},
                                oats = {'spontaneous_recovery': expr.oat(schedule_pos = ['experimental'],
                                                                         schedule_neg = ['control'],
                                                                         behav_score_pos = cs_score,
                                                                         behav_score_neg = cs_score)
})