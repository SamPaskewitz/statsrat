import pandas as pd
from statsrat import expr

'''
Simplified Pavlovian conditioning experiments (same designs as pvl_iti and pvl_no_iti).

This version includes inter-trial intervals (ITIs) and a continuous time cue.
Because of the continuous time cue, this is designed for use with exemplar models
with e.g. a Gaussian or exponential similarity function.
'''

iti = 5
n_rep_train = 5
n_rep_extn = 5
n_rep_test = 2
n_rep_pre_exp = 5
n_rep_no_stim = 10

##### DEFINE STAGES #####

no_stim_stage_t0 = expr.stage(x_pn = [[]],
                              x_bg = ['ctx', 'time'],
                              x_value = pd.Series({'cs': 1.0, 'ctx': 1.0, 'time': 0.0}),
                              u_psb = ['us'],
                              order_fixed = True,
                              iti = 0,
                              n_rep = n_rep_no_stim)

no_stim_ctx_a_stage_t0 = expr.stage(x_pn = [[]],
                                    x_bg = ['ctx_a', 'time'],
                                    x_value = pd.Series({'cs': 1.0, 'ctx_a': 1.0, 'time': 0.0}),
                                    u_psb = ['us'],
                                    order_fixed = True,
                                    iti = 0,
                                    n_rep = n_rep_no_stim)

test_stage_t0 = expr.stage(x_pn = [['cs']],
                           x_bg = ['ctx', 'time'],
                           x_value = pd.Series({'cs': 1.0, 'ctx': 1.0, 'time': 0.0}),
                           u_psb = ['us'],
                           order_fixed = True,
                           iti = iti,
                           n_rep = n_rep_test)

test_stage_t1 = expr.stage(x_pn = [['cs']],
                           x_bg = ['ctx', 'time'],
                           x_value = pd.Series({'cs': 1.0, 'ctx': 1.0, 'time': 1.0}),
                           u_psb = ['us'],
                           order_fixed = True,
                           iti = iti,
                           n_rep = n_rep_test)

test_stage_t2 = expr.stage(x_pn = [['cs']],
                           x_bg = ['ctx', 'time'],
                           x_value = pd.Series({'cs': 1.0, 'ctx': 1.0, 'time': 2.0}),
                           u_psb = ['us'],
                           order_fixed = True,
                           iti = iti,
                           n_rep = n_rep_test)

test_stage_t3 = expr.stage(x_pn = [['cs']],
                           x_bg = ['ctx', 'time'],
                           x_value = pd.Series({'cs': 1.0, 'ctx': 1.0, 'time': 3.0}),
                           u_psb = ['us'],
                           order_fixed = True,
                           iti = iti,
                           n_rep = n_rep_test)

test_stage_t5 = expr.stage(x_pn = [['cs']],
                           x_bg = ['ctx', 'time'],
                           x_value = pd.Series({'cs': 1.0, 'ctx': 1.0, 'time': 5.0}),
                           u_psb = ['us'],
                           order_fixed = True,
                           iti = iti,
                           n_rep = n_rep_test)

test_cs2_stage_t1 = expr.stage(x_pn = [['cs2']],
                               x_bg = ['ctx', 'time'],
                               x_value = pd.Series({'cs2': 1.0, 'ctx': 1.0, 'time': 1.0}),
                               u_psb = ['us'],
                               order_fixed = False,
                               iti = iti,
                               n_rep = n_rep_test)

test_cs2_stage_t2 = expr.stage(x_pn = [['cs2']],
                               x_bg = ['ctx', 'time'],
                               x_value = pd.Series({'cs2': 1.0, 'ctx': 1.0, 'time': 2.0}),
                               u_psb = ['us'],
                               order_fixed = False,
                               iti = iti,
                               n_rep = n_rep_test)

test_ctx_a_stage_t2 = expr.stage(x_pn = [['cs']],
                                 x_bg = ['ctx_a', 'time'],
                                 x_value = pd.Series({'cs': 1.0, 'ctx_a': 1.0, 'time': 2.0}),
                                 u_psb = ['us'],
                                 order_fixed = True,
                                 iti = iti,
                                 n_rep = n_rep_test)

test_ctx_b_stage_t2 = expr.stage(x_pn = [['cs']],
                                 x_bg = ['ctx_b', 'time'],
                                 x_value = pd.Series({'cs': 1.0, 'ctx_b': 1.0, 'time': 2.0}),
                                 u_psb = ['us'],
                                 order_fixed = True,
                                 iti = iti,
                                 n_rep = n_rep_test)

test_ctx_c_stage_t2 = expr.stage(x_pn = [['cs']],
                                 x_bg = ['ctx_c', 'time'],
                                 x_value = pd.Series({'cs': 1.0, 'ctx_c': 1.0, 'time': 2.0}),
                                 u_psb = ['us'],
                                 order_fixed = True,
                                 iti = iti,
                                 n_rep = n_rep_test)

one_cue_stage_t0 = expr.stage(x_pn = [['cs1']],
                              x_bg = ['ctx', 'time'],
                              x_value = pd.Series({'cs1': 1.0, 'ctx': 1.0, 'time': 0.0}),
                              u = [['us']],
                              u_psb = ['us'],
                              order_fixed = False,
                              iti = iti,
                              n_rep = n_rep_train)

two_cue_stage_t0 = expr.stage(x_pn = [['cs1', 'cs2']],
                              x_bg = ['ctx', 'time'],
                              x_value = pd.Series({'cs1': 1.0, 'cs2': 1.0, 'ctx': 1.0, 'time': 0.0}),
                              u = [['us']],
                              u_psb = ['us'],
                              order_fixed = False,
                              iti = iti,
                              n_rep = n_rep_train)

two_cue_stage_t1 = expr.stage(x_pn = [['cs1', 'cs2']],
                              x_bg = ['ctx', 'time'],
                              x_value = pd.Series({'cs1': 1.0, 'cs2': 1.0, 'ctx': 1.0, 'time': 1.0}),
                              u = [['us']],
                              u_psb = ['us'],
                              order_fixed = False,
                              iti = iti,
                              n_rep = n_rep_train)

pre_exp_stage_t0 = expr.stage(x_pn = [['cs']],
                              x_bg = ['ctx', 'time'],
                              x_value = pd.Series({'cs': 1.0, 'ctx': 1.0, 'time': 0.0}),
                              u_psb = ['us'],
                              order_fixed = True,
                              iti = iti,
                              n_rep = n_rep_pre_exp)

pre_exp_ctx_a_stage_t0 = expr.stage(x_pn = [['cs']],
                                    x_bg = ['ctx_a', 'time'],
                                    x_value = pd.Series({'cs': 1.0, 'ctx_a': 1.0, 'time': 0.0}),
                                    u_psb = ['us'],
                                    order_fixed = True,
                                    iti = iti,
                                    n_rep = n_rep_pre_exp)

training_stage_t0 = expr.stage(x_pn = [['cs']],
                               x_bg = ['ctx', 'time'],
                               x_value = pd.Series({'cs': 1.0, 'ctx': 1.0, 'time': 0.0}),
                               u = [['us']],
                               u_psb = ['us'],
                               order_fixed = True,
                               iti = iti,
                               n_rep = n_rep_train)

training_stage_t1 = expr.stage(x_pn = [['cs']],
                               x_bg = ['ctx', 'time'],
                               x_value = pd.Series({'cs': 1.0, 'ctx': 1.0, 'time': 1.0}),
                               u = [['us']],
                               u_psb = ['us'],
                               order_fixed = True,
                               iti = iti,
                               n_rep = n_rep_train)

training_ctx_a_stage_t0 = expr.stage(x_pn = [['cs']],
                                     x_bg = ['ctx_a', 'time'],
                                     x_value = pd.Series({'cs': 1.0, 'ctx_a': 1.0, 'time': 0.0}),
                                     u = [['us']],
                                     u_psb = ['us'],
                                     order_fixed = True,
                                     iti = iti,
                                     n_rep = n_rep_train)

training_ctx_b_stage_t0 = expr.stage(x_pn = [['cs']],
                                     x_bg = ['ctx_b', 'time'],
                                     x_value = pd.Series({'cs': 1.0, 'ctx_b': 1.0, 'time': 0.0}),
                                     u = [['us']],
                                     u_psb = ['us'],
                                     order_fixed = True,
                                     iti = iti,
                                     n_rep = n_rep_train)

training_ctx_b_stage_t1 = expr.stage(x_pn = [['cs']],
                                     x_bg = ['ctx_b', 'time'],
                                     x_value = pd.Series({'cs': 1.0, 'ctx_b': 1.0, 'time': 1.0}),
                                     u = [['us']],
                                     u_psb = ['us'],
                                     order_fixed = True,
                                     iti = iti,
                                     n_rep = n_rep_train)

extn_stage_t1 = expr.stage(x_pn = [['cs']],
                           x_bg = ['ctx', 'time'],
                           x_value = pd.Series({'cs': 1.0, 'ctx': 1.0, 'time': 1.0}),
                           u_psb = ['us'],
                           order_fixed = True,
                           iti = iti,
                           n_rep = n_rep_extn)

extn_ctx_a_stage_t1 = expr.stage(x_pn = [['cs']],
                                 x_bg = ['ctx_a', 'time'],
                                 x_value = pd.Series({'cs': 1.0, 'ctx_a': 1.0, 'time': 1.0}),
                                 u_psb = ['us'],
                                 order_fixed = True,
                                 iti = iti,
                                 n_rep = n_rep_extn)

extn_ctx_b_stage_t1 = expr.stage(x_pn = [['cs']],
                                 x_bg = ['ctx_b', 'time'],
                                 x_value = pd.Series({'cs': 1.0, 'ctx_b': 1.0, 'time': 1.0}),
                                 u_psb = ['us'],
                                 order_fixed = True,
                                 iti = iti,
                                 n_rep = n_rep_extn)

extra_us_stage_t2 = expr.stage(x_pn = [[]],
                               x_bg = ['ctx', 'time'],
                               x_value = pd.Series({'cs': 1.0, 'ctx': 1.0, 'time': 2.0}),
                               u = [['us']],
                               u_psb = ['us'],
                               order_fixed = True,
                               iti = iti,
                               n_rep = n_rep_train)

extra_us_ctl_stage_t2 = expr.stage(x_pn = [[]],
                                   x_bg = ['ctx', 'time'],
                                   x_value = pd.Series({'cs': 1.0, 'ctx': 1.0, 'time': 2.0}),
                                   u_psb = ['us'],
                                   order_fixed = True,
                                   iti = 0,
                                   n_rep = (iti + 1)*n_rep_train)

##### DEFINE SCHEDULES #####

# no conditioning (as a very basic control)
no_cond = expr.schedule(resp_type = 'exct', stages = {'test': test_stage_t0})

# basic conditioning
cond = expr.schedule(resp_type = 'exct', stages = {'cond': training_stage_t0, 'test': test_stage_t1})

# blocking
blocking = expr.schedule(resp_type = 'exct', stages = {'one_cue': one_cue_stage_t0, 'two_cue': two_cue_stage_t1, 'test': test_cs2_stage_t2})

# two cue conditioning (e.g. overshadowing)
two_cue = expr.schedule(resp_type = 'exct', stages = {'two_cue': two_cue_stage_t0, 'test': test_cs2_stage_t1})

# pre-exposure (for latent inhibition)
pre_exp = expr.schedule(resp_type = 'exct', stages = {'pre_exposure': pre_exp_stage_t0, 'cond': training_stage_t1, 'test': test_stage_t2})

# pre-exposure (ABC)
pre_exp_abc = expr.schedule(resp_type = 'exct', stages = {'pre_exposure': pre_exp_ctx_a_stage_t0, 'cond': training_ctx_b_stage_t1, 'test': test_ctx_c_stage_t2})

# extinction
extn = expr.schedule(resp_type = 'exct', stages = {'cond': training_stage_t0, 'extinction': extn_stage_t1, 'test': test_stage_t2})

# extinction (AAA)
extn_aaa = expr.schedule(resp_type = 'exct', stages = {'cond': training_ctx_a_stage_t0, 'extinction': extn_ctx_a_stage_t1, 'test': test_ctx_a_stage_t2})

# extinction (ABA)
extn_aba = expr.schedule(resp_type = 'exct', stages = {'cond': training_ctx_a_stage_t0, 'extinction': extn_ctx_b_stage_t1, 'test': test_ctx_a_stage_t2})

# extinction (ABC)
extn_abc = expr.schedule(resp_type = 'exct', stages = {'cond': training_ctx_a_stage_t0, 'extinction': extn_ctx_b_stage_t1, 'test': test_ctx_c_stage_t2})

# extinction (AAB)
extn_aab = expr.schedule(resp_type = 'exct', stages = {'cond': training_ctx_a_stage_t0, 'extinction': extn_ctx_a_stage_t1, 'test': test_ctx_b_stage_t2})

# extinction (ABB)
extn_abb = expr.schedule(resp_type = 'exct', stages = {'cond': training_ctx_a_stage_t0, 'extinction': extn_ctx_b_stage_t1, 'test': test_ctx_b_stage_t2})

# extinction with extra US (for reinstatement)
extn_extra_us = expr.schedule(resp_type = 'exct', stages = {'cond': training_stage_t0, 'extinction': extn_stage_t1, 'extra_us': extra_us_stage_t2, 'test': test_stage_t3})

# control condition for extinction with US (for reinstatement)
extn_extra_us_ctl = expr.schedule(resp_type = 'exct', stages = {'cond': training_stage_t0, 'extinction': extn_stage_t1, 'extra_us': extra_us_ctl_stage_t2, 'test': test_stage_t3})

# extinction with delay before test (represented by time cue)
extn_delay = expr.schedule(resp_type = 'exct', stages = {'cond': training_stage_t0, 'extinction': extn_stage_t1, 'test': test_stage_t5})

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

# spontaneous recovery
spont_rec = expr.experiment(schedules = {'experimental': extn_delay, 'control': extn},
                            oats = {'spontaneous_recovery': expr.oat(schedule_pos = ['experimental'],
                                                                     schedule_neg = ['control'],
                                                                     behav_score_pos = cs_score,
                                                                     behav_score_neg = cs_score)
})