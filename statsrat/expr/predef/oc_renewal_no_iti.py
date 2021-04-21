"""
Occasion setting by extinction context (based loosely on Harris et al 2000, Experiment 1).
I've given the CSs and contexts different labels from those in the Harris et al paper.

This version does not include inter-trial intervals (ITIs).

Notes on Harris et al 2000, Experiment 1
----------------------------------------

The procedure was conditioned freezing, rather than conditioned supression of bar pressing.

Conditioning took place on day 1, and consisted of one trial per CS (order counterbalanced).
The ITI was 3 minutes (6 times the CS length).

Extinction took place over two days (days 2 and 3), with extinction of both
CSs on each day in the order cs1, cs2, cs2, cs1.  There were 8 trials per session.
The ITI was 2 minutes (4 times the CS length).

The test took place on day 4, and consisted of 4 trials.  The ITI was 2 minutes (4 times the CS length).
"""

from statsrat import expr

iti = 0
n_rep_train = 5
n_rep_extn = 10
n_rep_test = 5

training = expr.stage(x_pn = [['cs1'], ['cs2']],
                      x_bg = ['ctx_a'],
                      u = 2*[['us']],
                      u_psb = ['us'],
                      order_fixed = True,
                      iti = iti,
                      n_rep = n_rep_train)

extinction_cs1 = expr.stage(x_pn = [['cs1']],
                            x_bg = ['ctx_b1'],
                            u_psb = ['us'],
                            order_fixed = True,
                            iti = iti,
                            n_rep = n_rep_extn)

extinction_cs2 = expr.stage(x_pn = [['cs2']],
                            x_bg = ['ctx_b2'],
                            u_psb = ['us'],
                            order_fixed = True,
                            iti = iti,
                            n_rep = n_rep_extn)

test_same = expr.stage(x_pn = [['cs1']],
                       x_bg = ['ctx_b1'],
                       u_psb = ['us'],
                       order_fixed = True,
                       iti = iti,
                       n_rep = n_rep_test)

test_different = expr.stage(x_pn = [['cs1']],
                            x_bg = ['ctx_b2'],
                            u_psb = ['us'],
                            order_fixed = True,
                            iti = iti,
                            n_rep = n_rep_test)

# group "same"
same = expr.schedule(resp_type = 'exct', stages = {'training': training, 'ex_cs1': extinction_cs1, 'ex_cs2': extinction_cs2, 'ex_cs2': extinction_cs2, 'ex_cs1': extinction_cs1, 'test': test_same})

# group "different"
different = expr.schedule(resp_type = 'exct', stages = {'training': training, 'ex_cs1': extinction_cs1, 'ex_cs2': extinction_cs2, 'ex_cs2': extinction_cs2, 'ex_cs1': extinction_cs1, 'test': test_different})

# behavioral score
cs1_score = expr.behav_score(stage = 'test',
                             trial_pos = ['cs1 -> nothing'],
                             resp_pos = 2*['us'])

# experiment object
oc_renewal = expr.experiment(schedules = {'same': same, 'different': different},
                             oats = {'renewal': expr.oat(schedule_pos = ['different'],
                                                         schedule_neg = ['same'],
                                                         behav_score_pos = cs1_score,
                                                         behav_score_neg = cs1_score)})