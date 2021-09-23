from statsrat import expr
'''
Inverse base rate effect (IBRE) and related experiments.
'''

##### BASIC INVERSE BASE RATE EFFECT DESIGN #####

# Kruschke (1996), Experiment 1 (inverse base rate effect)
# I'm missing some of the test trial types.
design = expr.schedule(resp_type = 'choice',
                  stages = {
                      'training': expr.stage(
                            freq = [3, 1, 3, 1],
                            x_pn = [['i1', 'pc1'], ['i1', 'pr1'], ['i2', 'pc2'], ['i2', 'pr2']],
                            y = [['c1'], ['r1'], ['c2'], ['r2']],
                            y_psb = ['c1', 'r1', 'c2', 'r2'],
                            n_rep = 15),
                      'test': expr.stage(
                            x_pn = [['pc1'], ['pr1'], ['pc2'], ['pr2'], ['pc1', 'pr1'], ['pc2', 'pr2']],
                            y_psb = ['c1', 'r1', 'c2', 'r2'],
                            lrn = False,
                            n_rep = 2)
                  })

pc_pr = expr.oat(schedule_pos = ['design'],
            behav_score_pos = expr.behav_score(stage = 'test',
                                          trial_pos = ['pc1.pr1 -> nothing', 'pc2.pr2 -> nothing'],
                                          trial_neg = ['pc1.pr1 -> nothing', 'pc2.pr2 -> nothing'],
                                          resp_pos = ['r1', 'r2'],
                                          resp_neg = ['c1', 'c2']))

basic_ibre = expr.experiment(schedules = {'design': design},
                             oats = {'pc_pr': pc_pr})

del design; del pc_pr

##### STANDARD VS. BALANCED INVERSE BASE RATE EFFECT DESIGNS #####

# Don and Livesey (2021), Experiment 2
# Note that I am only including the most important type of test trial ('conflicting').
# The main result is a stronger inverse base rate effect in the Standard condition than in the Balanced condition.
# The latter did not in fact show a detectable IBRE at all.
# This is a shorter version of designs from previous papers ** READ THESE **.

standard = expr.schedule(resp_type = 'choice',
                          stages = {
                              'training': expr.stage(
                                    freq = 4*[3, 1],
                                    x_pn = [['i1', 'pc1'], ['i1', 'pr1'], ['i2', 'pc2'], ['i2', 'pr2'], ['i3', 'pc3'], ['i3', 'pr3'], ['i4', 'pc4'], ['i4', 'pr4']],
                                    y = 4*[['o1'], ['o2']],
                                    y_psb = ['o1', 'o2'],
                                    n_rep = 6),
                              'test': expr.stage(
                                    x_pn = [['pc1', 'pr1'], ['pc2', 'pr2'], ['pc3', 'pr3'], ['pc4', 'pr4']],
                                    y_psb = ['o1', 'o2'],
                                    lrn = False,
                                    n_rep = 1)
                          })

balanced = expr.schedule(resp_type = 'choice',
                          stages = {
                              'training': expr.stage(
                                    freq = 4*[3, 1],
                                    x_pn = [['i1', 'pc1'], ['i1', 'pr1'], ['i2', 'pc2'], ['i2', 'pr2'], ['i3', 'pc3'], ['i3', 'pr3'], ['i4', 'pc4'], ['i4', 'pr4']],
                                    y = 2*[['o1'], ['o2'], ['o2'], ['o1']],
                                    y_psb = ['o1', 'o2'],
                                    n_rep = 6),
                              'test': expr.stage(
                                    x_pn = [['pc1', 'pr1'], ['pc2', 'pr2'], ['pc3', 'pr3'], ['pc4', 'pr4']],
                                    y_psb = ['o1', 'o2'],
                                    lrn = False,
                                    n_rep = 1)
                          })

# PC associated outcome vs. PR associated outcome in the Standard condition (measures the IBRE)
ibre_score_standard = expr.behav_score(stage = 'test',
                                      trial_pos = ['pc1.pr1 -> nothing', 'pc2.pr2 -> nothing', 'pc3.pr3 -> nothing', 'pc4.pr4 -> nothing'],
                                      trial_neg = ['pc1.pr1 -> nothing', 'pc2.pr2 -> nothing', 'pc3.pr3 -> nothing', 'pc4.pr4 -> nothing'],
                                      resp_pos = 8*['o1'],
                                      resp_neg = 8*['o2'])

# PC associated outcome vs. PR associated outcome in the Balanced condition (measures the IBRE)
ibre_score_balanced = expr.behav_score(stage = 'test',
                                      trial_pos = ['pc1.pr1 -> nothing', 'pc2.pr2 -> nothing', 'pc3.pr3 -> nothing', 'pc4.pr4 -> nothing'],
                                      trial_neg = ['pc1.pr1 -> nothing', 'pc2.pr2 -> nothing', 'pc3.pr3 -> nothing', 'pc4.pr4 -> nothing'],
                                      resp_pos = 4*['o1', 'o2'],
                                      resp_neg = 4*['o2', 'o1'])

contrast = expr.oat(schedule_pos = ['standard'],
                    schedule_neg = ['balanced'],
                    behav_score_pos = ibre_score_standard,
                    behav_score_neg = ibre_score_balanced)

standard_vs_balanced = expr.experiment(schedules = {'standard': standard, 'balanced': balanced},
                                       oats = {'ibre_between_groups': contrast})
## ADD NOTES ##