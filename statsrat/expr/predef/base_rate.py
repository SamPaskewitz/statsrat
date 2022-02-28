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

##### NOVEL TEST CUES IN THE INVERSE BASE RATE EFFECT DESIGN #####

# Juslin, Wennerholm and Winman (2001), Experiment 1
# This is a basic inverse base rate effect design with added novel cue test trials.
# Participants preferred the rare outcomes on the novel test trials, which 
# Winman et al took this as evidence for their eliminative inference explanation of the IBRE.
# I'm only including the 3:1 (common:rare) base rate ratio condition.
# Results were similar in the 7:1 condition DOUBLE CHECK.
# I'm only including the two most important the test trial types, viz. conflicting (PC + PC) and novel cue.

# UPDATE CODE!

design = expr.schedule(resp_type = 'choice',
                  stages = {
                      'training': expr.stage(
                            freq = 3*[3, 1],
                            x_pn = [['i1', 'pc1'], ['i1', 'pr1'], ['i2', 'pc2'], ['i2', 'pr2'], ['i3', 'pc3'], ['i3', 'pr3']],
                            y = [['c1'], ['r1'], ['c2'], ['r2'], ['c3'], ['r3']],
                            y_psb = ['c1', 'r1', 'c2', 'r2', 'c3', 'r3'],
                            n_rep = 15),
                      'test': expr.stage(
                            x_pn = [['pc1', 'pr1'], ['pc2', 'pr2'], ['pc3', 'pr3'], ['novel1'], ['novel2'], ['novel3']],
                            y_psb = ['c1', 'r1', 'c2', 'r2', 'c3', 'r3'],
                            lrn = False,
                            n_rep = 3)
                  })

ibre = expr.oat(schedule_pos = ['design'],
                behav_score_pos = expr.behav_score(stage = 'test',
                                              trial_pos = ['pc1.pr1 -> nothing', 'pc2.pr2 -> nothing', 'pc3.pr3 -> nothing'],
                                              trial_neg = ['pc1.pr1 -> nothing', 'pc2.pr2 -> nothing', 'pc3.pr3 -> nothing'],
                                              resp_pos = ['r1', 'r2', 'r3'],
                                              resp_neg = ['c1', 'c2', 'c3']))

# I NEED TO ADD FUNCTIONALITY TO ALLOW FOR MORE THAN ONE POSITIVE OR NEGATIVE RESPONSE FOR EACH TRIAL TYPE.
novel = expr.oat(schedule_pos = ['design'],
                 behav_score_pos = expr.behav_score(stage = 'test',
                                              trial_pos = ['novel1 -> nothing', 'novel2 -> nothing', 'novel3 -> nothing'],
                                              trial_neg = ['novel1 -> nothing', 'novel2 -> nothing', 'novel3 -> nothing'],
                                              resp_pos = ['r1', 'r2', 'r3'], # NEED TO CHANGE
                                              resp_neg = ['c1', 'c2', 'c3'])) # NEED TO CHANGE

novel = expr.experiment(schedules = {'design': design},
                             oats = {'ibre': ibre, 'novel': novel})

del design; del novel


##### ATTENTIONAL TRANSFER AFTER THE INVERSE BASE RATE EFFECT #####

# Don and Livesey (2021), Experiment 2
# Note that I am only including the most important type of test trial ('conflicting').
# There are two factors: length and design (standard vs. balanced).

standard_short = expr.schedule(resp_type = 'choice',
                              stages = {
                                  'training': expr.stage(
                                      freq = 4*[3, 1],
                                        x_pn = [['i1', 'pc1'], ['i1', 'pr1'], ['i2', 'pc2'], ['i2', 'pr2'], ['i3', 'pc3'], ['i3', 'pr3'], ['i4', 'pc4'], ['i4', 'pr4']],
                                        y = 4*[['o1'], ['o2']],
                                        y_psb = ['o1', 'o2'],
                                        n_rep = 3),
                                  'transfer': expr.stage(x_pn = [['pc1', 'pr3'], ['pc3', 'pr1'], ['pc2', 'pr4'], ['pc4', 'pr2']],
                                                         y = [['o3'], ['o4'], ['o3'], ['o4']],
                                                         y_psb = ['o3', 'o4'],
                                                         n_rep = 3),
                                  'test': expr.stage(
                                        x_pn = [['pc1', 'pr1'], ['pc2', 'pr2'], ['pc3', 'pr3'], ['pc4', 'pr4']],
                                        y_psb = ['o3', 'o4'],
                                        lrn = False,
                                        n_rep = 1)
                              })

balanced_short = expr.schedule(resp_type = 'choice',
                              stages = {
                                  'training': expr.stage(
                                        freq = 4*[3, 1],
                                        x_pn = [['i1', 'pc1'], ['i1', 'pr1'], ['i2', 'pc2'], ['i2', 'pr2'], ['i3', 'pc3'], ['i3', 'pr3'], ['i4', 'pc4'], ['i4', 'pr4']],
                                        y = 2*[['o1'], ['o2'], ['o2'], ['o1']],
                                        y_psb = ['o1', 'o2'],
                                        n_rep = 3),
                                  'transfer': expr.stage(x_pn = [['pc1', 'pr3'], ['pc3', 'pr1'], ['pc2', 'pr4'], ['pc4', 'pr2']],
                                                         y = [['o3'], ['o4'], ['o3'], ['o4']],
                                                         y_psb = ['o3', 'o4'],
                                                         n_rep = 3),
                                  'test': expr.stage(
                                        x_pn = [['pc1', 'pr1'], ['pc2', 'pr2'], ['pc3', 'pr3'], ['pc4', 'pr4']],
                                        y_psb = ['o3', 'o4'],
                                        lrn = False,
                                        n_rep = 1)
                              })

standard_long = expr.schedule(resp_type = 'choice',
                              stages = {
                                  'training': expr.stage(
                                      freq = 4*[3, 1],
                                        x_pn = [['i1', 'pc1'], ['i1', 'pr1'], ['i2', 'pc2'], ['i2', 'pr2'], ['i3', 'pc3'], ['i3', 'pr3'], ['i4', 'pc4'], ['i4', 'pr4']],
                                        y = 4*[['o1'], ['o2']],
                                        y_psb = ['o1', 'o2'],
                                        n_rep = 7),
                                  'transfer': expr.stage(x_pn = [['pc1', 'pr3'], ['pc3', 'pr1'], ['pc2', 'pr4'], ['pc4', 'pr2']],
                                                         y = [['o3'], ['o4'], ['o3'], ['o4']],
                                                         y_psb = ['o3', 'o4'],
                                                         n_rep = 3),
                                  'test': expr.stage(
                                        x_pn = [['pc1', 'pr1'], ['pc2', 'pr2'], ['pc3', 'pr3'], ['pc4', 'pr4']],
                                        y_psb = ['o3', 'o4'],
                                        lrn = False,
                                        n_rep = 1)
                              })

balanced_long = expr.schedule(resp_type = 'choice',
                              stages = {
                                  'training': expr.stage(
                                        freq = 4*[3, 1],
                                        x_pn = [['i1', 'pc1'], ['i1', 'pr1'], ['i2', 'pc2'], ['i2', 'pr2'], ['i3', 'pc3'], ['i3', 'pr3'], ['i4', 'pc4'], ['i4', 'pr4']],
                                        y = 2*[['o1'], ['o2'], ['o2'], ['o1']],
                                        y_psb = ['o1', 'o2'],
                                        n_rep = 7),
                                  'transfer': expr.stage(x_pn = [['pc1', 'pr3'], ['pc3', 'pr1'], ['pc2', 'pr4'], ['pc4', 'pr2']],
                                                         y = [['o3'], ['o4'], ['o3'], ['o4']],
                                                         y_psb = ['o3', 'o4'],
                                                         n_rep = 3),
                                  'test': expr.stage(
                                        x_pn = [['pc1', 'pr1'], ['pc2', 'pr2'], ['pc3', 'pr3'], ['pc4', 'pr4']],
                                        y_psb = ['o3', 'o4'],
                                        lrn = False,
                                        n_rep = 1)
                              })

# Attention difference between PC and PR cues, assessed via the transfer associations
transfer_score = expr.behav_score(stage = 'test',
                                  trial_pos = ['pc1.pr1 -> nothing', 'pc2.pr2 -> nothing', 'pc3.pr3 -> nothing', 'pc4.pr4 -> nothing'],
                                  trial_neg = ['pc1.pr1 -> nothing', 'pc2.pr2 -> nothing', 'pc3.pr3 -> nothing', 'pc4.pr4 -> nothing'],
                                  resp_pos = ['o4', 'o4', 'o3', 'o3'],
                                  resp_neg = ['o3', 'o3', 'o4', 'o4'])

short_vs_long = expr.oat(schedule_pos = ['standard_short', 'balanced_short'],
                         schedule_neg = ['standard_long', 'balanced_long'],
                         behav_score_pos = transfer_score,
                         behav_score_neg = transfer_score)

ibre_transfer = expr.experiment(schedules = {'standard_short': standard_short, 'balanced_short': balanced_short, 'standard_long': standard_long, 'balanced_long': balanced_long},
                                oats = {'short_vs_long': short_vs_long})


##### STANDARD VS. BALANCED INVERSE BASE RATE EFFECT DESIGNS #####

# Don and Livesey (2021), Experiment 2
# Note that I am only including the most important type of test trial ('conflicting').
# The main result is a stronger inverse base rate effect in the Standard condition than in the Balanced condition.
# The Balanced condition did not in fact show a detectable IBRE at all.
# This is a shorter version of designs from previous papers (similar to Experiment 2 from Don & Livesey, 2017).

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
                                      resp_pos = 4*['o2'],
                                      resp_neg = 4*['o1'])

# PC associated outcome vs. PR associated outcome in the Balanced condition (measures the IBRE)
ibre_score_balanced = expr.behav_score(stage = 'test',
                                      trial_pos = ['pc1.pr1 -> nothing', 'pc2.pr2 -> nothing', 'pc3.pr3 -> nothing', 'pc4.pr4 -> nothing'],
                                      trial_neg = ['pc1.pr1 -> nothing', 'pc2.pr2 -> nothing', 'pc3.pr3 -> nothing', 'pc4.pr4 -> nothing'],
                                      resp_pos = 2*['o2', 'o1'],
                                      resp_neg = 2*['o1', 'o2'])

ibre_standard = expr.oat(schedule_pos = ['standard'],
                         behav_score_pos = ibre_score_standard)

ibre_balanced = expr.oat(schedule_pos = ['balanced'],
                         behav_score_pos = ibre_score_balanced)

contrast = expr.oat(schedule_pos = ['standard'],
                    schedule_neg = ['balanced'],
                    behav_score_pos = ibre_score_standard,
                    behav_score_neg = ibre_score_balanced)

standard_vs_balanced = expr.experiment(schedules = {'standard': standard, 'balanced': balanced},
                                       oats = {'ibre_standard': ibre_standard,
                                               'ibre_balanced': ibre_balanced,
                                               'ibre_between_groups': contrast})
## ADD NOTES ##

##### GLOBAL BASE RATE DIFFERENCE INSUFFICIENT #####

# Don and Livesey (2017), Experiment 3
# The Standard condition is an ordinary IBRE design.
# In the Equal Trial condition, there is no difference in trial type base rates, but filler trials (f1 + f2 -> o1, f3 + f4 -> 03) make one outcome more common than the other.
# There was no IBRE in the Equal Trial condition.
# For simplicity, I only include the conflicting type test trials.

standard = expr.schedule(resp_type = 'choice',
                          stages = {
                              'training': expr.stage(
                                    freq = 4*[3, 1],
                                    x_pn = [['i1', 'pc1'], ['i1', 'pr1'], ['i2', 'pc2'], ['i2', 'pr2'], ['i3', 'pc3'], ['i3', 'pr3'], ['i4', 'pc4'], ['i4', 'pr4']],
                                    y = [['o1'], ['o2'], ['o1'], ['o2'], ['o3'], ['o4'], ['o3'], ['o4']],
                                    y_psb = ['o1', 'o2', 'o3', 'o4'],
                                    n_rep = 7),
                              'test': expr.stage(
                                    x_pn = [['pc1', 'pr1'], ['pc2', 'pr2'], ['pc3', 'pr3'], ['pc4', 'pr4']],
                                    y_psb = ['o1', 'o2', 'o3', 'o4'],
                                    lrn = False,
                                    n_rep = 1)
                          })

equal_trial = expr.schedule(resp_type = 'choice',
                              stages = {
                                  'training': expr.stage(
                                        freq = 8*[1] + 2*[4],
                                        x_pn = [['i1', 'pc1'], ['i1', 'pr1'], ['i2', 'pc2'], ['i2', 'pr2'], ['i3', 'pc3'], ['i3', 'pr3'], ['i4', 'pc4'], ['i4', 'pr4'], ['f1', 'f2'], ['f3', 'f4']],
                                        y = [['o1'], ['o2'], ['o1'], ['o2'], ['o3'], ['o4'], ['o3'], ['o4'], ['o1'], ['o3']],
                                        y_psb = ['o1', 'o2', 'o3', 'o4'],
                                        n_rep = 7),
                                  'test': expr.stage(
                                        x_pn = [['pc1', 'pr1'], ['pc2', 'pr2'], ['pc3', 'pr3'], ['pc4', 'pr4']],
                                        y_psb = ['o1', 'o2', 'o3', 'o4'],
                                        lrn = False,
                                        n_rep = 1)
                              })

# rare vs. common outcome (measures the IBRE)
ibre_score = expr.behav_score(stage = 'test',
                              trial_pos = ['pc1.pr1 -> nothing', 'pc2.pr2 -> nothing', 'pc3.pr3 -> nothing', 'pc4.pr4 -> nothing'],
                              trial_neg = ['pc1.pr1 -> nothing', 'pc2.pr2 -> nothing', 'pc3.pr3 -> nothing', 'pc4.pr4 -> nothing'],
                              resp_pos = ['o2', 'o2', 'o4', 'o4'],
                              resp_neg = ['o1', 'o1', 'o3', 'o3'])

ibre_standard = expr.oat(schedule_pos = ['standard'],
                         behav_score_pos = ibre_score)

ibre_equal_trial = expr.oat(schedule_pos = ['equal_trial'],
                            behav_score_pos = ibre_score)

contrast = expr.oat(schedule_pos = ['standard'],
                    schedule_neg = ['equal_trial'],
                    behav_score_pos = ibre_score,
                    behav_score_neg = ibre_score)

standard_vs_equal_trial = expr.experiment(schedules = {'standard': standard, 'equal_trial': equal_trial},
                                           oats = {'ibre_standard': ibre_standard,
                                                   'ibre_equal_trial': ibre_equal_trial,
                                                   'ibre_between_groups': contrast})
## ADD NOTES ##

