from statsrat import expr
'''
Category learning tasks.
'''

# FAST (Joel Stoddard collab)
design = expr.schedule(resp_type = 'choice',
                  stages = {
                      'tutorial_0a': expr.stage(
                            x_pn = [['alpha'], ['beta']],
                            u = [['cati'], ['catii']],
                            u_psb = ['cati', 'catii'],
                            n_rep = 6),
                      'tutorial_0b': expr.stage(
                            x_pn = [['alpha'], ['beta']],
                            u = [['cati'], ['catii']],
                            u_psb = ['cati', 'catii'],
                            n_rep = 6),
                      'tutorial_0c': expr.stage(
                            x_pn = [['alpha', 'theta'], ['beta', 'theta'], ['alpha', 'phi'], ['beta', 'phi']],
                            u = 2*[['cati'], ['catii']],
                            u_psb = ['cati', 'catii'],
                            n_rep = 6),
                      'relevance': expr.stage(
                            x_pn = [['b1', 't1'], ['b1', 't2'], ['b2', 't1'], ['b2', 't2'], ['t3', 'b3'], ['t3', 'b4'], ['t4', 'b3'], ['t4', 'b4']],
                            u = 2*[['cat1'], ['cat1'], ['cat2'], ['cat2']],
                            u_psb = ['cat1', 'cat2'],
                            n_rep = 12),
                      'transfer': expr.stage(
                            x_pn = [['t5', 'b5'], ['t6', 'b6'], ['t1', 'b1'], ['t2', 'b2'], ['t3', 'b3'], ['t4', 'b4']],
                            u = 3*[['cat3'], ['cat4']],
                            u_psb = ['cat3', 'cat4'],
                            n_rep = 8),
                      'test': expr.stage(
                            x_pn = [['t5', 'b6'], ['t6', 'b5'], ['t1', 'b2'], ['t2', 'b1'], ['t3', 'b4'], ['t4', 'b3']],
                            u_psb = ['cat3', 'cat4'],
                            lrn = False,
                            n_rep = 2)},
                   x_dims = {'fruits': ['alpha', 'beta', 'theta', 'phi'], 'benign_faces': ['b1', 'b2', 'b3', 'b4', 'b5', 'b6'], 'angry_faces': ['t1', 't2', 't3', 't4', 't5', 't6']})

rel_irl = expr.oat(schedule_pos = ['design'],
                    behav_score_pos = expr.behav_score(stage = 'test',
                                                      trial_pos = ['t1.b2 -> nothing', 
                                                                   't2.b1 -> nothing', 
                                                                   't3.b4 -> nothing', 
                                                                   't4.b3 -> nothing'],
                                                      trial_neg = ['t1.b2 -> nothing', 
                                                                   't2.b1 -> nothing', 
                                                                   't3.b4 -> nothing', 
                                                                   't4.b3 -> nothing'],
                                                      resp_pos = ['cat4', 'cat3', 'cat3', 'cat4'],
                                                      resp_neg = ['cat3', 'cat4', 'cat4', 'cat3'])
                  )

threat_benign_os = expr.oat(schedule_pos = ['design'],
                             behav_score_pos = expr.behav_score(stage = 'test',
                                                                 trial_pos = ['t5.b6 -> nothing', 't6.b5 -> nothing'],
                                                                 trial_neg = ['t5.b6 -> nothing', 't6.b5 -> nothing'],
                                                                 resp_pos = ['cat3', 'cat4'],
                                                                 resp_neg = ['cat4', 'cat3'])
                           )

threat_benign_brel = expr.oat(schedule_pos = ['design'],
                             behav_score_pos = expr.behav_score(stage = 'test',
                                                                 trial_pos = ['t1.b2 -> nothing', 
                                                                              't2.b1 -> nothing'],
                                                                 trial_neg = ['t1.b2 -> nothing', 
                                                                              't2.b1 -> nothing'],
                                                                 resp_pos = ['cat3', 'cat4'],
                                                                 resp_neg = ['cat4', 'cat3']))
threat_benign_trel = expr.oat(schedule_pos = ['design'],
                             behav_score_pos = expr.behav_score(stage = 'test',
                                                                 trial_pos = ['t3.b4 -> nothing', 
                                                                              't4.b3 -> nothing'],
                                                                 trial_neg = ['t3.b4 -> nothing', 
                                                                              't4.b3 -> nothing'],
                                                                 resp_pos = ['cat3', 'cat4'],
                                                                 resp_neg = ['cat4', 'cat3']))

fast = expr.experiment(
                  schedules = {'design': design},
                  oats = {'rel_irl': rel_irl, 
                          'threat_benign_os': threat_benign_os, 
                          'threat_benign_brel': threat_benign_brel, 
                          'threat_benign_trel': threat_benign_trel},
                  notes = 'Facial Affect Salience Task.  Cues b1, b2 etc. represent benign faces; cues t1, t2 etc. represent threatening faces.  In the "training" stage, b1, b2, t3 and t4 are relevant, while all other cues are irrelevant.  In the "transfer" stage, b5, b6, t5 and t6 form an embedded overshadowing design.  The OAT "rel_irl" measures attentional transfer toward relevant cues opposed to irrelevant ones.  The OAT "threat_benign_os" measures threat salience in overshadowing trials (b4, b5, t5 and t6); "threat_benign_brel" measures threat salience after the benign cues were relevant (b1, b2, t1 and t2) and "threat_benign_trel" measures threat salience after threat cues were relevant (b3, b4, t3 and t4).  Names of OAT scores in RO1 proposal: measure1 = threat_benign_os, measure2 = threat_benign_brel, and measure3 = threat_benign_trel.')

del design; rel_irl; threat_benign_os; threat_benign_brel; threat_benign_trel

# Kruschke 1996, Experiment 1 (inverse base rate effect)
# I'm missing some of the test trial types.
design = expr.schedule(resp_type = 'choice',
                  stages = {
                      'training': expr.stage(
                            freq = [3, 1, 3, 1],
                            x_pn = [['i1', 'pc1'], ['i1', 'pr1'], ['i2', 'pc2'], ['i2', 'pr2']],
                            u = [['c1'], ['r1'], ['c2'], ['r2']],
                            u_psb = ['c1', 'r1', 'c2', 'r2'],
                            n_rep = 15),
                      'test': expr.stage(
                            x_pn = [['pc1'], ['pr1'], ['pc2'], ['pr2'], ['pc1', 'pr1'], ['pc2', 'pr2']],
                            u_psb = ['c1', 'r1', 'c2', 'r2'],
                            lrn = False,
                            n_rep = 2)
                  })

pc_pr = expr.oat(schedule_pos = ['design'],
            behav_score_pos = expr.behav_score(stage = 'test',
                                          trial_pos = ['pc1.pr1 -> nothing', 'pc2.pr2 -> nothing'],
                                          trial_neg = ['pc1.pr1 -> nothing', 'pc2.pr2 -> nothing'],
                                          resp_pos = ['r1', 'r2'],
                                          resp_neg = ['c1', 'c2']))

ibre = expr.experiment(schedules = {'design': design},
                       oats = {'pc_pr': pc_pr})

del design; del pc_pr

# Le Pelley and McLaren 2003 (learned predictiveness)
# Test responses were really rating scales for both response options instead of choices.
design = expr.schedule(resp_type = 'choice',
                  stages = {
                      'training': expr.stage(
                            x_pn = [['a', 'v'], ['b', 'v'], ['a', 'w'], ['b', 'w'], ['c', 'x'], ['d', 'x'], ['c', 'y'], ['d', 'y']],
                            u = 4*[['cat1'], ['cat2']],
                            u_psb = ['cat1', 'cat2'],
                            n_rep = 14),
                      'transfer': expr.stage(
                            x_pn = [['a', 'x'], ['b', 'y'], ['c', 'v'], ['d', 'w'], ['e', 'f'], ['g', 'h'], ['i', 'j'], ['k', 'l']],
                            u = 4*[['cat3'], ['cat4']],
                            u_psb = ['cat3', 'cat4'],
                            n_rep = 4),
                      'test': expr.stage(
                            x_pn = [['a', 'c'], ['b', 'd'], ['v', 'x'], ['w', 'y'], ['e', 'h'], ['f', 'g'], ['i', 'j'], ['k', 'l']],
                            u_psb = ['cat3', 'cat4'],
                            lrn = False,
                            n_rep = 1)
                  })

rel_irl = expr.oat(schedule_pos = ['design'],
                  behav_score_pos = expr.behav_score(stage = 'test',
                                                    trial_pos = ['a.c -> nothing', 'b.d -> nothing'],
                                                    trial_neg = ['v.x -> nothing', 'w.y -> nothing'],
                                                    resp_pos = ['cat3', 'cat4'],
                                                    resp_neg = ['cat3', 'cat4']))

lrn_pred = expr.experiment(schedules = {'design': design},
                           oats = {'rel_irl': rel_irl})

del design; del rel_irl

# NIMH kids simplified learned predictiveness
# stages 0a, 0b and 0c constitute a tutorial
# cue position changes in stage 0b (it's constant in stage 0a), but now we're not representing cue position
design = expr.schedule(resp_type = 'choice',
                  stages = {
                      '0a': expr.stage(
                            x_pn = [['alpha'], ['beta']],
                            u = [['cati'], ['catii']],
                            u_psb = ['cati', 'catii'],
                            n_rep = 6),
                      '0b': expr.stage(
                            x_pn = [['alpha'], ['beta']],
                            u = [['cati'], ['catii']],
                            u_psb = ['cati', 'catii'],
                            n_rep = 4),
                      '0c': expr.stage(
                            x_pn = [['alpha', 'theta'], ['beta', 'theta'], ['alpha', 'phi'], ['beta', 'phi']],
                            u = 2*[['cati'], ['catii']],
                            u_psb = ['cati', 'catii'],
                            n_rep = 4),
                      '0_test': expr.stage(
                            x_pn = [['alpha', 'theta'], ['beta', 'theta'], ['alpha', 'phi'], ['beta', 'phi']],
                            u_psb = ['cati', 'catii'],
                            lrn = False,
                            n_rep = 1),
                      'training': expr.stage(
                            x_pn = [['a', 'v'], ['b', 'v'], ['a', 'w'], ['b', 'w']],
                            u = 2*[['cat1'], ['cat2']],
                            u_psb = ['cat1', 'cat2'],
                            n_rep = 12),
                      'transfer': expr.stage(
                            x_pn = [['a', 'v'], ['b', 'w'], ['e', 'f'], ['g', 'h']],
                            u = 2*[['cat3'], ['cat4']],
                            u_psb = ['cat3', 'cat4'],
                            n_rep = 8),
                      'test': expr.stage(
                            freq = 2*[2] + 10*[1],
                            x_pn = [['a', 'w'], ['b', 'v'], ['a', 'g'], ['h', 'v'], ['e', 'b'], ['f', 'w'], ['a', 'h'], ['g', 'v'], ['b', 'f'], ['e', 'w']],
                            u_psb = 2*['cat3', 'cat4'],
                            lrn = False,
                            n_rep = 1)
                  })

# ADD OAT LATER

kids_pred = expr.experiment(schedules = {'design': design},
                            oats = {})

del design

# NIMH kids simplified angry vs. neutral face overshadowing task
design = expr.schedule(resp_type = 'choice',
                  stages = {
                      'overshadowing': expr.stage(
                            x_pn = [['t1', 'b1'], ['t2', 'b2'], ['t3', 'b3'], ['t4', 'b4']],
                            u = [['cat_c'], ['cat_c'], ['cat_m'], ['cat_m']],
                            u_psb = ['cat_c', 'cat_m'],
                            n_rep = 8),
                      'os_test': expr.stage(
                            x_pn = [['t1', 'b3'], ['t1', 'b4'], ['t2', 'b3'], ['t2', 'b4'], ['t3', 'b1'], ['t3', 'b2'], ['t4', 'b1'], ['t4', 'b2']],
                            u_psb = ['cat_c', 'cat_m'],
                            lrn = False,
                            n_rep = 1),
                      'attenuation': expr.stage(
                            x_pn = [['c1', 'b1'], ['c2', 'b2'], ['c3', 'b3'], ['c4', 'b4']],
                            u = [['cat_s'], ['cat_r'], ['cat_s'], ['cat_r']],
                            u_psb = ['cat_s', 'cat_r'],
                            n_rep = 8),
                      'at_test': expr.stage(
                            x_pn = [['c1', 'b2'], ['c1', 'b4'], ['c2', 'b1'], ['c2', 'b3'], ['c3', 'b2'], ['c3', 'b4'], ['c4', 'b1'], ['c4', 'b3']],
                            u_psb = ['cat_s', 'cat_r'],
                            lrn = False,
                            n_rep = 1)
                  })

# ADD OAT LATER

kids_os = expr.experiment(schedules = {'design': design},
                          oats = {})

del design