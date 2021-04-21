import pandas as pd
from statsrat import expr
'''
Simplified category learning tasks (smaller versions of the tasks like those in 'cat' -> 'kitten').
'''

# simple learned predictiveness
design = expr.schedule(resp_type = 'choice',
                      stages = {'relevance': expr.stage(x_pn = [['a', 'x'], ['a', 'y'], ['b', 'x'], ['b', 'y']],
                                                        u = [['cat1'], ['cat1'], ['cat2'], ['cat2']],
                                                        n_rep = 10),
                                'transfer': expr.stage(x_pn = [['a', 'x'], ['b', 'y']],
                                                       u = [['cat3'], ['cat4']],
                                                       n_rep = 5),
                                'test': expr.stage(x_pn = [['a', 'y'], ['b', 'x']],
                                                   u_psb = ['cat3', 'cat4'],
                                                   lrn = False,
                                                   n_rep = 1)})

rel_irl = expr.oat(schedule_pos = ['design'],
                  behav_score_pos = expr.behav_score(stage = 'test',
                                                    trial_pos = ['a.y -> nothing', 'b.x -> nothing'],
                                                    trial_neg = ['a.y -> nothing', 'b.x -> nothing'],
                                                    resp_pos = ['cat3', 'cat4'],
                                                    resp_neg = ['cat4', 'cat3']))

lrn_pred = expr.experiment(schedules = {'design': design},
                           oats = {'rel_irl': rel_irl})

del design; del rel_irl

# simple blocking and inattention after blocking
design = expr.schedule(resp_type = 'choice',
                  stages = {'single_cue': expr.stage(x_pn = [['a'], ['b']], u = [['cat1'], ['cat2']], n_rep = 20),
                            'double_cue': expr.stage(x_pn = [['a', 'x'], ['b', 'y'], ['e', 'f'], ['g', 'h']], u = 2*[['cat1'], ['cat2']], n_rep = 20),
                            'blocking_test': expr.stage(x_pn = [['e', 'y'], ['g', 'x']], u_psb = ['cat1', 'cat2'], lrn = False, n_rep = 1),
                            'transfer': expr.stage(x_pn = [['a', 'x'], ['b', 'y']], u = [['cat3'], ['cat4']], n_rep = 5),
                            'inattention_test': expr.stage(x_pn = [['a', 'y'], ['b', 'x']], u_psb = ['cat3', 'cat4'], lrn = False, n_rep = 1)})

blocking = expr.oat(schedule_pos = ['design'],
                          behav_score_pos = expr.behav_score(stage = 'blocking_test',
                                                            trial_pos = ['e.y -> nothing', 'g.x -> nothing'],
                                                            trial_neg = ['e.y -> nothing', 'g.x -> nothing'],
                                                            resp_pos = ['cat1', 'cat2'],
                                                            resp_neg = ['cat2', 'cat1']))
        
inattention = expr.oat(schedule_pos = ['design'],
                          behav_score_pos = expr.behav_score(stage = 'inattention_test',
                                                            trial_pos = ['a.y -> nothing', 'b.x -> nothing'],
                                                            trial_neg = ['a.y -> nothing', 'b.x -> nothing'],
                                                            resp_pos = ['cat3', 'cat4'],
                                                            resp_neg = ['cat4', 'cat3']))
blk_inatn = expr.experiment(schedules = {'design': design},
                            oats = {'blocking': blocking, 'inattention': inattention})

del design; del blocking; del inattention

# value effect on salience
design = expr.schedule(resp_type = 'choice',
                      stages = {'value': expr.stage(x_pn = [['a'], ['b'], ['x'], ['y']],
                                        u = [['cat1'], ['cat1'], ['cat2'], ['cat2']],
                                        u_value = pd.Series({'cat1': 1, 'cat2': 0.1}),
                                        n_rep = 10),
                                 'transfer': expr.stage(x_pn = [['a', 'x'], ['b', 'y']],
                                        u = [['cat3'], ['cat4']],
                                        n_rep = 5),
                                 'test': expr.stage(x_pn = [['a', 'y'], ['b', 'x']],
                                        u_psb = ['cat3', 'cat4'],
                                        lrn = False,
                                        n_rep = 1)})

value = expr.oat(schedule_pos = ['design'],
                  behav_score_pos = expr.behav_score(stage = 'test',
                                                    trial_pos = ['a.y -> nothing', 'b.x -> nothing'],
                                                    trial_neg = ['a.y -> nothing', 'b.x -> nothing'],
                                                    resp_pos = ['cat3', 'cat4'],
                                                    resp_neg = ['cat4', 'cat3']))
value_sal = expr.experiment(schedules = {'design': design},
                            oats = {'value': value})

del design; del value

# simple backwards blocking
design = expr.schedule(resp_type = 'choice',
                      stages = {'double_cue': expr.stage(
                                        x_pn = [['a', 'x'], ['b', 'y'], ['e', 'f'], ['g', 'h']],
                                        u = 2*[['cat1'], ['cat2']],
                                        n_rep = 20),
                                  'single_cue': expr.stage(
                                        x_pn = [['a'], ['b']],
                                        u = [['cat1'], ['cat2']],
                                        n_rep = 20),
                                  'test': expr.stage(
                                        x_pn = [['e', 'y'], ['g', 'x']],
                                        u_psb = ['cat1', 'cat2'],
                                        lrn = False,
                                        n_rep = 1)})

blocking = expr.oat(schedule_pos = ['design'],
                          behav_score_pos = expr.behav_score(stage = 'test',
                                                            trial_pos = ['e.y -> nothing', 'g.x -> nothing'],
                                                            trial_neg = ['e.y -> nothing', 'g.x -> nothing'],
                                                            resp_pos = ['cat1', 'cat2'],
                                                            resp_neg = ['cat2', 'cat1']))
bkwd_blk = expr.experiment(schedules = {'design': design},
                           oats = {'blocking': blocking})

del design; del blocking