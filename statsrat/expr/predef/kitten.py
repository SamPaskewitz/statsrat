import pandas as pd
from statsrat import expr
'''
Simplified category learning tasks (smaller versions of the tasks like those in 'cat' -> 'kitten').
'''

# learned predictiveness
design = expr.schedule(resp_type = 'choice',
                      stages = {'relevance': expr.stage(x_pn = [['a', 'x'], ['a', 'y'], ['b', 'x'], ['b', 'y']],
                                                        y = [['cat1'], ['cat1'], ['cat2'], ['cat2']],
                                                        n_rep = 10),
                                'transfer': expr.stage(x_pn = [['a', 'x'], ['b', 'y']],
                                                       y = [['cat3'], ['cat4']],
                                                       n_rep = 10),
                                'test': expr.stage(x_pn = [['a', 'y'], ['b', 'x']],
                                                   y_psb = ['cat3', 'cat4'],
                                                   lrn = False,
                                                   n_rep = 1)})

rel_irl_oat = expr.oat(schedule_pos = ['design'],
                      behav_score_pos = expr.behav_score(stage = 'test',
                                                        trial_pos = ['a.y -> nothing', 'b.x -> nothing'],
                                                        trial_neg = ['a.y -> nothing', 'b.x -> nothing'],
                                                        resp_pos = ['cat3', 'cat4'],
                                                        resp_neg = ['cat4', 'cat3']))

learned_predictiveness = expr.experiment(schedules = {'design': design},
                                         oats = {'rel_irl': rel_irl_oat})

del design; del rel_irl_oat

# inattention after blocking
design = expr.schedule(resp_type = 'choice',
                  stages = {'single_cue': expr.stage(x_pn = [['a'], ['b']], y = [['cat1'], ['cat2']], n_rep = 5),
                            'double_cue': expr.stage(x_pn = [['a', 'x'], ['b', 'y'], ['e', 'f'], ['g', 'h']], y = 2*[['cat1'], ['cat2']], n_rep = 5),
                            'transfer': expr.stage(x_pn = [['e', 'y'], ['g', 'x']], y = [['cat3'], ['cat4']], n_rep = 10),
                            'inattention_test': expr.stage(x_pn = [['e', 'x'], ['g', 'y']], y_psb = ['cat3', 'cat4'], lrn = False, n_rep = 1)})
        
inattention_oat = expr.oat(schedule_pos = ['design'],
                          behav_score_pos = expr.behav_score(stage = 'inattention_test',
                                                            trial_pos = ['e.x -> nothing', 'g.y -> nothing'],
                                                            trial_neg = ['e.x -> nothing', 'g.y -> nothing'],
                                                            resp_pos = ['cat3', 'cat4'],
                                                            resp_neg = ['cat4', 'cat3']))
blocking_inattention = expr.experiment(schedules = {'design': design},
                                        oats = {'inattention': inattention_oat})

del design; del inattention_oat

# inattention after backward blocking
design = expr.schedule(resp_type = 'choice',
                  stages = {'double_cue': expr.stage(x_pn = [['a', 'x'], ['b', 'y'], ['e', 'f'], ['g', 'h']], y = 2*[['cat1'], ['cat2']], n_rep = 5),
                            'single_cue': expr.stage(x_pn = [['a'], ['b']], y = [['cat1'], ['cat2']], n_rep = 5),
                            'transfer': expr.stage(x_pn = [['e', 'y'], ['g', 'x']], y = [['cat3'], ['cat4']], n_rep = 10),
                            'inattention_test': expr.stage(x_pn = [['e', 'x'], ['g', 'y']], y_psb = ['cat3', 'cat4'], lrn = False, n_rep = 1)})
        
inattention_oat = expr.oat(schedule_pos = ['design'],
                          behav_score_pos = expr.behav_score(stage = 'inattention_test',
                                                            trial_pos = ['e.x -> nothing', 'g.y -> nothing'],
                                                            trial_neg = ['e.x -> nothing', 'g.y -> nothing'],
                                                            resp_pos = ['cat3', 'cat4'],
                                                            resp_neg = ['cat4', 'cat3']))
backward_blocking_inattention = expr.experiment(schedules = {'design': design},
                                                oats = {'inattention': inattention_oat})

del design; del inattention_oat

# value effect on attention
design = expr.schedule(resp_type = 'choice',
                      stages = {'value': expr.stage(x_pn = [['a'], ['b'], ['c'], ['d']],
                                                    y = [['cat1'], ['cat1'], ['cat2'], ['cat2']],
                                                    y_value = pd.Series({'cat1': 1, 'cat2': 0.1}),
                                                    n_rep = 10),
                                 'transfer': expr.stage(x_pn = [['a', 'd'], ['b', 'c']],
                                                        y = [['cat3'], ['cat4']],
                                                        n_rep = 10),
                                 'test': expr.stage(x_pn = [['a', 'c'], ['b', 'd']],
                                                    y_psb = ['cat3', 'cat4'],
                                                    lrn = False,
                                                    n_rep = 1)})

value_oat = expr.oat(schedule_pos = ['design'],
                  behav_score_pos = expr.behav_score(stage = 'test',
                                                    trial_pos = ['a.c -> nothing', 'b.d -> nothing'],
                                                    trial_neg = ['a.c -> nothing', 'b.d -> nothing'],
                                                    resp_pos = ['cat3', 'cat4'],
                                                    resp_neg = ['cat4', 'cat3']))
value = expr.experiment(schedules = {'design': design},
                        oats = {'value': value_oat})

del design; del value_oat

# blocking (forward blocking)
design = expr.schedule(resp_type = 'choice',
                      stages = {'single_cue': expr.stage(
                                        x_pn = [['a'], ['b']],
                                        y = [['cat1'], ['cat2']],
                                        n_rep = 5),
                                'double_cue': expr.stage(
                                        x_pn = [['a', 'x'], ['b', 'y'], ['e', 'f'], ['g', 'h']],
                                        y = 2*[['cat1'], ['cat2']],
                                        n_rep = 5),
                                  'test': expr.stage(
                                        x_pn = [['e', 'y'], ['g', 'x']],
                                        y_psb = ['cat1', 'cat2'],
                                        lrn = False,
                                        n_rep = 1)})

blocking_oat = expr.oat(schedule_pos = ['design'],
                          behav_score_pos = expr.behav_score(stage = 'test',
                                                            trial_pos = ['e.y -> nothing', 'g.x -> nothing'],
                                                            trial_neg = ['e.y -> nothing', 'g.x -> nothing'],
                                                            resp_pos = ['cat1', 'cat2'],
                                                            resp_neg = ['cat2', 'cat1']))
blocking = expr.experiment(schedules = {'design': design},
                           oats = {'blocking': blocking_oat})

del design; del blocking_oat

# backward blocking
design = expr.schedule(resp_type = 'choice',
                      stages = {'double_cue': expr.stage(x_pn = [['a', 'x'], ['b', 'y'], ['e', 'f'], ['g', 'h']],
                                                        y = 2*[['cat1'], ['cat2']],
                                                        n_rep = 5),
                                  'single_cue': expr.stage(x_pn = [['a'], ['b']],
                                                           y = [['cat1'], ['cat2']],
                                                           n_rep = 5),
                                  'test': expr.stage(x_pn = [['e', 'y'], ['g', 'x']],
                                                    y_psb = ['cat1', 'cat2'],
                                                    lrn = False,
                                                    n_rep = 1)})

blocking_oat = expr.oat(schedule_pos = ['design'],
                      behav_score_pos = expr.behav_score(stage = 'test',
                                                        trial_pos = ['e.y -> nothing', 'g.x -> nothing'],
                                                        trial_neg = ['e.y -> nothing', 'g.x -> nothing'],
                                                        resp_pos = ['cat1', 'cat2'],
                                                        resp_neg = ['cat2', 'cat1']))
backward_blocking = expr.experiment(schedules = {'design': design},
                                   oats = {'blocking': blocking_oat})

del design; del blocking_oat