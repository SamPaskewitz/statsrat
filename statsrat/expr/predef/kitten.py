import pandas as pd
from statsrat import expr
'''
Simplified category learning tasks (smaller versions of the tasks like those in 'cat' -> 'kitten').
'''

# simple learned predictiveness
design = expr.schedule(name = 'design',
                      stage_list = [
                                  expr.stage(name = 'relevance',
                                        x_pn = [['a', 'x'], ['a', 'y'], ['b', 'x'], ['b', 'y']],
                                        u = [['cat1'], ['cat1'], ['cat2'], ['cat2']],
                                        n_rep = 10),
                                  expr.stage(name = 'transfer',
                                        x_pn = [['a', 'x'], ['b', 'y']],
                                        u = [['cat3'], ['cat4']],
                                        n_rep = 5),
                                  expr.stage(name = 'test',
                                        x_pn = [['a', 'y'], ['b', 'x']],
                                        u_psb = ['cat3', 'cat4'],
                                        lrn = False,
                                        n_rep = 1)])

rel_irl = expr.oat(schedule_pos = ['design'],
                  behav_score_pos = expr.behav_score(stage = 'test',
                                                    trial_pos = ['a.y -> nothing', 'b.x -> nothing'],
                                                    trial_neg = ['a.y -> nothing', 'b.x -> nothing'],
                                                    resp_pos = ['cat3', 'cat4'],
                                                    resp_neg = ['cat4', 'cat3']))

lrn_pred = expr.experiment(resp_type = 'choice',
                            schedules = {'design': design},
                            oats = {'rel_irl': rel_irl})

del design; del rel_irl

# simple blocking and inattention after blocking
design = expr.schedule(name = 'design',
                  stage_list = [
                              expr.stage(name = 'single_cue',
                                    x_pn = [['a'], ['b']],
                                    u = [['cat1'], ['cat2']],
                                    n_rep = 20),
                              expr.stage(name = 'double_cue',
                                    x_pn = [['a', 'x'], ['b', 'y'], ['e', 'f'], ['g', 'h']],
                                    u = 2*[['cat1'], ['cat2']],
                                    n_rep = 20),
                              expr.stage(name = 'blocking_test',
                                    x_pn = [['e', 'y'], ['g', 'x']],
                                    u_psb = ['cat1', 'cat2'],
                                    lrn = False,
                                    n_rep = 1),
                              expr.stage(name = 'transfer',
                                    x_pn = [['a', 'x'], ['b', 'y']],
                                    u = [['cat3'], ['cat4']],
                                    n_rep = 5),
                              expr.stage(name = 'inattention_test',
                                    x_pn = [['a', 'y'], ['b', 'x']],
                                    u_psb = ['cat3', 'cat4'],
                                    lrn = False,
                                    n_rep = 1)])

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
blk_inatn = expr.experiment(resp_type = 'choice',
                            schedules = {'design': design},
                            oats = {'blocking': blocking, 'inattention': inattention})

del design; del blocking; del inattention

# value effect on salience
design = expr.schedule(name = 'design',
                      stage_list = [
                                  expr.stage(name = 'value',
                                        x_pn = [['a'], ['b'], ['x'], ['y']],
                                        u = [['cat1'], ['cat1'], ['cat2'], ['cat2']],
                                        u_value = pd.Series({'cat1': 1, 'cat2': 0.1}),
                                        n_rep = 10),
                                  expr.stage(name = 'transfer',
                                        x_pn = [['a', 'x'], ['b', 'y']],
                                        u = [['cat3'], ['cat4']],
                                        n_rep = 5),
                                  expr.stage(name = 'test',
                                        x_pn = [['a', 'y'], ['b', 'x']],
                                        u_psb = ['cat3', 'cat4'],
                                        lrn = False,
                                        n_rep = 1)])

value = expr.oat(schedule_pos = ['design'],
                  behav_score_pos = expr.behav_score(stage = 'test',
                                                    trial_pos = ['a.y -> nothing', 'b.x -> nothing'],
                                                    trial_neg = ['a.y -> nothing', 'b.x -> nothing'],
                                                    resp_pos = ['cat3', 'cat4'],
                                                    resp_neg = ['cat4', 'cat3']))
value_sal = expr.experiment(resp_type = 'choice',
                             schedules = {'design': design},
                             oats = {'value': value})

del design; del value

# simple backwards blocking
design = expr.schedule(name = 'design',
                      stage_list = [
                                  expr.stage(name = 'double_cue',
                                        x_pn = [['a', 'x'], ['b', 'y'], ['e', 'f'], ['g', 'h']],
                                        u = 2*[['cat1'], ['cat2']],
                                        n_rep = 20),
                                  expr.stage(name = 'single_cue',
                                        x_pn = [['a'], ['b']],
                                        u = [['cat1'], ['cat2']],
                                        n_rep = 20),
                                  expr.stage(name = 'test',
                                        x_pn = [['e', 'y'], ['g', 'x']],
                                        u_psb = ['cat1', 'cat2'],
                                        lrn = False,
                                        n_rep = 1)])

blocking = expr.oat(schedule_pos = ['design'],
                          behav_score_pos = expr.behav_score(stage = 'test',
                                                            trial_pos = ['e.y -> nothing', 'g.x -> nothing'],
                                                            trial_neg = ['e.y -> nothing', 'g.x -> nothing'],
                                                            resp_pos = ['cat1', 'cat2'],
                                                            resp_neg = ['cat2', 'cat1']))
bkwd_blk = expr.experiment(resp_type = 'choice',
                            schedules = {'design': design},
                            oats = {'blocking': blocking})

del design; del blocking