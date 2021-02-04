from statsrat.expr import learn
'''
Simplified category learning tasks (smaller versions of the tasks like those in 'cat' -> 'kitten').
'''

# simple learned predictiveness
design = learn.schedule(name = 'design',
                      stage_list = [
                                  learn.stage(name = 'training',
                                        x_pn = [['a', 'x'], ['a', 'y'], ['b', 'x'], ['b', 'y']],
                                        u = [['cat1'], ['cat1'], ['cat2'], ['cat2']],
                                        u_psb = ['cat1', 'cat2'],
                                        n_rep = 10),
                                  learn.stage(name = 'transfer',
                                        x_pn = [['a', 'x'], ['b', 'y']],
                                        u = [['cat3'], ['cat4']],
                                        u_psb = ['cat3', 'cat4'],
                                        n_rep = 5),
                                  learn.stage(name = 'test',
                                        x_pn = [['a', 'y'], ['b', 'x']],
                                        u_psb = ['cat3', 'cat4'],
                                        lrn = False,
                                        n_rep = 1)])

rel_irl = learn.oat(schedule_pos = ['design'],
                  behav_score_pos = learn.behav_score(stage = 'test',
                                                    trial_pos = ['a.y -> nothing', 'b.x -> nothing'],
                                                    trial_neg = ['a.y -> nothing', 'b.x -> nothing'],
                                                    resp_pos = ['cat3', 'cat4'],
                                                    resp_neg = ['cat4', 'cat3']))

lrn_pred = learn.experiment(resp_type = 'choice',
                            schedules = {'design': design},
                            oats = {'rel_irl': rel_irl})

del design; del rel_irl

# simple blocking and inattention after blocking
design = learn.schedule(name = 'design',
                  stage_list = [
                              learn.stage(name = 'single_cue',
                                    x_pn = [['a'], ['b']],
                                    u = [['cat1'], ['cat2']],
                                    u_psb = ['cat1', 'cat2'],
                                    n_rep = 20),
                              learn.stage(name = 'double_cue',
                                    x_pn = [['a', 'x'], ['b', 'y']],
                                    u = [['cat1'], ['cat2']],
                                    u_psb = ['cat1', 'cat2'],
                                    n_rep = 20),
                              learn.stage(name = 'blocking_test',
                                    x_pn = [['a', 'y'], ['b', 'x']],
                                    u_psb = ['cat1', 'cat2'],
                                    lrn = False,
                                    n_rep = 1),
                              learn.stage(name = 'transfer',
                                    x_pn = [['a', 'x'], ['b', 'y']],
                                    u = [['cat3'], ['cat4']],
                                    u_psb = ['cat3', 'cat4'],
                                    n_rep = 5),
                              learn.stage(name = 'inattention_test',
                                    x_pn = [['a', 'y'], ['b', 'x']],
                                    u_psb = ['cat3', 'cat4'],
                                    lrn = False,
                                    n_rep = 1)])

blocking = learn.oat(schedule_pos = ['design'],
                          behav_score_pos = learn.behav_score(stage = 'blocking_test',
                                                            trial_pos = ['a.y -> nothing', 'b.x -> nothing'],
                                                            trial_neg = ['a.y -> nothing', 'b.x -> nothing'],
                                                            resp_pos = ['cat1', 'cat2'],
                                                            resp_neg = ['cat2', 'cat1']))
        
inattention = learn.oat(schedule_pos = ['design'],
                          behav_score_pos = learn.behav_score(stage = 'inattention_test',
                                                            trial_pos = ['a.y -> nothing', 'b.x -> nothing'],
                                                            trial_neg = ['a.y -> nothing', 'b.x -> nothing'],
                                                            resp_pos = ['cat3', 'cat4'],
                                                            resp_neg = ['cat4', 'cat3']))
blk_inatn = learn.experiment(resp_type = 'choice',
                            schedules = {'design': design},
                            oats = {'blocking': blocking, 'inattention': inattention})

del design; del blocking; del inattention

# simple backwards blocking
design = learn.schedule(name = 'design',
                  stage_list = [
                              learn.stage(name = 'double_cue',
                                    x_pn = [['a', 'x'], ['b', 'y']],
                                    u = [['cat1'], ['cat2']],
                                    u_psb = ['cat1', 'cat2'],
                                    n_rep = 20),
                              learn.stage(name = 'single_cue',
                                    x_pn = [['a'], ['b']],
                                    u = [['cat1'], ['cat2']],
                                    u_psb = ['cat1', 'cat2'],
                                    n_rep = 20),
                              learn.stage(name = 'test',
                                    x_pn = [['a', 'y'], ['b', 'x']],
                                    u_psb = ['cat1', 'cat2'],
                                    lrn = False,
                                    n_rep = 1)])

blocking = learn.oat(schedule_pos = ['design'],
                          behav_score_pos = learn.behav_score(stage = 'test',
                                                            trial_pos = ['a.y -> nothing', 'b.x -> nothing'],
                                                            trial_neg = ['a.y -> nothing', 'b.x -> nothing'],
                                                            resp_pos = ['cat1', 'cat2'],
                                                            resp_neg = ['cat2', 'cat1']))
bkwd_blk = learn.experiment(resp_type = 'choice',
                            schedules = {'design': design},
                            oats = {'blocking': blocking})

del design; del blocking