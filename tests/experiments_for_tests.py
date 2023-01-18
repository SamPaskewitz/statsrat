import numpy as np
import pandas as pd
import xarray as xr
from statsrat import expr

# These are simple experiment objects for test purposes.

# BLOCKING (PAVLOVIAN)

# make stage objects
one_cue_stage = expr.stage(x_pn = [['cs1']],
                           x_bg = ['ctx'],
                           y = [['us']],
                           y_psb = ['us'],
                           order_fixed = False, 
                           iti = 1,
                           n_rep = 2)

two_cue_stage = expr.stage(x_pn = [['cs1', 'cs2']],
                           x_bg = ['ctx'],
                           y = [['us']],
                           y_psb = ['us'],
                           order_fixed = False, 
                           iti = 1,
                           n_rep = 2)

test_cs2_stage = expr.stage(x_pn = [['cs2']],
                            x_bg = ['ctx'],
                            y_psb = ['us'],
                            order_fixed = False,
                            iti = 1,
                            n_rep = 2)

# make behavioral score
cs2_score = expr.behav_score(stage = 'test',
                             trial_pos = ['cs2 -> nothing'],
                             resp_pos = ['us'])

# make schedule objects
blocking_schedule = expr.schedule(resp_type = 'exct', stages = {'one_cue': one_cue_stage, 'two_cue': two_cue_stage, 'test': test_cs2_stage})

two_cue_schedule = expr.schedule(resp_type = 'exct', stages = {'two_cue': two_cue_stage, 'test': test_cs2_stage})

# make experiment object
blocking = expr.experiment(schedules = {'control': two_cue_schedule, 'experimental': blocking_schedule},
                           oats = {'blocking': expr.oat(schedule_pos = ['control'],
                                                        schedule_neg = ['experimental'],
                                                        behav_score_pos = cs2_score,
                                                        behav_score_neg = cs2_score)})

# SIMPLE CATEGORY LEARNING

# make schedule object
design = expr.schedule(resp_type = 'choice',
                       stages = {'training': expr.stage(x_pn = [['a', 'x'], ['a', 'y'], ['b', 'x'], ['b', 'y']],
                                                         y = [['cat1'], ['cat1'], ['cat2'], ['cat2']],
                                                         n_rep = 2),
                                 'test': expr.stage(x_pn = [['a', 'y'], ['b', 'x']],
                                                    y_psb = ['cat1', 'cat2'],
                                                    lrn = False,
                                                    n_rep = 1)})
# make OAT
oat = expr.oat(schedule_pos = ['design'],
               behav_score_pos = expr.behav_score(stage = 'test',
                                                  trial_pos = ['a.y -> nothing', 'b.x -> nothing'],
                                                  trial_neg = ['a.y -> nothing', 'b.x -> nothing'],
                                                  resp_pos = ['cat1', 'cat2'],
                                                  resp_neg = ['cat2', 'cat1']))

# make experiment object
learned_pred = expr.experiment(schedules = {'design': design},
                               oats = {'learning': oat})