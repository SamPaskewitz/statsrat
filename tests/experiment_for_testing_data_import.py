import numpy as np
import pandas as pd
import xarray as xr
from statsrat import expr

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
