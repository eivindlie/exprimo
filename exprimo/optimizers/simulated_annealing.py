import json
import os
from random import randint, random

import numpy as np
import scipy
from scipy.special import expit
from tqdm import tqdm

from exprimo import log, get_log_dir
from exprimo.optimizers.base import BaseOptimizer
from exprimo.optimizers.utils import apply_placement
from exprimo.graph import get_flattened_layer_names


def exponential_multiplicative_decay(initial_value, decay):
    return lambda t: initial_value * decay ** t


temp_schedules = {
    'exponential_multiplicative_decay': exponential_multiplicative_decay
}


class SimulatedAnnealingOptimizer(BaseOptimizer):

    def __init__(self, *args, temp_schedule=exponential_multiplicative_decay(40, 0.95), steps=500,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.temp_schedule = temp_schedule
        self.steps = steps

    def optimize(self, net_string, device_graph):
        n_devices = len(device_graph.devices)

        groups = self.create_colocation_groups(get_flattened_layer_names(net_string))

        placement = [randint(0, n_devices - 1) for n in range(len(groups))]  # [0] * len(groups)
        score = self.evaluate_placement(apply_placement(net_string, placement, groups), device_graph)

        if self.score_save_period:
            with open(os.path.join(get_log_dir(), 'time_history.csv'), 'w') as f:
                f.write('step, time\n')

        for i in tqdm(range(self.steps), disable=not self.verbose):
            new_placement = placement[:]
            new_placement[randint(0, len(new_placement) - 1)] = randint(0, n_devices - 1)
            new_score = self.evaluate_placement(apply_placement(net_string, new_placement, groups), device_graph)

            if self.verbose and (i + 1) % self.verbose == 0:
                log(f'[{i + 1}/{self.steps}] Best run time: {score:,.2f}ms')

            if self.score_save_period and i % self.score_save_period == 0:
                with open(os.path.join(get_log_dir(), 'time_history.csv'), 'a') as f:
                    f.write(f'{i + 1}, {score}\n')

            if new_score != -1:
                if new_score < score or score == -1 \
                        or random() < expit((score - new_score) / self.temp(i)):
                    score = new_score
                    placement = new_placement

        return json.dumps(apply_placement(net_string, placement, groups))

    def temp(self, i):
        if callable(self.temp_schedule):
            return self.temp_schedule(i)
        return self.temp_schedule


class ScipySimulatedAnnealingOptimizer(BaseOptimizer):

    def __init__(self, *args, steps=500, **kwargs):
        super().__init__(*args, **kwargs)
        self.steps = steps

    def optimize(self, net_string, device_graph):
        n_devices = len(device_graph.devices)

        groups = self.create_colocation_groups(get_flattened_layer_names(net_string))

        def eval_function(x):
            new_placement = [int(round(g)) for g in x]
            score = self.evaluate_placement(apply_placement(net_string, new_placement, groups), device_graph)
            return score

        result = scipy.optimize.dual_annealing(eval_function, [(0, n_devices - 1)] * len(groups),
                                               no_local_search=True,
                                               maxfun=self.steps)

        placement = [int(round(g)) for g in result.x]

        if self.verbose:
            log(f'Best found placement: {placement}')

        return json.dumps(apply_placement(net_string, placement, groups))
