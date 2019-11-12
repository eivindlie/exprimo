import json
from random import randint
import numpy as np
from tqdm import tqdm

from exprimo import DeviceGraph
from exprimo.optimizers.base import BaseOptimizer
from exprimo.optimizers.utils import evaluate_placement, apply_placement


def exponential_multiplicative_decay(initial_value, decay):
    return lambda t: initial_value * decay**t


class SimulatedAnnealingOptimizer(BaseOptimizer):

    def __init__(self, *args, temp_schedule=exponential_multiplicative_decay(40, 0.95), steps=500,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.temp_schedule = temp_schedule
        self.steps = steps

    def optimize(self, net_string, device_graph):
        n_devices = len(device_graph.devices)
        net = json.loads(net_string)

        groups = self.create_colocation_groups(net['layers'].keys())

        placement = [randint(0, n_devices - 1) for n in range(len(groups))]  # [0] * len(groups)
        score = evaluate_placement(apply_placement(net_string, placement, groups), device_graph,
                                   batches=self.batches, pipeline_batches=self.pipeline_batches)

        for i in tqdm(range(self.steps)):
            new_placement = placement[:]
            new_placement[randint(0, len(new_placement) - 1)] = randint(0, n_devices - 1)
            new_score = evaluate_placement(apply_placement(net_string, new_placement, groups), device_graph)

            if self.verbose and (i+1) % 50 == 0:
                print(f'[{i+1}/{self.steps}] Best run time: {score:,.2f}ms \tBest placement: {placement}')

            if new_score != -1:
                if new_score < score or score == -1\
                        or randint(0, 1) < 1 - np.exp((new_score - score)/self.temp(i)):
                    score = new_score
                    placement = new_placement

        return json.dumps(apply_placement(net_string, placement, groups))

    def temp(self, i):
        if callable(self.temp_schedule):
            return self.temp_schedule(i)
        return self.temp_schedule
