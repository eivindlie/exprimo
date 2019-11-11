import json
from random import randint
import numpy as np

from exprimo import DeviceGraph
from exprimo.optimizers.base import BaseOptimizer
from exprimo.optimizers.utils import evaluate_placement, apply_placement


def exponential_multiplicate_decay(initial_value, decay):
    return lambda t: initial_value * decay**t


class SimulatedAnnealingOptimizer(BaseOptimizer):

    def __init__(self, colocation_heuristic=None, temp_schedule=exponential_multiplicate_decay(10, 0.95), patience=25):
        super().__init__(colocation_heuristic)
        self.temp_schedule = temp_schedule
        self.patience = patience

    def optimize(self, net_string, device_graph):
        n_devices = len(device_graph.devices)
        net = json.loads(net_string)
        groups = self.create_colocation_groups(net['layers'].keys())

        placement = [0] * len(groups)
        score = evaluate_placement(apply_placement(net_string, placement, groups), device_graph)

        i = 0
        tests = 0
        while tests < self.patience:
            new_placement = placement[:]
            new_placement[randint(0, len(new_placement) - 1)] = randint(0, n_devices - 1)
            new_score = evaluate_placement(apply_placement(net_string, new_placement, groups), device_graph)

            if new_score != -1:
                if new_score < score or score == -1\
                        or randint(0, 1) < 1 - np.exp((new_score - score)/self.temp(i)):
                    score = new_score
                    placement = new_placement
                    tests = 0
            i += 1
            tests += 1

        return apply_placement(net_string, placement, groups)

    def temp(self, i):
        if callable(self.temp_schedule):
            return self.temp_schedule(i)
        return self.temp_schedule


if __name__ == '__main__':
    optimizer = SimulatedAnnealingOptimizer()
    device_graph = DeviceGraph.load_from_file('../device_graphs/cluster2-reduced-memory.json')
    with open('../nets/mnist.json') as f:
        net_string = f.read()

    best_net = optimizer.optimize(net_string, device_graph)
    print(f'Best discovered configuration: {best_net}')
