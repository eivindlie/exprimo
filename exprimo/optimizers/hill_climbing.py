import json
from random import randint

from exprimo import DeviceGraph
from optimizers.base import BaseOptimizer
from optimizers.utils import generate_random_placement, evaluate_placement, apply_placement


class HillClimbingOptimizer(BaseOptimizer):

    def optimize(self, net_string, device_graph):
        n_devices = len(device_graph.devices)

        def generate_neighbours(placement):
            if n_devices == 1:
                return

            i = 0
            while i < len(placement):
                p = placement[i]
                if p < n_devices - 1:
                    n = placement[:]
                    n[i] = p + 1
                    yield n
                if p > 0:
                    n = placement[:]
                    n[i] = p - 1
                    yield n
                i += 1

        net = json.loads(net_string)
        groups = self.create_colocation_groups(net['layers'].keys())

        placement = generate_random_placement(len(groups), n_devices)
        score = evaluate_placement(apply_placement(net_string, placement, groups), device_graph,
                                   batches=self.batches, pipeline_batches=self.pipeline_batches)

        i = 0
        while True:
            i += 1
            if self.verbose:
                print(f'Iteration {i}. Best running time: {score:.2f}ms')

            for n in generate_neighbours(placement):
                new_score = evaluate_placement(apply_placement(net_string, n, groups), device_graph,
                                               batches=self.batches, pipeline_batches=self.pipeline_batches)
                if (new_score < score or score == -1) and new_score != -1:
                    placement = n
                    score = new_score
                    break
            else:
                break

        return placement


class RandomHillClimbingOptimizer(BaseOptimizer):

    def __init__(self, colocation_heuristic=None, patience=100):
        super().__init__(colocation_heuristic)
        self.patience = patience

    def optimize(self, net_string, device_graph):
        n_devices = len(device_graph.devices)
        net = json.loads(net_string)
        groups = self.create_colocation_groups(net['layers'].keys())

        placement = [0] * len(groups)  # generate_random_placement(len(groups), n_devices)
        score = evaluate_placement(apply_placement(net_string, placement, groups), device_graph)

        tests = 0
        while tests < self.patience:
            new_placement = placement[:]
            new_placement[randint(0, len(new_placement) - 1)] = randint(0, n_devices - 1)
            new_score = evaluate_placement(apply_placement(net_string, new_placement, groups), device_graph)

            if new_score < score:
                tests = 0
                score = new_score
                placement = new_placement
            else:
                tests += 1
        return json.dumps(apply_placement(net_string, placement, groups))
