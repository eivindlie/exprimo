import json
from random import randint

from tqdm import tqdm

from exprimo import DeviceGraph
from exprimo.optimizers.base import BaseOptimizer
from exprimo.optimizers.utils import generate_random_placement, evaluate_placement, apply_placement
from exprimo.graph import get_flattened_layer_names


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
        groups = self.create_colocation_groups(get_flattened_layer_names(net_string))

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

    def __init__(self, *args, steps=5000, **kwargs):
        super().__init__(*args, **kwargs)
        self.steps = steps

    def optimize(self, net_string, device_graph):
        n_devices = len(device_graph.devices)
        groups = self.create_colocation_groups(get_flattened_layer_names(net_string))

        placement = [0] * len(groups)  # generate_random_placement(len(groups), n_devices)
        score = evaluate_placement(apply_placement(net_string, placement, groups), device_graph)

        for i in tqdm(range(self.steps)):
            new_placement = placement[:]
            new_placement[randint(0, len(new_placement) - 1)] = randint(0, n_devices - 1)
            new_score = evaluate_placement(apply_placement(net_string, new_placement, groups), device_graph)

            if (new_score < score or score == -1) and new_score != -1:
                score = new_score
                placement = new_placement

            if self.verbose and (i + 1) % 50 == 0:
                print(f'[{i+1}/{self.steps}] Current time: {score:.2f}ms \t Current solution: {placement}')

        return json.dumps(apply_placement(net_string, placement, groups))
