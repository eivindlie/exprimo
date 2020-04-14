import json
import random
from collections import Counter
from typing import Sequence

from itertools import repeat
from multiprocessing.pool import Pool

import numpy as np
from tqdm import tqdm

from exprimo.optimizers.base import BaseOptimizer
from exprimo.optimizers.utils import evaluate_placement, apply_placement, generate_random_placement, flatten
from exprimo.graph import get_flattened_layer_names, ComputationGraph


def _evaluate(individual, net_string, groups, device_graph, dimension_sizes, pipeline_batches=1, batches=1,
              simulator_comp_penalty=1, simulator_comm_penalty=1):

    c = Counter(individual)
    device_mode = c.most_common(1)[0][0]
    device_mode = round((device_mode / len(device_graph.devices)) * dimension_sizes[0])

    used_devices = round(((len(set(individual)) - 1) / (len(device_graph.devices) - 1)) * dimension_sizes[1])

    comp_graph_dict = apply_placement(net_string, individual, groups)
    comp_graph = ComputationGraph()
    comp_graph.load_from_string(json.dumps(comp_graph_dict))

    num_jumps, max_jumps = comp_graph.get_number_of_jumps(return_max_jumps=True)
    num_jumps = round((num_jumps / max_jumps) * dimension_sizes[2])

    description = (device_mode, used_devices, num_jumps)

    score = 1 / evaluate_placement(comp_graph_dict, device_graph,
                                   pipeline_batches=pipeline_batches, batches=batches,
                                   comp_penalty=simulator_comp_penalty, comm_penalty=simulator_comm_penalty)

    return score, description, individual


class MapElitesOptimizer(BaseOptimizer):

    def __init__(self, dimension_sizes=(-1, -1, 10), initial_size=50,
                 simulator_comp_penalty=1, simulator_comm_penalty=1,
                 steps=1000, allow_cpu=True, mutation_rate=0.05, **kwargs):
        super().__init__(**kwargs)
        self.dimension_sizes = dimension_sizes
        self.initial_size = initial_size
        self.simulator_comp_penalty = simulator_comp_penalty
        self.simulator_comm_penalty = simulator_comm_penalty
        self.steps = steps
        self.allow_cpu = allow_cpu
        self.mutation_rate = mutation_rate

        if self.n_threads > 1:
            self.worker_pool = Pool(self.n_threads)
        else:
            self.worker_pool = None

    def optimize(self, net_string, device_graph):

        n_devices = len(device_graph.devices)
        groups = self.create_colocation_groups(get_flattened_layer_names(net_string))

        if self.dimension_sizes[0] == -1:
            self.dimension_sizes[0] = n_devices

        if self.dimension_sizes[1] == -1:
            self.dimension_sizes[1] = n_devices - 1

        archive = [
            [
                [
                    None for _ in range(self.dimension_sizes[2])
                ] for _ in range(self.dimension_sizes[1])
            ] for _ in range(self.dimension_sizes[0])
        ]

        def evaluate(individual):
            return _evaluate(individual, net_string, groups, device_graph, self.dimension_sizes, self.pipeline_batches,
                             self.batches, self.simulator_comp_penalty, self.simulator_comm_penalty)

        def mutate_single_gene(gene):
            if self.allow_cpu:
                return random.randint(0, n_devices - 1)
            return random.randint(1, n_devices - 1)

        def mutate(individual):
            placement = [mutate_single_gene(g) if random.random() < self.mutation_rate else g
                         for g in individual]
            return placement

        def create_candidates(n, create_random=False):
            if n == 0:
                return []
            candidates = []
            if create_random:
                while len(candidates) < n:
                    candidates.append(generate_random_placement(len(groups), n_devices, allow_device_0=self.allow_cpu))
            else:
                flattened_archive = list(flatten(archive, depth=2, include_none=False))
                while len(candidates) < n:
                    candidate = random.choice(flattened_archive)[1]
                    candidate = mutate(candidate)
                    candidates.append(candidate)

            return candidates

        for i in tqdm(range(0, self.steps, self.n_threads)):
            init_number = min(max(0, self.initial_size - i), self.n_threads)
            candidates = create_candidates(init_number, create_random=True)
            candidates += create_candidates(self.n_threads - init_number)

            if self.n_threads == 1:
                eval_results = [evaluate(candidates[0])]
            else:
                fn_args = zip(candidates, repeat(net_string), repeat(groups), repeat(device_graph),
                              repeat(self.pipeline_batches), repeat(self.batches), repeat(self.simulator_comp_penalty),
                              repeat(self.simulator_comm_penalty))
                eval_results = self.worker_pool.starmap(_evaluate, fn_args)

            for result in eval_results:
                score, description, individual = result

                previous_elite = archive[description[0]][description[1]][description[2]]
                if previous_elite is None or previous_elite[0] < score:
                    archive[description[0]][description[1]][description[2]] = (score, individual)

        return archive
