import json
import random

import numpy as np
from tqdm import tqdm

from exprimo.optimizers.base import BaseOptimizer
from exprimo.graph import get_flattened_layer_names
from exprimo.optimizers.utils import generate_random_placement, apply_placement


class Particle:
    def __init__(self, position, velocity):
        self.position = np.array(position, dtype=np.float64)
        self.best_position = position
        self.best_score = 0
        self.velocity = np.array(velocity)

    def evaluate(self, evaluation_function):
        score = evaluation_function(self.position)
        if score > self.best_score:
            self.best_score = score

    def update_velocity(self, w, l1, l2, swarm_best_position):
        u1 = np.random.uniform(size=self.position.shape[0])
        u2 = np.random.uniform(size=self.position.shape[0])
        self.velocity = w * self.velocity + l1 * u1 * (self.best_position - self.position) \
                            + l2 * u2 * (swarm_best_position - self.position)

    def update_position(self):
        self.position += self.velocity


class ParticleSwarmOptimizer(BaseOptimizer):

    def __init__(self, swarm_size=100, w=0.01, l1=0.01, l2=0.01, steps=100, **kwargs):
        super().__init__(**kwargs)
        self.swarm_size = swarm_size
        self.w = w
        self.l1 = l1
        self.l2 = l2
        self.steps = steps

    def optimize(self, net_string, device_graph):
        n_devices = len(device_graph.devices)
        groups = self.create_colocation_groups(get_flattened_layer_names(net_string))

        def initialize_swarm():
            swarm = []

            for i in range(self.swarm_size):
                position = generate_random_placement(len(groups), n_devices)
                velocity = [random.random() * n_devices * 2 - n_devices]
                particle = Particle(position, velocity)
                particle.evaluate(evaluate)
                swarm.append(particle)

            return swarm

        def find_global_best(swarm):
            global_best = max(swarm, key=lambda x: x.best_score)

            return global_best.position

        def position_to_placement(position):
            return [min(max(int(g), 0), n_devices - 1) for g in position]

        def evaluate(position):
            placement = position_to_placement(position)
            return self.evaluate_placement(apply_placement(net_string, placement, groups), device_graph)

        swarm = initialize_swarm()
        global_best_position = find_global_best(swarm)

        for i in tqdm(range(self.steps)):
            for particle in swarm:
                particle.update_velocity(self.w, self.l1, self.l2, global_best_position)
                particle.update_position()
                particle.evaluate(evaluate)

            global_best_position = find_global_best(swarm)

        return json.dumps(apply_placement(net_string, position_to_placement(global_best_position), groups))


