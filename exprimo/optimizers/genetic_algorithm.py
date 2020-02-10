import random

import numpy as np

from exprimo.optimizers.base import BaseOptimizer
from exprimo.graph import get_flattened_layer_names
from exprimo.optimizers.utils import evaluate_placement, apply_placement, generate_random_placement


def _create_parent_selection_function(type, s=1):
    if type in ('linear', 'lin'):
        def linear(rank, n_parents):
            i = n_parents - rank - 1
            return ((2 - s) / n_parents) + ((2 * i * (s - 1)) / (n_parents * (n_parents - 1)))
        return linear
    elif type in ('exponential', 'exp'):
        def exponential(rank, n_parents):
            i = n_parents - rank - 1
            return (1 - np.exp(-i)) / 1
        return exponential


class GAOptimizer(BaseOptimizer):

    def __init__(self, mutation_rate=0.05, crossover_rate=0.8, crossover_type='one-point',
                 parent_selection_function='linear', parent_selection_function_s=1,
                 population_size=100, generations=100):
        """
        Initializes the GA optimizer, setting important hyperparameters.
        :param mutation_rate: The rate at which mutation will be applied, set at the gene level.
        :param crossover_rate: The rate at which crossover will be applied.
        :param crossover_type: The type of crossover. ['uniform', 'one-point', 'n-point'] (n is any integer)
        :param parent_selection_function: The type of distribution function applied during parent selection.
                                          ['linear', 'exponential']
        """
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.population_size = population_size

        if crossover_type == 'uniform':
            self.crossover = 'uniform'
        elif crossover_type == 'one-point':
            self.crossover = 1
        elif crossover_type.endswith('-point'):
            self.crossover = int(crossover_type.split('-')[0])
        else:
            raise Exception('Invalid crossover type.')

        self.parent_selection_function = _create_parent_selection_function(parent_selection_function,
                                                                           parent_selection_function_s)
        self.generations = generations

    def optimize(self, net_string, device_graph):
        n_devices = len(device_graph.devices)
        groups = self.create_colocation_groups(get_flattened_layer_names(net_string))

        def initialize(population_size):
            return [generate_random_placement(len(groups), n_devices) for i in range(population_size)]

        def evaluate(individual):
            return 1 / evaluate_placement(apply_placement(net_string, individual, groups))

        def rank(population):
            return sorted(population, key=evaluate)

        def select_parents(population):
            return population

        def crossover(parent1, parent2):
            if random.random() > self.crossover_rate:
                return parent1, parent2

            if self.crossover == 'uniform' or crossover >= len(parent1):
                child1, child2 = [], []

                for g in range(len(parent1)):
                    if random.random() > 0.5:
                        child1.append(parent1[g])
                        child2.append(parent2[g])
                    else:
                        child2.append(parent1[g])
                        child1.append(parent2[g])
            else:
                pass

        def recombine(mating_pool):
            assert len(mating_pool) % 2 == 0, "Mating pool must contain an equal number of parents"
            mating_pool = random.shuffle(mating_pool)
            children = []
            for i in range(0, len(mating_pool), 2):
                children.extend(crossover(mating_pool[i], mating_pool[i + 1]))
            return children

        def mutate(individual):
            return [random.randint(0, n_devices - 1) if random.random() < self.mutation_rate else g
                    for g in individual]

        def mutate_population(population):
            return [mutate(ind) for ind in population]

        def select_offspring(previous_generation, candidates):
            return candidates

        pop = initialize(self.population_size)

        for i in range(self.generations):
            ranked_pop = rank(pop)
            mating_pool = select_parents(ranked_pop)
            children = recombine(mating_pool)
            candidates = mutate_population(children)
            pop = select_offspring(pop, candidates)
