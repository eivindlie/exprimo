import json
import random
import sys
from multiprocessing import Pool
from itertools import repeat

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from exprimo.optimizers.base import BaseOptimizer
from exprimo.graph import get_flattened_layer_names
from exprimo.optimizers.utils import evaluate_placement, apply_placement, generate_random_placement


def _create_parent_selection_function(type, s=2):
    if type in ('linear', 'lin'):
        def linear(n_parents):
            return np.array([((2 - s) / n_parents) + ((2 * i * (s - 1)) / (n_parents * (n_parents - 1)))
                             for i in range(n_parents - 1, -1, -1)])

        return linear
    elif type in ('exponential', 'exp'):
        def exponential(n_parents):
            probs = np.array([1.0 - np.exp(-i) for i in range(n_parents - 1, -1, -1)])
            return probs / probs.sum()

        return exponential


def _evaluate(individual, net_string, groups, device_graph):
    return 1 / evaluate_placement(apply_placement(net_string, individual.placement, groups), device_graph)


def _calculate_binary_difference_diversity(population):
    return sum(
        int(g[0] != g[1]) for ind1 in population for ind2 in population for g in
        zip(ind1.placement, ind2.placement)) / (len(population[0].placement) * len(population) ** 2)


class GAOptimizer(BaseOptimizer):

    def __init__(self, mutation_rate=0.05, crossover_rate=0.8, crossover_type='one-point',
                 parent_selection_function='linear', parent_selection_function_s=2,
                 population_size=100, generations=100, plot_fitness_history=False,
                 evolve_mutation_rate=False, elite_size=1, print_diversity=False,
                 min_mutation_rate=0.05, max_mutation_rate=0.9, **kwargs):
        """
        Initializes the GA optimizer, setting important hyperparameters.
        :param mutation_rate: The rate at which mutation will be applied, set at the gene level.
        :param crossover_rate: The rate at which crossover will be applied.
        :param crossover_type: The type of crossover. ['uniform', 'one-point', 'n-point'] (n is any integer)
        :param parent_selection_function: The type of distribution function applied during parent selection.
                                          ['linear', 'exponential']
        """
        super().__init__(**kwargs)

        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.population_size = population_size
        self.elite_size = elite_size

        if crossover_type == 'uniform':
            self.crossover = 'uniform'
        elif crossover_type == 'one-point':
            self.crossover = 1
        elif crossover_type.endswith('-point'):
            self.crossover = int(crossover_type.split('-')[0])
        else:
            raise Exception('Invalid crossover type.')

        self.parent_selection_distribution = _create_parent_selection_function(parent_selection_function,
                                                                               parent_selection_function_s)
        self.generations = generations
        self.evolve_mutation_rate = evolve_mutation_rate
        self.plot_fitness_history = plot_fitness_history
        self.print_diversity = print_diversity

        self.max_mutation_rate = max_mutation_rate
        self.min_mutation_rate = min_mutation_rate

        if self.n_threads > 1:
            self.worker_pool = Pool(self.n_threads)

    def optimize(self, net_string, device_graph):
        n_devices = len(device_graph.devices)
        groups = self.create_colocation_groups(get_flattened_layer_names(net_string))

        def initialize(population_size):
            if self.evolve_mutation_rate:
                return [Candidate(generate_random_placement(len(groups), n_devices),
                                  min(max(random.normalvariate(self.mutation_rate, 0.1), self.min_mutation_rate),
                                      self.max_mutation_rate))
                        for _ in range(population_size)]

            return [Candidate(generate_random_placement(len(groups), n_devices)) for i in range(population_size)]

        def evaluate(individual):
            return _evaluate(individual, net_string, groups, device_graph)

        def rank(population):
            if self.n_threads > 1:
                fn_arg = zip(population, repeat(net_string), repeat(groups), repeat(device_graph))
                fitness_scores = self.worker_pool.starmap(_evaluate, fn_arg)
                fitness_db = dict(zip(population, fitness_scores))
                return sorted(population, key=lambda x: -fitness_db[x])
            return sorted(population, key=lambda x: -evaluate(x))

        def select_parents(population, points=None):
            if points is None:
                points = len(population)

            prob_dist = self.parent_selection_distribution(points)
            cum_dist = np.zeros(len(population))
            cum_sum = 0
            for i in range(len(population)):
                cum_sum += prob_dist[i]
                cum_dist[i] = cum_sum
            mating_pool = []

            i = 0
            r = random.random() * 1 / points
            while len(mating_pool) < points:
                while r <= cum_dist[i]:
                    mating_pool.append(population[i])
                    r += 1 / points
                i += 1

            return mating_pool

        def crossover(parent1, parent2):
            if random.random() > self.crossover_rate:
                return parent1, parent2

            mutation_rate1, mutation_rate2 = parent1.mutation_rate, parent2.mutation_rate
            parent1, parent2 = parent1.placement, parent2.placement

            if self.crossover == 'uniform' or self.crossover >= len(parent1) - 1:
                child1, child2 = [], []

                for g in range(len(parent1)):
                    if random.random() > 0.5:
                        child1.append(parent1[g])
                        child2.append(parent2[g])
                    else:
                        child2.append(parent1[g])
                        child1.append(parent2[g])
                children = child1, child2
            else:
                crossover_points = []
                while len(crossover_points) < self.crossover:
                    new_point = random.randint(1, len(parent1) - 1)
                    if new_point not in crossover_points:
                        crossover_points.append(new_point)

                children = ([], [])
                parent_sel = int(random.random())
                crossover_points = [0] + crossover_points + [len(parent1)]
                for i in range(len(crossover_points) - 1):
                    children[parent_sel][crossover_points[i]:crossover_points[i + 1]] \
                        = parent1[crossover_points[i]:crossover_points[i + 1]]
                    children[(parent_sel + 1) % 2][crossover_points[i]:crossover_points[i + 1]] \
                        = parent2[crossover_points[i]:crossover_points[i + 1]]
                    parent_sel = (parent_sel + 1) % 2

            if self.evolve_mutation_rate:
                mix_rate = random.normalvariate(0.5, 0.1)
                mr1 = mutation_rate1 * mix_rate + mutation_rate2 * (1 - mix_rate)
                mr2 = mutation_rate2 * mix_rate + mutation_rate1 * (1 - mix_rate)
                children = Candidate(children[0], mr1), Candidate(children[1], mr2)
            else:
                children = Candidate(children[0]), Candidate(children[1])
            return children

        def recombine(mating_pool):
            assert len(mating_pool) % 2 == 0, "Mating pool must contain an equal number of parents"
            random.shuffle(mating_pool)
            children = []
            for i in range(0, len(mating_pool), 2):
                children.extend(crossover(mating_pool[i], mating_pool[i + 1]))
            return children

        def mutate(individual):
            if self.evolve_mutation_rate:
                mutation_rate = individual.mutation_rate
                placement = individual.placement
                placement = [random.randint(0, n_devices - 1) if random.random() < mutation_rate else g
                             for g in placement]
                new_mutation_rate = max(min(mutation_rate + random.normalvariate(0, 0.05), self.max_mutation_rate),
                                        self.min_mutation_rate)
                return Candidate(placement, new_mutation_rate)
            else:
                placement = [random.randint(0, n_devices - 1) if random.random() < self.mutation_rate else g
                             for g in individual.placement]
                return Candidate(placement)

        def mutate_population(population):
            return [mutate(ind) for ind in population]

        def select_offspring(previous_generation_ranked, candidates):
            if self.elite_size:
                random.shuffle(candidates)
                return previous_generation_ranked[:self.elite_size] \
                       + candidates[:self.population_size - self.elite_size]
            return candidates

        pop = initialize(self.population_size)

        if self.plot_fitness_history:
            fitness_history = []

        if self.print_diversity:
            diversity_history = []

        for i in tqdm(range(self.generations), file=sys.stdout):
            ranked_pop = rank(pop)

            if self.plot_fitness_history:
                fitness_history.append(1 / evaluate(ranked_pop[0]))

            mating_pool = select_parents(ranked_pop)
            children = recombine(mating_pool)
            candidates = mutate_population(children)
            pop = select_offspring(ranked_pop, candidates)

            if self.verbose and (i + 1) % int(self.verbose) == 0:
                best_score = evaluate(ranked_pop[0])
                best_time = 1 / best_score
                if self.print_diversity:
                    diversity = _calculate_binary_difference_diversity(ranked_pop)
                    diversity_history.append(diversity)
                    tqdm.write(
                        f'[{i + 1}/{self.generations}] Best current time: {best_time:.2f}ms Diversity: {diversity:.4f}')
                else:
                    tqdm.write(f'[{i + 1}/{self.generations}] Best current time: {best_time:.2f}ms')

        if self.plot_fitness_history:
            plt.plot(fitness_history)
            plt.title('Fitness')
            plt.show()

        if self.print_diversity:
            plt.plot(diversity_history)
            plt.title('Diversity')
            plt.show()


        ranked_pop = rank(pop)
        best_solution = ranked_pop[0]
        return json.dumps(apply_placement(net_string, best_solution.placement, groups))


class Candidate:
    def __init__(self, placement, mutation_rate=0):
        self.placement = placement
        self.mutation_rate = mutation_rate

    def __str__(self):
        return f'Placement: {self.placement}\t Mutation rate: {self.mutation_rate}'
