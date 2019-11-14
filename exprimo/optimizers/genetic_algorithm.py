import operator
import random
import json

import numpy as np
import pandas as pd
from tqdm import tqdm

from exprimo.optimizers.base import BaseOptimizer
from exprimo.optimizers.utils import generate_random_placement, evaluate_placement, apply_placement


class GAOptimizer(BaseOptimizer):

    def __init__(self, population_size=100, mutation_rate=0.01, elite_size=20, steps=500,
                 early_stopping_threshold=None, use_caching=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.steps = steps
        self.early_stopping_threshold = early_stopping_threshold
        self.use_caching = use_caching

    def optimize(self, net_string, device_graph):
        net = json.loads(net_string)
        groups = self.create_colocation_groups(net['layers'].keys())

        n_groups = len(groups)
        n_devices = len(device_graph.devices)

        best_score = 0
        early_stopping_counter = 0

        if self.use_caching:
            fitness_cache = {}

        def create_initial_population(population_size):
            population = []

            for i in range(population_size):
                ind = generate_random_placement(n_groups, n_devices)
                population.append(ind)

            return population

        def calculate_fitness(placement):
            if self.use_caching:
                placement_t = tuple(placement)
                if placement_t in fitness_cache:
                    return fitness_cache[placement_t]

            run_time = evaluate_placement(apply_placement(net_string, placement, groups), device_graph)

            if self.use_caching:
                fitness_cache[tuple(placement)] = 1 / run_time

            return 1 / run_time

        def create_ranking(population):
            fitness_results = {}
            for i in range(len(population)):
                fitness_results[i] = calculate_fitness(population[i])
            return sorted(fitness_results.items(), key=operator.itemgetter(1), reverse=True)

        def selection(ranking, elite_size):
            selection_results = []
            df = pd.DataFrame(np.array(ranking), columns=['index', 'fitness'])
            df['cum_sum'] = df['fitness'].cumsum() / df['fitness'].sum()

            for i in range(elite_size):
                selection_results.append(ranking[i][0])

            for i in range(len(ranking) - elite_size):
                pick = random.random()

                picked_idx = int(df[df['cum_sum'] >= pick]['index'].iloc[0])
                selection_results.append(picked_idx)

            return selection_results

        def get_mating_pool(population, selection_results):
            mating_pool = [population[idx] for idx in selection_results]
            return mating_pool

        def breed(parent1, parent2):
            split = random.randint(0, len(parent1) - 1)
            child = parent1[:split] + parent2[split:]
            return child

        def breed_population(mating_pool, elite_size):
            children = []
            breed_size = len(mating_pool) - elite_size
            pool = random.sample(mating_pool, len(mating_pool))

            for i in range(elite_size):
                children.append(mating_pool[i])

            for i in range(breed_size):
                child = breed(pool[i], pool[-i - 1])
                children.append(child)

            return children

        def mutate(individual, mutation_rate):
            for i in range(len(individual)):
                if random.random() < mutation_rate:
                    individual[i] = random.randint(0, n_devices - 1)
            return individual

        def mutate_population(population, mutation_rate):
            mutated_pop = []
            for ind in population:
                mutated = mutate(ind, mutation_rate)
                mutated_pop.append(mutated)
            return mutated_pop


        pop = create_initial_population(self.population_size)

        for g in tqdm(range(self.steps)):
            pop_rank = create_ranking(pop)

            if pop_rank[0][1] > best_score:
                best_score = pop_rank[0][1]
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1

            selection_results = selection(pop_rank, self.elite_size)
            mating_pool = get_mating_pool(pop, selection_results)
            children = breed_population(mating_pool, self.elite_size)
            next_gen = mutate_population(children, self.mutation_rate)

            pop = next_gen

            if self.early_stopping_threshold and early_stopping_counter > self.early_stopping_threshold:
                print('Early stopping criterion achieved. Stopping training.')
                break

            if self.verbose and (g+1) % 10 == 0:
                best_solution = pop[pop_rank[0][0]]
                best_score = pop_rank[0][1]
                print(f'[{g+1}/{self.steps}] Best time: {1 / best_score:.2f}ms \t Best solution: {best_solution}')

        ranking = create_ranking(pop)
        best_solution = pop[ranking[0][0]]
        return json.dumps(apply_placement(net_string, best_solution, groups))
