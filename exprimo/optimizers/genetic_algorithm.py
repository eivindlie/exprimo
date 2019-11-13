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
                 early_stopping_threshold=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.steps = steps
        self.early_stopping_threshold = early_stopping_threshold

    def optimize(self, net_string, device_graph):
        net = json.loads(net_string)
        groups = self.create_colocation_groups(net['layers'].keys())

        n_groups = len(groups)
        n_devices = len(device_graph.devices)

        best_score = 0
        early_stopping_counter = 0

        def create_initial_population(population_size):
            population = []

            for i in range(population_size):
                ind = generate_random_placement(n_groups, n_devices)
                population.append(ind)

            return population

        def calculate_fitness(placement):
            run_time = evaluate_placement(apply_placement(net_string, placement, groups), device_graph)
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

        def create_next_generation(current_gen, elite_size, mutation_rate):
            nonlocal early_stopping_counter, best_score
            pop_rank = create_ranking(current_gen)

            if pop_rank[0][1] > best_score:
                best_score = pop_rank[0][1]
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1

            selection_results = selection(pop_rank, elite_size)
            mating_pool = get_mating_pool(current_gen, selection_results)
            children = breed_population(mating_pool, elite_size)
            next_gen = mutate_population(children, mutation_rate)

            return next_gen

        pop = create_initial_population(self.population_size)

        for g in tqdm(range(self.steps)):
            pop = create_next_generation(pop, self.elite_size, self.mutation_rate)

            if self.early_stopping_threshold and early_stopping_counter > self.early_stopping_threshold:
                print('Early stopping criterion achieved. Stopping training.')
                break

            if self.verbose and (g+1) % 50 == 0:
                ranking = create_ranking(pop)
                best_solution = pop[ranking[0][0]]
                best_score = ranking[0][1]
                print(f'[{g+1}/{self.steps}] Best time: {1 / best_score:.2f}ms \t Best solution: {best_solution}')

        ranking = create_ranking(pop)
        best_solution = pop[ranking[0][0]]
        return json.dumps(apply_placement(net_string, best_solution, groups))
