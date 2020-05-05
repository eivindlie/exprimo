import json
import random
import os
from collections import Counter
from glob import glob

from itertools import repeat
from multiprocessing.pool import Pool

from exprimo import get_log_dir
from exprimo.optimizers.base import BaseOptimizer
from exprimo.optimizers.utils import evaluate_placement, apply_placement, generate_random_placement, flatten, \
    get_device_assignment
from exprimo.graph import get_flattened_layer_names, ComputationGraph

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from exprimo import PLOT_STYLE, log
import seaborn as sns

from exprimo.plotting import plot_map_elites_archive, plot_archive_animation

sns.set(style=PLOT_STYLE)


def _evaluate(individual, net_string, groups, device_graph, pipeline_batches=1, batches=1,
              simulator_comp_penalty=1, simulator_comm_penalty=1):
    description, individual = individual

    comp_graph_dict = apply_placement(net_string, individual, groups)

    score = 1 / evaluate_placement(comp_graph_dict, device_graph,
                                   pipeline_batches=pipeline_batches, batches=batches,
                                   comp_penalty=simulator_comp_penalty, comm_penalty=simulator_comm_penalty)

    return score, description, individual


class MapElitesOptimizer(BaseOptimizer):

    def __init__(self, dimension_sizes=(-1, -1, 10), initial_size=50,
                 simulator_comp_penalty=1, simulator_comm_penalty=1,
                 steps=1000, allow_cpu=True, mutation_rate=0.05, copy_mutation_rate=0, replace_mutation_rate=0,
                 zone_mutation_rate=0, zone_fail_rate=0.2, crossover_rate=0.4,
                 benchmarking_function=None, benchmarking_steps=0, benchmark_before_selection=False,
                 benchmarking_n_keep=None, benchmarking_time_threshold=None, include_trivial_solutions=True,
                 show_score_plot=False, plot_axes=(0, 2), plot_animation=False, animation_fps=1,
                 archive_log_period=None, **kwargs):
        super().__init__(**kwargs)
        self.dimension_sizes = dimension_sizes
        self.initial_size = initial_size
        self.simulator_comp_penalty = simulator_comp_penalty
        self.simulator_comm_penalty = simulator_comm_penalty
        self.steps = steps
        self.allow_cpu = allow_cpu
        self.mutation_rate = mutation_rate
        self.copy_mutation_rate = copy_mutation_rate
        self.replace_mutation_rate = replace_mutation_rate
        self.zone_mutation_rate = zone_mutation_rate
        self.zone_fail_rate = zone_fail_rate
        self.crossover_rate = crossover_rate
        self.include_trivial_solutions = include_trivial_solutions
        self.benchmarking_steps = benchmarking_steps
        self.benchmark_before_selection = benchmark_before_selection
        self.benchmarking_n_keep = benchmarking_n_keep
        self.benchmarking_time_threshold = benchmarking_time_threshold
        self.benchmarking_function = benchmarking_function
        self.plot_axes = plot_axes
        self.show_score_plot = show_score_plot
        self.plot_animation = plot_animation
        self.animation_fps = animation_fps
        self.archive_log_period = archive_log_period

        if self.archive_log_period is not None:
            if not os.path.exists(os.path.join(get_log_dir(), 'archive_logs')):
                os.makedirs(os.path.join(get_log_dir(), 'archive_logs'))

        self.worker_pool = None

    def optimize(self, net_string, device_graph, return_full_archive=False):

        if self.n_threads > 1:
            self.worker_pool = Pool(self.n_threads)

        n_devices = len(device_graph.devices)
        groups = self.create_colocation_groups(get_flattened_layer_names(net_string))

        if self.dimension_sizes[0] == -1:
            self.dimension_sizes[0] = n_devices

        if self.dimension_sizes[1] == -1:
            self.dimension_sizes[1] = n_devices

        archive_scores = np.empty(self.dimension_sizes)
        archive_scores[:] = np.NaN
        archive_individuals = np.zeros(list(self.dimension_sizes) + [len(groups)], dtype=int)

        def evaluate(individual):
            return _evaluate(individual, net_string, groups, device_graph, self.dimension_sizes, self.pipeline_batches,
                             self.batches, self.simulator_comp_penalty, self.simulator_comm_penalty)

        def mutate(individual):
            new_individual = []
            if random.random() < self.replace_mutation_rate:
                devices_present = list(set(individual))
                i1 = random.choice(devices_present)
                i2 = random.choice(devices_present)

                new_individual = [i2 if i == i1 else i for i in individual]
            elif random.random() < self.zone_mutation_rate:
                split1 = random.randint(0, len(individual) - 1)
                split2 = split1 + min(np.random.geometric(0.2), len(individual) - split1)
                dev = random.randint(0 if self.allow_cpu else 1, n_devices - 1)
                new_individual = individual[:split1] + [dev] * (split2 - split1) + individual[split2:]
            else:
                for i, gene in enumerate(individual):
                    if random.random() < self.copy_mutation_rate and i > 0:
                        new_individual.append(individual[i - 1])
                    elif random.random() < self.mutation_rate:
                        if self.allow_cpu:
                            new_individual.append(random.randint(0, n_devices - 1))
                        else:
                            new_individual.append(random.randint(1, n_devices - 1))
                    else:
                        new_individual.append(gene)

            return new_individual

        def crossover(parent1, parent2):
            crossover_point = random.randint(1, len(parent1) - 1)
            return parent1[:crossover_point] + parent2[crossover_point:]

        def create_candidates(n, create_random=False, create_trivial=False, selectable_candidates=None):
            if n <= 0:
                return []
            candidates = []
            if create_trivial:
                candidates.extend([
                    [i] * len(groups) for i in range(1, n_devices)
                ])
                n -= n_devices - 1

                if self.allow_cpu:
                    candidates.append([0] * len(groups))
                    n -= 1

            if create_random:
                while len(candidates) < n:
                    candidates.append(generate_random_placement(len(groups), n_devices, allow_device_0=self.allow_cpu))
            else:
                selectable_indices = np.argwhere(np.isfinite(archive_scores))
                while len(candidates) < n:
                    c = []
                    if selectable_candidates:
                        for _ in range(1 + int(random.random() < self.crossover_rate)):
                            c.append(random.choice(selectable_candidates))
                    else:
                        for _ in range(1 + int(random.random() < self.crossover_rate)):
                            idx = random.choice(selectable_indices)
                            c.append(archive_individuals[idx[0], idx[1], idx[2], :].tolist())

                    if len(c) == 2:
                        candidate = crossover(*c)
                    else:
                        candidate = c[0]
                    candidate = mutate(candidate)
                    candidates.append(candidate)

            return candidates

        def create_description(individual):
            c = Counter(individual)
            device_mode = c.most_common(1)[0][0]
            device_mode = round((device_mode / len(device_graph.devices)) * self.dimension_sizes[0])

            used_devices = round(((len(set(individual)) - 1) / (len(device_graph.devices))) * self.dimension_sizes[1])

            comp_graph_dict = apply_placement(net_string, individual, groups)
            comp_graph = ComputationGraph()
            comp_graph.load_from_string(json.dumps(comp_graph_dict))

            num_jumps, max_jumps = comp_graph.get_number_of_jumps(return_max_jumps=True)
            num_jumps = round((num_jumps / max_jumps) * (self.dimension_sizes[2] - 1))

            return (device_mode, used_devices, num_jumps)

        def benchmark(individual, benchmarking_function):
            device_assignment = get_device_assignment(apply_placement(net_string, individual, groups))
            time, memory_overflow = benchmarking_function(device_assignment, return_memory_overflow=True)

            description = create_description(individual)

            # Time is set to -1 if memory overflows - but we check with memory_overflow instead
            time = max(time, 0)

            if memory_overflow == -1:
                memory_overflow = 1

            if memory_overflow > 0:
                time += memory_overflow * 10 ** 9 * 1

            return 1 / time, description, individual

        def reevaluate_archive(benchmarking_function=None, n_keep=None, time_threshold=None):
            indices = list(np.argwhere(np.isfinite(archive_scores)))

            if time_threshold:
                indices = [i for i in indices if archive_scores[i[0], i[1], i[2]] >= 1 / time_threshold]

            if n_keep:
                indices = sorted(indices, key=lambda i: -archive_scores[i[0], i[1], i[2]])
                indices = indices[:n_keep]

            assert len(indices), 'No solutions fulfill the specified requirements'

            archive_scores[:] = np.NaN
            if self.verbose:
                if n_keep:
                    log(f'Reevaluating {n_keep} best individuals in archive (and throwing away the rest)')
                else:
                    log('Reevaluating all individuals in archive')
                if time_threshold:
                    log(f'Time threshold: {time_threshold}ms')
            for i in tqdm(indices, disable=not self.verbose):
                individual = archive_individuals[i[0], i[1], i[2], :].tolist()
                if benchmarking_function:
                    archive_scores[i[0], i[1], i[2]] = benchmark(individual, benchmarking_function)[0]
                else:
                    archive_scores[i[0], i[1], i[2]] = evaluate(individual)[0]

        def log_archive(file_name):
            indices = list(np.argwhere(np.isfinite(archive_scores)))
            indices = sorted(indices, key=lambda i: -archive_scores[i[0], i[1], i[2]])

            with open(os.path.join(get_log_dir(), 'archive_logs', file_name), 'w') as f:
                f.write('niche; time; placement\n')
                for i in indices:
                    niche = tuple(i)
                    time = 1 / archive_scores[i[0], i[1], i[2]]
                    placement = archive_individuals[i[0], i[1], i[2]].tolist()

                    f.write(f'{niche}; {time}; {placement}\n')

        def run_optimization(steps, benchmarking_function=None, start_step=0):
            nonlocal archive_individuals, archive_scores

            if self.verbose:
                if benchmarking_function:
                    log('Optimizing with benchmarking...')
                else:
                    log('Optimizing with simulator...')

            step_size = 1 if benchmarking_function else self.n_threads

            for i in tqdm(range(0, steps, step_size), disable=not self.verbose):
                init_number = min(max(0, self.initial_size - i), self.n_threads)

                if self.include_trivial_solutions and i == 0:
                    candidates = create_candidates(init_number, create_trivial=True, create_random=True)
                else:
                    candidates = create_candidates(init_number, create_random=True)
                if init_number > 0:
                    candidates += create_candidates(self.n_threads - init_number, selectable_candidates=candidates[:])
                else:
                    candidates += create_candidates(self.n_threads - init_number)

                if benchmarking_function:
                    eval_results = [benchmark(candidates[0], benchmarking_function)]
                elif self.n_threads == 1:
                    eval_results = [evaluate(candidates[0])]
                else:
                    fn_args = zip(((create_description(c), c) for c in candidates), repeat(net_string), repeat(groups),
                                  repeat(device_graph), repeat(self.pipeline_batches), repeat(self.batches),
                                  repeat(self.simulator_comp_penalty), repeat(self.simulator_comm_penalty))

                    eval_results = self.worker_pool.starmap(_evaluate, fn_args)

                for result in eval_results:
                    score, description, individual = result

                    previous_elite_score = archive_scores[description[0], description[1], description[2]]
                    if np.isnan(previous_elite_score) or previous_elite_score < score:
                        archive_scores[description[0], description[1], description[2]] = score
                        archive_individuals[description[0], description[1], description[2], :] = individual

                if self.verbose and (i + 1) % self.verbose < step_size:
                    best_time = 1 / np.nanmax(archive_scores)
                    log(f'[{i + 1}/{steps}] Best time: {best_time:.4f}ms')

                    with open(os.path.join(get_log_dir(), 'time_history.csv'), 'a') as f:
                        f.write(f'{i + 1}, {best_time}\n')

                if self.archive_log_period and (i + 1) % self.archive_log_period < step_size:
                    log_archive(f'step_{i + start_step + 1:06}.csv')

        if self.verbose:
            with open(os.path.join(get_log_dir(), 'time_history.csv'), 'w') as f:
                f.write('step, time\n')


        run_optimization(self.steps)

        if self.archive_log_period is not None:
            log_archive('1_simulation_finished.csv')

        if self.benchmarking_steps > 0 or self.benchmark_before_selection:
            reevaluate_archive(self.benchmarking_function, n_keep=self.benchmarking_n_keep,
                               time_threshold=self.benchmarking_time_threshold)

            if self.archive_log_period is not None:
                log_archive('2_reevaluated.csv')

        if self.benchmarking_steps > 0:
            run_optimization(self.benchmarking_steps, self.benchmarking_function, self.steps)
            log_archive('3_benchmarking_finished.csv')

        if self.show_score_plot:
            if self.verbose:
                log('Plotting archive scores...', end='')
            graph = ComputationGraph()
            graph.load_from_string(net_string)
            _, max_jumps = graph.get_number_of_jumps(return_max_jumps=True)
            plot_map_elites_archive(archive_scores, n_devices, max_jumps, self.plot_axes,
                                    save_path=os.path.join(get_log_dir(), 'archive_plot.pdf'))
            if self.verbose:
                log('Done')

        if self.plot_animation:
            if not self.archive_log_period and self.verbose:
                log('self.plot_animation was set to True, but archive logging was not enabled. '
                    'Skipping animation plot.')
            else:
                if self.verbose:
                    log('Plotting archive animation...', end='')
                paths = glob(os.path.join(get_log_dir(), 'archive_logs', 'step_*.csv'))
                plot_archive_animation(paths, (os.path.join(get_log_dir(), 'archive_animation.mp4'),
                                               os.path.join(get_log_dir(), 'archive_animation.gif')),
                                       self.dimension_sizes,
                                       n_devices=n_devices, max_jumps=max_jumps, axes=self.plot_axes,
                                       fps=self.animation_fps)
                if self.verbose:
                    log('Done')

        if return_full_archive:
            return archive_scores, archive_individuals

        best_index = np.nanargmax(archive_scores)
        best_individual = archive_individuals.reshape((-1, len(groups)))[best_index]

        if self.verbose:
            log(f'Best individual: {best_individual.tolist()}')

        if self.worker_pool:
            self.worker_pool.close()

        return json.dumps(apply_placement(net_string, best_individual.tolist(), groups))
