import json

import itertools
from tqdm import tqdm

from exprimo import DeviceGraph, Simulator, ComputationGraph
from exprimo.optimizers import HillClimbingOptimizer, LinearSearchOptimizer, SimulatedAnnealingOptimizer, GAOptimizer, \
    MapElitesOptimizer, RandomHillClimbingOptimizer
from exprimo.optimizers.particle_swarm_optimizer import ParticleSwarmOptimizer
from exprimo.optimizers.simulated_annealing import temp_schedules


def do_parameter_search(config_path, parameter_grid, repeats=10, verbose=False):
    with open(config_path) as f:
        config = json.load(f)

    device_graph_path = config['device_graph_path']
    net_path = config['net_path']
    device_graph = DeviceGraph.load_from_file(device_graph_path)
    with open(net_path) as f:
        net_string = f.read()

    args_blueprint = config.get('optimizer_args', {})
    batches = args_blueprint.get('batches', 1)
    pipeline_batches = args_blueprint.get('pipeline_batches', 1)
    args_blueprint['batches'] = batches
    args_blueprint['pipeline_batches'] = pipeline_batches
    args_blueprint['verbose'] = False

    comp_penalty = args_blueprint.get('simulator_comp_penalty', 1.0)
    comm_penalty = args_blueprint.get('simulator_comm_penalty', 1.0)

    if 'benchmarking_generations' in args_blueprint:
        args_blueprint['benchmarking_generations'] = 0
    if 'benchmarking_steps' in args_blueprint:
        args_blueprint['benchmarking_steps'] = 0
    if 'benchmark_before_selection' in args_blueprint:
        args_blueprint['benchmark_before_selection'] = False

    optimizers = {
        'random_hill_climber': RandomHillClimbingOptimizer,
        'hill_climber': HillClimbingOptimizer,
        'linear_search': LinearSearchOptimizer,
        'simulated_annealing': SimulatedAnnealingOptimizer,
        'sa': SimulatedAnnealingOptimizer,
        'genetic_algorithm': GAOptimizer,
        'ga': GAOptimizer,
        'pso': ParticleSwarmOptimizer,
        'particle_swarm': ParticleSwarmOptimizer,
        'map_elites': MapElitesOptimizer,
        'map-elites': MapElitesOptimizer
    }

    grid_rows = []
    for key, value in parameter_grid.items():
        row = []
        for v in value:
            row.append((key, v))
        grid_rows.append(row)

    best_time = None
    best_params = None
    for i, combination in tqdm(enumerate(itertools.product(*grid_rows))):
        if verbose:
            print(f'Testing combination {i}: {combination}...')

        args = args_blueprint.copy()
        args = dict(tuple(args.items()) + combination)

        if config['optimizer'] in ['sa', 'simulated_annealing']:
            tp = args['temp_schedule']
            args['temp_schedule'] = temp_schedules[tp[0]](*tp[1:])

        optimizer = optimizers[config['optimizer']](**args)

        best_nets = [
            optimizer.optimize(net_string, device_graph) for _ in range(repeats)
        ]

        best_times = []
        for net in best_nets:
            graph = ComputationGraph()
            graph.load_from_string(net)
            simulator = Simulator(graph, device_graph)
            execution_time = simulator.simulate(batch_size=128, batches=batches,
                                                pipeline_batches=pipeline_batches,
                                                comm_penalization=comm_penalty,
                                                comp_penalization=comp_penalty)
            best_times.append(execution_time)

        mean_time = sum(best_times) / len(best_times)

        if verbose:
            print(f'{mean_time}ms')

        if best_time is None or mean_time < best_time:
            best_time = mean_time
            best_params = combination

    return dict(best_params)


if __name__ == '__main__':
    VERBOSE = True

    ga_grid = {
        'population_size': [30, 50, 100],
        'crossover_rate': [0.2, 0.4, 0.6, 0.8],
    }

    grid = (
        ga_grid,
    )[0]

    config_path = (
        'configs/ga-malvik-resnet50.json',
    )[0]

    best_params = do_parameter_search(config_path, grid, verbose=VERBOSE)
    print(f'Best parameters: \n{best_params}')
