import json
import os

import sys

import exprimo
from exprimo import DeviceGraph, Simulator, plot_event_trace, ComputationGraph, log
from exprimo.optimizers import SimulatedAnnealingOptimizer, HillClimbingOptimizer, RandomHillClimbingOptimizer, \
    GAOptimizer, LinearSearchOptimizer, MapElitesOptimizer
from exprimo.benchmarking import create_benchmark_function
from exprimo.optimizers.particle_swarm_optimizer import ParticleSwarmOptimizer
from exprimo.optimizers.utils import get_device_assignment
from exprimo.optimizers.simulated_annealing import temp_schedules

config_path = 'configs/me-malvik-resnet50.json'


def optimize_with_config(config_path=None, config=None, verbose=True, set_log_dir=False):
    assert config_path or config, 'Either a config path or a config dictionary must be provided'
    assert config is None or isinstance(config, dict), 'config must be a dictionary'

    if config_path:
        with open(config_path) as f:
            config = json.load(f)

    device_graph_path = config['device_graph_path']
    net_path = config['net_path']

    log_dir = config.get('log_dir', '')
    if log_dir and set_log_dir:
        exprimo.set_log_dir(log_dir)

    if verbose:
        log('\n\n\n')
        log('='*100)
        log('EXPRIMO OPTIMIZATION'.rjust(60))
        log('='*100)
        log()

    if verbose:
        if config_path:
            log(f'Using config path {config_path}')
        else:
            log('Using config provided as dictionary')

    args = config.get('optimizer_args', {})

    batches = args.get('batches', 1)
    pipeline_batches = args.get('pipeline_batches', 1)

    args['batches'] = batches
    args['pipeline_batches'] = pipeline_batches

    if 'benchmarking_function' in args and isinstance(args['benchmarking_function'], dict):
        args['benchmarking_function'] = create_benchmark_function(**args['benchmarking_function'])

    comp_penalty = args.get('simulator_comp_penalty', 1.0)
    comm_penalty = args.get('simulator_comm_penalty', 1.0)

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

    if config['optimizer'] in ['sa', 'simulated_annealing'] and isinstance(args['temp_schedule'], list):
        tp = args['temp_schedule']
        args['temp_schedule'] = temp_schedules[tp[0]](*tp[1:])

    optimizer = optimizers[config['optimizer']](**args)

    device_graph = DeviceGraph.load_from_file(device_graph_path)
    with open(net_path) as f:
        net_string = f.read()

    if verbose:
        log(f'Optimizing {net_path} on {device_graph_path} using {optimizer}')
        log(args)
        log()


    best_net = optimizer.optimize(net_string, device_graph)
    net_dict = json.loads(best_net)

    graph = ComputationGraph()
    graph.load_from_string(best_net)
    simulator = Simulator(graph, device_graph)
    simulated_execution_time, events = simulator.simulate(batch_size=128,
                                                          print_memory_usage=config.get('print_memory_usage', False),
                                                          print_event_trace=config.get('print_event_trace', False),
                                                          return_event_trace=True, batches=batches, pipeline_batches=pipeline_batches,
                                                          comm_penalization=comm_penalty, comp_penalization=comp_penalty)

    if config.get('plot_event_trace', True):
        save_path = os.path.join(exprimo.get_log_dir(), 'event_trace.pdf')
        plot_event_trace(events, simulator, save_path=save_path)

    if verbose:
        log('\n')
        # print(f'Best discovered configuration: {[layer["device"] for layer in net_dict["layers"].values()]}')
        log(f'Simulated execution time: {simulated_execution_time:.2f}ms')

        if config.get('benchmark_solution', False) and args.get('benchmarking_function', None):
            device_assignment = get_device_assignment(net_dict)
            time = args['benchmarking_function'](device_assignment)
            log(f'Benchmarked execution time: {time:.2f}ms')

    return best_net, simulated_execution_time


if __name__ == '__main__':
    if len(sys.argv) > 1:
        config_path = sys.argv[1]

    optimize_with_config(config_path, set_log_dir=True)