import json
import sys

from exprimo import DeviceGraph, Simulator, plot_event_trace, ComputationGraph, log
from exprimo.optimizers import SimulatedAnnealingOptimizer, HillClimbingOptimizer, RandomHillClimbingOptimizer, \
    GAOptimizer, LinearSearchOptimizer, MapElitesOptimizer
from exprimo.benchmarking import create_benchmark_function
from exprimo.optimizers.particle_swarm_optimizer import ParticleSwarmOptimizer


config_path = 'configs/malvik-resnet50-map-elites.json'

if len(sys.argv) > 1:
    config_path = sys.argv[1]

with open(config_path) as f:
    config = json.load(f)

device_graph_path = config['device_graph_path']
net_path = config['net_path']

args = config.get('optimizer_args', {})

batches = args.get('batches', 1)
pipeline_batches = args.get('pipeline_batches', 1)

args['batches'] = batches
args['pipeline_batches'] = pipeline_batches

if 'benchmarking_function' in args:
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

optimizer = optimizers[config['optimizer']](**args)

device_graph = DeviceGraph.load_from_file(device_graph_path)
with open(net_path) as f:
    net_string = f.read()

log(f'Optimizing {net_path} on {device_graph_path} using {optimizer}')
log(args)
log()


best_net = optimizer.optimize(net_string, device_graph)
net_dict = json.loads(best_net)

graph = ComputationGraph()
graph.load_from_string(best_net)
simulator = Simulator(graph, device_graph)
execution_time, events = simulator.simulate(batch_size=128,
                                            print_memory_usage=config.get('print_memory_usage', False),
                                            print_event_trace=config.get('print_event_trace', False),
                                            return_event_trace=True, batches=batches, pipeline_batches=pipeline_batches,
                                            comm_penalization=comm_penalty, comp_penalization=comp_penalty)

if config.get('plot_event_trace', True):
    plot_event_trace(events, simulator)

log('\n')
# print(f'Best discovered configuration: {[layer["device"] for layer in net_dict["layers"].values()]}')
log(f'Execution time: {execution_time:.2f}ms')

