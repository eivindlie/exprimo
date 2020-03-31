import json

from exprimo import DeviceGraph, Simulator, plot_event_trace, ComputationGraph
from exprimo.optimizers import SimulatedAnnealingOptimizer, HillClimbingOptimizer, RandomHillClimbingOptimizer, \
    GAOptimizer, exponential_multiplicative_decay, LinearSearchOptimizer
from exprimo.benchmarking import create_benchmark_function
from exprimo.optimizers.particle_swarm_optimizer import ParticleSwarmOptimizer
from exprimo.optimizers.utils import prefix_heuristic

config_path = 'configs/malvik-resnet50.json'

with open(config_path) as f:
    config = json.load(f)

device_graph_path = config['device_graph_path']
net_path = config['net_path']

batches = config.get('batches', 1)
pipeline_batches = config.get('pipeline_batches', 1)

args = config.get('optimizer_args', {})
args['batches'] = batches
args['pipeline_batches'] = pipeline_batches

if 'benchmarking_function' in args:
    args['benchmarking_function'] = create_benchmark_function(**args['benchmarking_function'])


optimizers = {
    'random_hill_climber': RandomHillClimbingOptimizer,
    'linear_search': LinearSearchOptimizer,
    'simulated_annealing': SimulatedAnnealingOptimizer,
    'sa': SimulatedAnnealingOptimizer,
    'genetic_algorithm': GAOptimizer,
    'ga': GAOptimizer,
    'pso': ParticleSwarmOptimizer,
    'particle_swarm': ParticleSwarmOptimizer
}

optimizer = optimizers[config['optimizer']](**args)

device_graph = DeviceGraph.load_from_file(device_graph_path)
with open(net_path) as f:
    net_string = f.read()

print(f'Optimizing {net_path} on {device_graph_path} using {optimizer}')
print(args)
print()


best_net = optimizer.optimize(net_string, device_graph)
net_dict = json.loads(best_net)

graph = ComputationGraph()
graph.load_from_string(best_net)
simulator = Simulator(graph, device_graph)
execution_time, events = simulator.simulate(batch_size=128, print_memory_usage=True, print_event_trace=True,
                                            return_event_trace=True, batches=batches, pipeline_batches=pipeline_batches)

plot_event_trace(events, simulator)

print('\n')
# print(f'Best discovered configuration: {[layer["device"] for layer in net_dict["layers"].values()]}')
print(f'Execution time: {execution_time:.2f}ms')

