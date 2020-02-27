import json

from exprimo import DeviceGraph, Simulator, plot_event_trace, ComputationGraph
from exprimo.optimizers import SimulatedAnnealingOptimizer, HillClimbingOptimizer, RandomHillClimbingOptimizer, \
    GAOptimizer, exponential_multiplicative_decay
from optimizers.particle_swarm_optimizer import ParticleSwarmOptimizer
from optimizers.utils import prefix_heuristic

batches = 1
pipeline_batches = 1

device_graph_path = 'device_graphs/malvik.json'
net_path = 'nets/resnet50.json'

args = {
    'plot_fitness_history': True,
    'generations': 500,
    'population_size': 100,
    'mutation_rate': 0.4,
    'crossover_rate': 0.8,
    'crossover_type': '1-point',
    'parent_selection_mechanism': 'rank',
    'evolve_mutation_rate': True,
    'verbose': 25,
    'elite_size': 5,
    'max_mutation_rate': 0.9,
    'min_mutation_rate': 0.05,
    'print_diversity': True,
    'include_trivial_solutions_in_initialization': False,
    'pipeline_batches': pipeline_batches,
    'batches': batches,
    'checkpoint_period': 25,
    'checkpoint_dir': 'experiment_results/sim_real_comp/nets',
    'n_threads': -1
}
# optimizer = RandomHillClimbingOptimizer(patience=100)
# optimizer = LinearSearchOptimizer(prefix_heuristic(prefix_length=4))
# optimizer = SimulatedAnnealingOptimizer(temp_schedule=exponential_multiplicative_decay(50, 0.98),
#                                         steps=30000, batches=batches,
#                                         pipeline_batches=pipeline_batches, verbose=True)
optimizer = GAOptimizer(**args)
# optimizer = GAIndirectOptimizer(population_size=50, mutation_rate=0.05, elite_size=10, steps=500,
#                                 verbose=False,
#                                 use_caching=True,
#                                 colocation_heuristic=prefix_heuristic(prefix_length=5))

# optimizer = ParticleSwarmOptimizer(w=10, l1=20, l2=10, steps=20)

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
