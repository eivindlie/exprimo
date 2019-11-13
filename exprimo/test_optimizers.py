import json

from exprimo import DeviceGraph, Simulator, plot_event_trace, ComputationGraph
from exprimo.optimizers import SimulatedAnnealingOptimizer, HillClimbingOptimizer, RandomHillClimbingOptimizer, \
    GAOptimizer, exponential_multiplicative_decay

batches = 1
pipeline_batches = 1

# optimizer = RandomHillClimbingOptimizer(patience=100)
# optimizer = LinearSearchOptimizer(prefix_heuristic(prefix_length=4))
# optimizer = SimulatedAnnealingOptimizer(temp_schedule=exponential_multiplicative_decay(50, 0.98),
#                                         steps=30000, batches=batches,
#                                         pipeline_batches=pipeline_batches, verbose=True)
optimizer = GAOptimizer(population_size=100, mutation_rate=0.01, elite_size=20, steps=500, verbose=True)

device_graph = DeviceGraph.load_from_file('../device_graphs/cluster2-reduced-memory.json')
with open('../nets/alex_v2.json') as f:
    net_string = f.read()

best_net = optimizer.optimize(net_string, device_graph)
net_dict = json.loads(best_net)

graph = ComputationGraph()
graph.load_from_string(best_net)
simulator = Simulator(graph, device_graph)
execution_time, events = simulator.simulate(batch_size=128, print_memory_usage=True, print_event_trace=True,
                                            return_event_trace=True, batches=batches, pipeline_batches=pipeline_batches)

plot_event_trace(events)

print('\n')
print(f'Best discovered configuration: {[layer["device"] for layer in net_dict["layers"].values()]}')
print(f'Execution time: {execution_time:.2f}ms')

