import json

from exprimo import DeviceGraph, Simulator, plot_event_trace, ComputationGraph
from exprimo.optimizers import SimulatedAnnealingOptimizer, HillClimbingOptimizer, RandomHillClimbingOptimizer

batches = 4
pipeline_batches = 2

# optimizer = RandomHillClimbingOptimizer(patience=100)
# optimizer = LinearSearchOptimizer(prefix_heuristic(prefix_length=4))
optimizer = SimulatedAnnealingOptimizer(temp_schedule=50, steps=3000, batches=batches,
                                        pipeline_batches=pipeline_batches)
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

print(f'Best discovered configuration: {[layer["device"] for layer in net_dict["layers"].values()]}')
print(f'Execution time: {execution_time:.2f}ms')

