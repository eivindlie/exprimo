from exprimo import DeviceGraph
from exprimo.optimizers import SimulatedAnnealingOptimizer, HillClimbingOptimizer, RandomHillClimbingOptimizer

# optimizer = RandomHillClimbingOptimizer(patience=100)
# optimizer = LinearSearchOptimizer(prefix_heuristic(prefix_length=4))
optimizer = SimulatedAnnealingOptimizer(temp_schedule=50, steps=1000)
device_graph = DeviceGraph.load_from_file('../device_graphs/cluster2-reduced-memory.json')
with open('../nets/mnist.json') as f:
    net_string = f.read()

best_net = optimizer.optimize(net_string, device_graph)
print(f'Best discovered configuration: {[layer["device"] for layer in best_net["layers"].values()]}')

