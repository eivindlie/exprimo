import json
from itertools import product
from tqdm import tqdm

from device import DeviceGraph
from graph import ComputationGraph
from simulator import Simulator

from optimizers.utils import prefix_heuristic, create_colocation_groups





class LinearSearchOptimizer:
    """
    A naive optimizer that carries out a brute-force search for the best placement. Will take significant time
    to run if the search space is too large.
    """

    def __init__(self, colocation_heuristic=None):
        self.colocation_heuristic = colocation_heuristic

    def optimize(self, net_string, device_graph):
        """
        Optimizes a configuration for the given net on the given hardware.
        :param net: The network that should be optimized, given as a json string.
        :param device_graph: The device graph that the network should be optimized for.
        :return: A network JSON string with optimized device placements.
        """

        net = json.loads(net_string)
        groups = create_colocation_groups(net['layers'].keys(), self.colocation_heuristic)

        best_score = -1
        best_net = None
        for comb in tqdm(product(range(len(device_graph.devices)), repeat=len(groups)),
                         total=len(device_graph.devices) ** len(groups), unit='placements'):
            net = json.loads(net_string)
            for i, device in enumerate(comb):
                for layer in groups[i]:
                    net['layers'][layer]['device'] = device

            graph = ComputationGraph()
            graph.load_from_string(json.dumps(net))
            simulator = Simulator(graph, device_graph)

            score = simulator.simulate(print_event_trace=False, batch_size=128, batches=2)

            if score < best_score or best_net is None:
                best_net = net
                best_score = score

        return best_net


if __name__ == '__main__':
    optimizer = LinearSearchOptimizer(prefix_heuristic(prefix_length=4))
    device_graph = DeviceGraph.load_from_file('../device_graphs/cluster1.json')
    with open('../nets/resnet50.json') as f:
        net_string = f.read()

    best_net = optimizer.optimize(net_string, device_graph)
    pass
