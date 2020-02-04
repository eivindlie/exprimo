import json
from itertools import product
from tqdm import tqdm

from exprimo.device import DeviceGraph
from exprimo.optimizers.base import BaseOptimizer

from exprimo.optimizers.utils import prefix_heuristic, apply_placement, evaluate_placement
from graph import get_flattened_layer_names


class LinearSearchOptimizer(BaseOptimizer):
    """
    A naive optimizer that carries out a brute-force search for the best placement. Will take significant time
    to run if the search space is too large.
    """

    def optimize(self, net_string, device_graph):
        """
        Optimizes a configuration for the given net on the given hardware.
        :param net: The network that should be optimized, given as a json string.
        :param device_graph: The device graph that the network should be optimized for.
        :return: A network JSON string with optimized device placements.
        """

        net = json.loads(net_string)
        groups = self.create_colocation_groups(get_flattened_layer_names(net_string))

        best_score = -1
        best_net = None
        for comb in tqdm(product(range(len(device_graph.devices)), repeat=len(groups)),
                         total=len(device_graph.devices) ** len(groups), unit='placements'):

            net = apply_placement(net_string, comb, groups)

            score = evaluate_placement(net, device_graph, batch_size=128, batches=1)

            if score < best_score or best_net is None:
                best_net = net
                best_score = score

        return best_net
