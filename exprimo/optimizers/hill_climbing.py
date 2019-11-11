import json
from random import randint

from exprimo import DeviceGraph
from exprimo.optimizers.base import BaseOptimizer
from exprimo.optimizers.utils import generate_random_placement, evaluate_placement, apply_placement


class HillClimbingOptimizer(BaseOptimizer):

    def optimize(self, net_string, device_graph):
        n_devices = len(device_graph.devices)

        def generate_neighbours(placement):
            if n_devices == 1:
                return

            i = 0
            while i < len(placement):
                p = placement[i]
                if p < n_devices - 1:
                    n = placement[:]
                    n[i] = p + 1
                    yield n
                if p > 0:
                    n = placement[:]
                    n[i] = p - 1
                    yield n
                i += 1

        net = json.loads(net_string)
        groups = self.create_colocation_groups(net['layers'].keys())

        placement = generate_random_placement(len(groups), n_devices)
        score = evaluate_placement(apply_placement(net_string, placement, groups), device_graph)

        i = 0
        while True:
            i += 1
            if self.verbose:
                print(f'Iteration {i}. Best running time: {score:.2f}ms')

            for n in generate_neighbours(placement):
                new_score = evaluate_placement(apply_placement(net_string, n, groups), device_graph)
                if new_score < score:
                    placement = n
                    score = new_score
                    break
            else:
                break

        return placement


if __name__ == '__main__':
    optimizer = HillClimbingOptimizer(verbose=True)
    device_graph = DeviceGraph.load_from_file('../device_graphs/cluster1.json')
    with open('../nets/mnist.json') as f:
        net_string = f.read()

    best_net = optimizer.optimize(net_string, device_graph)
    pass
