from abc import abstractmethod


class BaseOptimizer:

    def __init__(self, colocation_heuristic=None, verbose=False):
        self.colocation_heuristic = colocation_heuristic
        self.verbose = verbose

    def create_colocation_groups(self, layer_names):
        if not self.colocation_heuristic:
            return [[name] for name in layer_names]

        groups = []
        for layer in layer_names:
            for group in groups:
                if self.colocation_heuristic(layer, group[0]):
                    group.append(layer)
                    break
            else:
                groups.append([layer])
        return groups

    @abstractmethod
    def optimize(self, net_string, device_graph):
        raise NotImplementedError()
