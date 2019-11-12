from abc import abstractmethod


class BaseOptimizer:

    def __init__(self, colocation_heuristic=None, verbose=False, batches=1, pipeline_batches=1):
        self.colocation_heuristic = colocation_heuristic
        self.verbose = verbose
        self.batches = batches
        self.pipeline_batches = pipeline_batches

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
