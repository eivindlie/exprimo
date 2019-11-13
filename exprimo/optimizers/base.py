from abc import abstractmethod

from optimizers.utils import create_colocation_groups


class BaseOptimizer:

    def __init__(self, colocation_heuristic=None, verbose=False, batches=1, pipeline_batches=1):
        self.colocation_heuristic = colocation_heuristic
        self.verbose = verbose
        self.batches = batches
        self.pipeline_batches = pipeline_batches

    def create_colocation_groups(self, layer_names):
        if not self.colocation_heuristic:
            return [[name] for name in layer_names]

        return create_colocation_groups(layer_names, self.colocation_heuristic)

    @abstractmethod
    def optimize(self, net_string, device_graph):
        raise NotImplementedError()
