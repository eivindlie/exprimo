from abc import abstractmethod

from exprimo.optimizers.utils import create_colocation_groups
import multiprocessing


class BaseOptimizer:

    def __init__(self, colocation_heuristic=None, verbose=False, batches=1, pipeline_batches=1, n_threads=-1,
                 score_save_period=None):
        self.colocation_heuristic = colocation_heuristic
        self.verbose = verbose
        self.batches = batches
        self.pipeline_batches = pipeline_batches
        if n_threads == -1:
            self.n_threads = multiprocessing.cpu_count()
        else:
            self.n_threads = n_threads

        if score_save_period is None and verbose:
            self.score_save_period = verbose
        else:
            self.score_save_period = score_save_period

    def create_colocation_groups(self, layer_names):
        if not self.colocation_heuristic:
            return [[name] for name in layer_names]

        return create_colocation_groups(layer_names, self.colocation_heuristic)

    @abstractmethod
    def optimize(self, net_string, device_graph):
        raise NotImplementedError()
