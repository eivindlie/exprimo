from abc import abstractmethod

from exprimo.optimizers.utils import create_colocation_groups, evaluate_placement
import multiprocessing


class BaseOptimizer:

    def __init__(self, colocation_heuristic=None, verbose=False, batches=1, pipeline_batches=1, n_threads=-1,
                 score_save_period=None, simulator_comp_penalty=1, simulator_comm_penalty=1,
                 device_memory_utilization=1):
        self.colocation_heuristic = colocation_heuristic
        self.verbose = verbose
        self.batches = batches
        self.pipeline_batches = pipeline_batches
        self.simulator_comp_penalty = simulator_comp_penalty
        self.simulator_comm_penalty = simulator_comm_penalty
        self.device_memory_utilization = device_memory_utilization
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

    def evaluate_placement(self, net, device_graph, batch_size=128):
        return evaluate_placement(net, device_graph, batch_size=batch_size, batches=self.batches,
                                  pipeline_batches=self.pipeline_batches,
                                  comp_penalty=self.simulator_comp_penalty,
                                  comm_penalty=self.simulator_comm_penalty,
                                  device_memory_utilization=self.device_memory_utilization)

    @abstractmethod
    def optimize(self, net_string, device_graph):
        raise NotImplementedError()
