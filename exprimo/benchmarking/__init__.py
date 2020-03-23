from statistics import mean
from exprimo.benchmarking.benchmark import benchmark_with_placement


def create_benchmark_function(model_type, batches=50, drop_batches=1, aggregate_function=mean, lr=0.01,
                              device_map=None):
    def benchmark_placement(placement):
        batch_times = benchmark_with_placement(model_type, placement, batches=batches, drop_batches=drop_batches,
                                               lr=lr, device_map=device_map)
        return aggregate_function(batch_times)

    return benchmark_placement
