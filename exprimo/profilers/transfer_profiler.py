import numpy as np

from paleo.profilers.flops_profiler import FlopsProfiler as PaleoFlopsProfiler
from paleo.profilers.base import ProfilerOptions


def calculate_tensor_size(shape, dtype='float32'):
    return np.prod(shape) * np.dtype(dtype).itemsize


class TransferProfiler:
    @staticmethod
    def profile(layer_spec, comm_channel, parent_device, backward=False, batch_size=None, dtype='float32',
                comm_penalization=None, comp_penalization=None):
        layer = layer_spec.operation

        if batch_size:
            layer.batch_size = batch_size

        profiler_options = ProfilerOptions()
        direction = 'backward' if backward else 'forward'
        profiler_options.direction = direction
        profiler_options.use_cudnn_heuristics = False
        profiler_options.include_bias_and_activation = False
        profiler_options.ppp_comp = comp_penalization
        profiler_options.ppp_comm = comm_penalization

        profiler = PaleoFlopsProfiler(profiler_options, parent_device)

        num_bytes = calculate_tensor_size(layer.outputs, dtype)
        time = profiler.estimate_comm_time(
            num_bytes, comm_channel.bandwidth / 8, ppp=profiler.options.ppp_comm)

        return time
