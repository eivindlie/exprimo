from paleo.profilers.flops_profiler import FlopsProfiler as PaleoFlopsProfiler
from paleo.profilers.base import ProfilerOptions


class TransferProfiler:
    @staticmethod
    def profile(layer_spec, comm_channel, parent_device, backward=False, batch_size=None):
        layer = layer_spec.operation

        if batch_size:
            layer.batch_size = batch_size

        profiler_options = ProfilerOptions()
        direction = 'backward' if backward else 'forward'
        profiler_options.direction = direction
        profiler_options.use_cudnn_heuristics = False
        profiler_options.include_bias_and_activation = False

        profiler = PaleoFlopsProfiler(profiler_options, parent_device)
        time = profiler.estimate_remote_fetch(layer, 0, [1], comm_channel.bandwidth / 8)

        return time.comm_time
