from paleo.profilers.flops_profiler import FlopsProfiler as PaleoFlopsProfiler
from paleo.profilers.base import ProfilerOptions


class FlopsProfiler:
    @staticmethod
    def profile(layer_spec, device, backward=False, batch_size=None, comm_penalization=None, comp_penalization=None):
        layer = layer_spec.operation

        assert layer is not None, f'{layer_spec} has no operation'

        if batch_size:
            layer.batch_size = batch_size

        profiler_options = ProfilerOptions()
        direction = 'backward' if backward else 'forward'
        profiler_options.direction = direction
        profiler_options.use_cudnn_heuristics = False
        profiler_options.include_bias_and_activation = False
        profiler_options.ppp_comm = comm_penalization
        profiler_options.ppp_comp = comp_penalization

        profiler = PaleoFlopsProfiler(profiler_options, device)
        time = profiler.profile(layer, cross_device_bandwidth=0)

        return time.comp_time + time.comm_time
