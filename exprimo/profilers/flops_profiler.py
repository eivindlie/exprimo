from paleo.profilers.flops_profiler import FlopsProfiler as PaleoFlopsProfiler
from paleo.profilers.base import ProfilerOptions, TimeMeasure


class FlopsProfiler:
    @staticmethod
    def profile(layer_spec, device, backward=False):
        layer = layer_spec.operation

        profiler_options = ProfilerOptions()
        direction = 'backward' if backward else 'forward'
        profiler_options.direction = direction
        profiler_options.use_cudnn_heuristics = False
        profiler_options.include_bias_and_activation = False

        profiler = PaleoFlopsProfiler(profiler_options, device)
        time = profiler.profile(layer)

        return time.comp_time
