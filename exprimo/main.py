import json

from graph import ComputationGraph
from device import DeviceGraph
from simulator import Simulator
from plotting import plot_event_trace

from exprimo.optimizers.utils import prefix_heuristic, create_colocation_groups, apply_placement

if __name__ == '__main__':
    graph_file = '../nets/resnet50.json'
    device_file = '../device_graphs/cluster2.json'
    batches = 1
    pipeline_batches = 1

    with open(graph_file) as f:
        net_dict = json.load(f)

    net_string = json.dumps(net_dict)

    graph = ComputationGraph(force_device=None)
    graph.load_from_string(net_string)
    device_graph = DeviceGraph.load_from_file(device_file)
    simulator = Simulator(graph, device_graph)
    run_time, events = simulator.simulate(batch_size=128, batches=batches, pipeline_batches=pipeline_batches,
                                          return_event_trace=True, print_event_trace=True)

    plot_event_trace(events, simulator, show_transfer_lines=True, plot_op_time_distribution=True)

    print()
    print(f'Total batch run time: {run_time:,.2f}ms')
