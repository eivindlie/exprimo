import json
import os

from graph import ComputationGraph
from device import DeviceGraph
from simulator import Simulator
from plotting import plot_event_trace

from exprimo.optimizers.utils import prefix_heuristic, create_colocation_groups, apply_placement


def place_all_layers_to_device(net_dict, device=1):
    for layer_name in net_dict['layers'].keys():
        layer = net_dict['layers'][layer_name]

        if layer['type'] == 'Block':
            for sublayer_name in layer['layers']:
                sublayer = layer['layers'][sublayer_name]

                sublayer['device'] = device

        layer['device'] = device

    return net_dict


if __name__ == '__main__':
    graph_folder = '../experiment_results/sim_real_comp/nets'
    device_file = '../device_graphs/malvik-double-bandwidth.json'

    results_file = '../experiment_results/sim_real_comp/scores_double_bandwidth.csv'

    with open(results_file, 'w') as f:
        f.write('')

    batches = 1
    pipeline_batches = 1

    dir_list = os.listdir(graph_folder)
    for i, file_path in enumerate(dir_list):
        if not file_path.endswith('.json'):
            continue

        print(f'Simulating assignment {i + 1}/{len(dir_list)}: {file_path}')

        generation = int(file_path.replace('gen_', '').replace('.json', ''))

        with open(os.path.join(graph_folder, file_path)) as f:
            net_dict = json.load(f)

        net_string = json.dumps(net_dict)

        graph = ComputationGraph(force_device=None)
        graph.load_from_string(net_string)
        device_graph = DeviceGraph.load_from_file(device_file)
        simulator = Simulator(graph, device_graph)
        run_time, events = simulator.simulate(batch_size=128, batches=batches, pipeline_batches=pipeline_batches,
                                              return_event_trace=True, print_event_trace=False,
                                              print_memory_usage=False)

        with open(results_file, 'a') as f:
            f.write(f'{generation}, {run_time}\n')

        # plot_event_trace(events, simulator, show_transfer_lines=True, plot_op_time_distribution=True)

        print()
        print(f'Total batch run time: {run_time:,.2f}ms')
