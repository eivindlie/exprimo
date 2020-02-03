import json

from graph import ComputationGraph
from device import DeviceGraph
from simulator import Simulator
from plotting import plot_event_trace

from exprimo.optimizers.utils import prefix_heuristic, create_colocation_groups, apply_placement

if __name__ == '__main__':
    graph_file = '../nets/inception_v3.json'
    device_file = '../device_graphs/cluster2.json'
    batches = 1
    pipeline_batches = 1

    with open(graph_file) as f:
        net_dict = json.load(f)
    # groups = create_colocation_groups(net_dict['layers'], prefix_heuristic(prefix_length=5))
    # groups = [[layer] for layer in net_dict['layers']]
    #
    # placement = [
    #                  2,    # [['data'],
    #                  2,    #  ['conv1'],
    #                  2,    #  ['pool1'],
    #                  2, 2, 2, 2, 2,    # ['res2a_branch1', 'res2a_branch2a', 'res2a_branch2b', 'res2a_branch2c', 'res2a'],
    #                  2, 2, 2, 2,    # ['res2b_branch2a', 'res2b_branch2b', 'res2b_branch2c', 'res2b'],
    #                  2, 2, 2, 2,    # ['res2c_branch2a', 'res2c_branch2b', 'res2c_branch2c', 'res2c'],
    #                  1, 2, 2, 2, 1,    # ['res3a_branch1', 'res3a_branch2a', 'res3a_branch2b', 'res3a_branch2c', 'res3a'],
    #                  1, 1, 1, 1,    # ['res3b_branch2a', 'res3b_branch2b', 'res3b_branch2c', 'res3b'],
    #                  1, 1, 1, 1,    # ['res3c_branch2a', 'res3c_branch2b', 'res3c_branch2c', 'res3c'],
    #                  1, 1, 1, 1,    # ['res3d_branch2a', 'res3d_branch2b', 'res3d_branch2c', 'res3d'],
    #                  1, 1, 1, 1, 1,    # ['res4a_branch1', 'res4a_branch2a', 'res4a_branch2b', 'res4a_branch2c', 'res4a'],
    #                  1, 1, 1, 1,   # ['res4b_branch2a', 'res4b_branch2b', 'res4b_branch2c', 'res4b'],
    #                  1, 1, 1, 1,    # ['res4c_branch2a', 'res4c_branch2b', 'res4c_branch2c', 'res4c'],
    #                  1, 1, 1, 1,    # ['res4d_branch2a', 'res4d_branch2b', 'res4d_branch2c', 'res4d'],
    #                  1, 1, 1, 1,    # ['res4e_branch2a', 'res4e_branch2b', 'res4e_branch2c', 'res4e'],
    #                  1, 1, 1, 1,    # ['res4f_branch2a', 'res4f_branch2b', 'res4f_branch2c', 'res4f'],
    #                  1, 1, 1, 1, 1,    # ['res5a_branch1', 'res5a_branch2a', 'res5a_branch2b', 'res5a_branch2c', 'res5a'],
    #                  1, 1, 1, 1,    # ['res5b_branch2a', 'res5b_branch2b', 'res5b_branch2c', 'res5b'],
    #                  1, 1, 1, 1,    # ['res5c_branch2a', 'res5c_branch2b', 'res5c_branch2c', 'res5c'],
    #                  1,    # ['pool5'],
    #                  1,    # ['fc1000'],
    #                  1    # ['prob']]
    #              ]
    #
    # net_dict = apply_placement(json.dumps(net_dict), placement, groups)
    # for layer_name, layer in net_dict['layers'].items():
    #     print(f'{layer_name}: {layer["device"]}')
    net_string = json.dumps(net_dict)

    graph = ComputationGraph()
    graph.load_from_string(net_string)
    device_graph = DeviceGraph.load_from_file(device_file)
    simulator = Simulator(graph, device_graph)
    run_time, events = simulator.simulate(batch_size=128, batches=batches, pipeline_batches=pipeline_batches,
                                          return_event_trace=True, print_event_trace=False)

    plot_event_trace(events)

    print()
    print(f'Total batch run time: {run_time:,.2f}ms')
