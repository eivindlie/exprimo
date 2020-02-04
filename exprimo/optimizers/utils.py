import json
from random import randint

from exprimo import ComputationGraph, Simulator


def prefix_heuristic(prefix_length=None, delimiter=None):
    def should_colocate(a, b):
        if prefix_length:
            return a[:prefix_length] == b[:prefix_length]
        if delimiter:
            return a.split(delimiter)[0] == b.split(delimiter)

    return should_colocate


def create_colocation_groups(layer_names, colocation_heuristic):
    groups = []
    for layer in layer_names:
        for group in groups:
            if colocation_heuristic(layer, group[0]):
                group.append(layer)
                break
        else:
            groups.append([layer])
    return groups


def generate_random_placement(n_groups, n_devices):
    placement = []
    for i in range(n_groups):
        placement.append(randint(0, n_devices - 1))
    return placement


def evaluate_placement(net, device_graph, batch_size=128, batches=1, pipeline_batches=1, memory_penalization_factor=1):
    net_string = json.dumps(net)
    graph = ComputationGraph()
    graph.load_from_string(net_string)
    simulator = Simulator(graph, device_graph)

    return simulator.simulate(print_event_trace=False, print_memory_usage=False,
                              batch_size=batch_size, batches=batches, pipeline_batches=pipeline_batches,
                              memory_penalization_factor=memory_penalization_factor)


def apply_placement(net_string, placement, groups):
    net = json.loads(net_string)

    for i, device in enumerate(placement):
        for layer in groups[i]:
            if layer in net['layers']:
                net['layers'][layer]['device'] = device
            else:
                # Layer is inside a block
                block = layer.split('/')[0]
                net['layers'][block]['layers'][layer]['device'] = device

    return net