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


def generate_random_placement(net_string, n_devices):
    net = json.loads(net_string)

    for layer_name, layer in net['layers'].items():
        layer['device'] = randint(0, n_devices - 1)

    return net


def evaluate_placement(net, device_graph, batch_size=128, batches=1):
    net_string = json.dumps(net)
    graph = ComputationGraph()
    graph.load_from_string(net_string)
    simulator = Simulator(graph, device_graph)

    return simulator.simulate(print_event_trace=False, batch_size=batch_size, batches=batches)


def apply_placement(net_string, placement, groups):
    net = json.loads(net_string)

    for i, device in enumerate(placement):
        for layer in groups[i]:
            net['layers'][layer]['device'] = device

    return net