import json
from random import randint
import random

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


def evaluate_placement(net, device_graph, batch_size=128, batches=1, pipeline_batches=1, memory_penalization_factor=1,
                       noise_std=0, comp_penalty=1, comm_penalty=1):
    net_string = json.dumps(net)
    graph = ComputationGraph()
    graph.load_from_string(net_string)
    simulator = Simulator(graph, device_graph)

    time = simulator.simulate(print_event_trace=False, print_memory_usage=False,
                              batch_size=batch_size, batches=batches, pipeline_batches=pipeline_batches,
                              memory_penalization_factor=memory_penalization_factor, comp_penalization=comp_penalty,
                              comm_penalization=comm_penalty)

    if noise_std:
        time += random.normalvariate(0, noise_std)
    return time


def apply_placement(net_string, placement, groups):
    net = json.loads(net_string)

    for i, device in enumerate(placement):
        for layer in groups[i]:
            layer_name = layer.name
            if layer_name in net['layers']:
                layer = net['layers'][layer_name]
            else:
                # Layer is inside a block
                block_name = layer_name.split('/')[0]
                layer_subname = layer_name[len(block_name) + 1:]
                layer = net['layers'][block_name]['layers'][layer_subname]
            layer['device'] = device

    return net


def get_device_assignment(net_dict):
    device_assignment = {}

    for layer_name in net_dict['layers'].keys():
        layer = net_dict['layers'][layer_name]

        try:
            device_assignment[layer_name] = layer['device']
        except KeyError:
            device_assignment[layer_name] = 0

        if layer['type'] == 'Block':
            for sublayer_name in layer['layers'].keys():
                sublayer = layer['layers'][sublayer_name]
                try:
                    device_assignment[f'{layer_name}/{sublayer_name}'] = sublayer['device']
                except KeyError:
                    device_assignment[f'{layer_name}/{sublayer_name}'] = 0
    return device_assignment
