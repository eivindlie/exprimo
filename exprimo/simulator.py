"""
Execution simulator based on the technique described in Placeto (Addanki et al., 2019, https://arxiv.org/abs/1906.08879)
"""

from collections import deque

from device import DeviceGraph
from graph import ComputationGraph


class Simulator:

    def __init__(self, computation_graph, device_graph):
        if type(device_graph) == str:
            self.device_graph = DeviceGraph.load_from_file(device_graph)
        else:
            self.device_graph = device_graph

        if type(computation_graph) == str:
            self.computation_graph = ComputationGraph(computation_graph)
        else:
            self.computation_graph = computation_graph

    def simulate(self, print_event_trace=True):
        op_queues = [deque() for device in self.device_graph.devices]
        transfer_queues = [deque() for device in self.device_graph.devices]
        events = []

        for layer in self.computation_graph.topological_order:
            if not len(layer.inbounds):
                op_queues[layer['device']].append(layer)
                if len(op_queues[layer['device']]) == 1:
                    events.append(Event('wakeup', layer['device'], 0))

        def op_done(event):
            pass

        def transfer_done(event):
            pass

        def wakeup(event):
            pass

        event_map = {
            'op_done': op_done,
            'transfer_done': transfer_done,
            'wakeup': wakeup
        }

        e = 0
        while True:
            if e >= len(events):
                break

            event = events[e]
            event_map[event.type](event)

            e += 1

        if print_event_trace:
            for event in events:
                print(event)


class Event:

    def __init__(self, type, device, start_time, end_time=None, operation=None, parent_device=None):
        self.type = type
        self.device = device
        self.start_time = start_time
        self.end_time = end_time
        self.operation = operation
        self.parent_device = parent_device
        self.handled = False

    def __str__(self):
        return f'[{self.type.capitalize()}] ' \
               f'Device: {self.device} ' \
               f'Start time: {self.start_time} ' \
               f'{f"End time: {self.end_time} " if self.end_time else ""}' \
               f'{f"Name: {self.operation.name} " if self.operation else ""}'
