"""
Execution simulator based on the technique described in Placeto (Addanki et al., 2019, https://arxiv.org/abs/1906.08879)
"""

from collections import deque

from device import DeviceGraph
from graph import ComputationGraph
from profilers.flops_profiler import FlopsProfiler
from profilers.transfer_profiler import TransferProfiler


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

    def simulate(self, print_event_trace=True, include_backward=True, return_event_trace=False):
        op_queues = [deque() for device in self.device_graph.devices]
        transfer_queues = [deque() for comm_channel in self.device_graph.comm_channels]
        events = []

        for layer in self.computation_graph.topological_order:
            layer.forward_done = False
            layer.backward_done = False
            if not len(layer.inbounds):
                op_queues[layer['device']].append((layer, False))
                if len(op_queues[layer['device']]) == 1:
                    events.append(Event('wakeup', layer['device'], 0, subtype='op'))

        def run_op(op, backward, start_time):
            device = self.device_graph.devices[op['device']].device
            run_time = FlopsProfiler.profile(op, device, backward)
            end_time = start_time + run_time
            events.append(Event('op_done', op['device'], start_time,
                                end_time=end_time, operation=(op, backward)))

        def run_transfer(op, backward, comm_channel_id, target, start_time):
            parent_device = self.device_graph.devices[op['device']].device
            comm_channel = self.device_graph.comm_channels[comm_channel_id]
            transfer_time = TransferProfiler.profile(op, comm_channel, parent_device, backward)
            end_time = start_time + transfer_time
            events.append(Event('transfer_done', comm_channel_id, start_time,
                                operation=((op, backward), op['device'], target),
                                end_time=end_time))

        def can_run(op, backward):
            parents = op.outbounds if backward else op.inbounds
            for parent in parents:
                if (backward and not parent.backward_done) or (not backward and not parent.forward_done):
                    return False
            return True

        def op_done(event):
            op, backward = event.operation
            if backward:
                op.backward_done = True
                children = op.inbounds
            else:
                op.forward_done = True
                children = op.outbounds

            for child in children:
                if child['device'] != op['device']:
                    op_device = self.device_graph.devices[op['device']]
                    child_device = self.device_graph.devices[child['device']]
                    comm_channel = op_device.neighbours[child_device]

                    if backward:
                        # If we are doing the backward pass, we are transferring the gradients, which are
                        # equal in size to the previous layer. We therefore set the op as child instead of op.
                        transfer_queues[comm_channel.id].append(((child, backward), op_device, child_device))
                    else:
                        transfer_queues[comm_channel.id].append(((op, backward), op_device, child_device))
                    if len(transfer_queues[comm_channel.id]) == 1:
                        events.append(Event('wakeup', comm_channel.id, event.end_time, subtype='transfer'))
                else:
                    if can_run(child, backward):
                        op_queues[child['device']].append((child, backward))
                        # Don't need to check if device is free, as it is the current device

            if not backward and not len(children) and include_backward:
                op_queues[op['device']].append((op, True))

            if len(op_queues[event.device]):
                op2, backward2 = op_queues[event.device].popleft()
                run_op(op2, backward2, event.end_time)

        def transfer_done(event):
            (op, backward), start_device, end_device = event.operation
            children = op.inbounds if backward else op.outbounds

            for child in children:
                if can_run(child, backward):
                    op_queues[child['device']].append((child, backward))
                    if len(op_queues[child['device']]) == 1:
                        events.append(Event('wakeup', child['device'], event.end_time, subtype='op'))

            if len(transfer_queues[event.device]):
                (op2, backward2), start_device2, end_device2 = transfer_queues[event.device].popleft()
                run_transfer(op2, backward2, event.device, end_device2, event.end_time)

        def wakeup(event):
            event.end_time = event.start_time
            if event.subtype == 'transfer':
                (op, backward), start_device, end_device = transfer_queues[event.device].popleft()
                run_transfer(op, backward, event.device, end_device, event.start_time)
            else:
                (op, backward) = op_queues[event.device].popleft()
                run_op(op, backward, event.start_time)

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
            event.handled = True
            event_map[event.type](event)

            e += 1

        events.sort(key=lambda ev: ev.end_time)

        if print_event_trace:
            for event in events:
                print(event)

        if return_event_trace:
            return events[-1].end_time, events
        return events[-1].end_time


class Event:

    def __init__(self, type, device, start_time, end_time=None, operation=None, subtype=None):
        self.type = type
        self.device = device
        self.start_time = start_time
        self.end_time = end_time
        self.operation = operation
        self.subtype = subtype

        self.handled = False

    def __str__(self):
        if self.operation:
            if self.type == 'op_done' or self.subtype == 'op':
                op_name = self.operation[0].name
                backward = self.operation[1]
            else:
                op_name = self.operation[0][0].name
                backward = self.operation[0][1]
        else:
            op_name = None
            backward = None

        return f'[{self.type.capitalize()}{f"/{self.subtype}" if self.subtype else ""}] ' \
               f'Device: {self.device}   ' \
               f'Start time: {self.start_time}   ' \
               f'{f"End time: {self.end_time}   " if self.end_time else ""}' \
               f'{f"Operation: {op_name}   " if op_name else ""}' \
               f'{f"Backward: {backward}   " if backward is not None else ""}'
