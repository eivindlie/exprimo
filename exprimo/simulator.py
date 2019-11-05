"""
Execution simulator based on the technique described in Placeto (Addanki et al., 2019, https://arxiv.org/abs/1906.08879)
"""

from collections import deque, defaultdict

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

    def simulate(self, print_event_trace=True, include_backward=True, return_event_trace=False, batch_size=None,
                 batches=1):
        op_queues = [deque() for device in self.device_graph.devices]
        transfer_queues = [deque() for comm_channel in self.device_graph.comm_channels]
        comm_free = [True for i in range(len(self.device_graph.comm_channels))]
        device_free = [True for i in range(len(self.device_graph.devices))]
        forward_done = [defaultdict(lambda: False) for b in range(batches)]
        backward_done = [defaultdict(lambda: False) for b in range(batches)]
        events = []

        for b in range(batches):
            for layer in self.computation_graph.topological_order:
                if not len(layer.inbounds):
                    op_queues[layer['device']].append((layer, False, b))
                    if device_free[layer['device']]:
                        device_free[layer['device']] = False
                        events.append(Event('wakeup', layer['device'], 0, subtype='op', batch=b))

        def run_op(op, backward, start_time, batch=0):
            device = self.device_graph.devices[op['device']].device
            run_time = FlopsProfiler.profile(op, device, backward, batch_size)
            end_time = start_time + run_time
            events.append(Event('op_done', op['device'], start_time,
                                end_time=end_time, operation=(op, backward, batch), batch=batch))

        def run_transfer(op, backward, comm_channel_id, target, start_time, batch=0):
            parent_device = self.device_graph.devices[op['device']].device
            comm_channel = self.device_graph.comm_channels[comm_channel_id]
            transfer_time = TransferProfiler.profile(op, comm_channel, parent_device, backward, batch_size)
            end_time = start_time + transfer_time
            events.append(Event('transfer_done', comm_channel_id, start_time,
                                operation=((op, backward, batch), op['device'], target),
                                end_time=end_time, batch=batch))

        def can_run(op, backward, batch):
            parents = op.outbounds if backward else op.inbounds
            for parent in parents:
                if (backward and not backward_done[batch][parent]) or \
                        (not backward and not forward_done[batch][parent]):
                    return False
            return True

        def op_done(event):
            op, backward, batch = event.operation
            if backward:
                backward_done[batch][op] = True
                children = op.inbounds
            else:
                forward_done[batch][op] = True
                children = op.outbounds

            for child in children:
                if child['device'] != op['device']:
                    op_device = self.device_graph.devices[op['device']]
                    child_device = self.device_graph.devices[child['device']]
                    comm_channel = op_device.neighbours[child_device]

                    if backward:
                        # If we are doing the backward pass, we are transferring the gradients, which are
                        # equal in size to the previous layer. We therefore set the op as child instead of op.
                        transfer_queues[comm_channel.id].append(((child, backward, batch), op_device, child_device))
                    else:
                        transfer_queues[comm_channel.id].append(((op, backward, batch), op_device, child_device))
                    if comm_free[comm_channel.id]:
                        comm_free[comm_channel.id] = False
                        events.append(Event('wakeup', comm_channel.id, event.end_time, subtype='transfer',
                                            batch=batch))
                else:
                    if can_run(child, backward, batch):
                        op_queues[child['device']].append((child, backward, batch))
                        # Don't need to check if device is free, as it is the current device

            if not backward and not len(children) and include_backward:
                op_queues[op['device']].append((op, True, batch))

            if len(op_queues[event.device]):
                op2, backward2, batch2 = op_queues[event.device].popleft()
                run_op(op2, backward2, event.end_time, batch=batch2)
            else:
                device_free[event.device] = True

        def transfer_done(event):
            (op, backward, batch), start_device, end_device = event.operation
            children = op.inbounds if backward else op.outbounds

            for child in children:
                if can_run(child, backward, batch):
                    op_queues[child['device']].append((child, backward, batch))
                    if device_free[child['device']]:
                        device_free[child['device']] = False
                        events.append(Event('wakeup', child['device'], event.end_time, subtype='op', batch=batch))

            if len(transfer_queues[event.device]):
                (op2, backward2, batch2), start_device2, end_device2 = transfer_queues[event.device].popleft()
                run_transfer(op2, backward2, event.device, end_device2, event.end_time, batch=batch2)
            else:
                comm_free[event.device] = True

        def wakeup(event):
            event.end_time = event.start_time
            if event.subtype == 'transfer':
                (op, backward, batch), start_device, end_device = transfer_queues[event.device].popleft()
                run_transfer(op, backward, event.device, end_device, event.start_time, batch=batch)
            else:
                (op, backward, batch) = op_queues[event.device].popleft()
                run_op(op, backward, event.start_time, batch=batch)

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

    def __init__(self, type, device, start_time, end_time=None, operation=None, subtype=None, batch=0):
        self.type = type
        self.device = device
        self.start_time = start_time
        self.end_time = end_time
        self.operation = operation
        self.subtype = subtype
        self.batch = batch

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
               f'Batch: {self.batch}   ' \
               f'Start time: {self.start_time}   ' \
               f'{f"End time: {self.end_time}   " if self.end_time else ""}' \
               f'{f"Operation: {op_name}   " if op_name else ""}' \
               f'{f"Backward: {backward}   " if backward is not None else ""}'
