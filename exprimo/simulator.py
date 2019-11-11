"""
Execution simulator based on the technique described in Placeto (Addanki et al., 2019, https://arxiv.org/abs/1906.08879)
"""

import heapq
from collections import deque, defaultdict

import numpy as np

from device import DeviceGraph
from graph import ComputationGraph
from exprimo.profilers import FlopsProfiler
from exprimo.profilers import TransferProfiler


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

    def simulate(self, batch_size=None, batches=1,
                 print_event_trace=True, include_backward=True, return_event_trace=False, print_memory_usage=True):
        op_queues = [deque() for device in self.device_graph.devices]
        transfer_queues = [deque() for comm_channel in self.device_graph.comm_channels]
        comm_free = [True for i in range(len(self.device_graph.comm_channels))]
        device_free = [True for i in range(len(self.device_graph.devices))]
        forward_done = [defaultdict(lambda: False) for b in range(batches)]
        backward_done = [defaultdict(lambda: False) for b in range(batches)]
        event_queue = MinHeap()
        events = []

        for b in range(batches):
            for layer in self.computation_graph.topological_order:
                if not len(layer.inbounds):
                    op_queues[layer['device']].append((layer, False, b))
                    if device_free[layer['device']]:
                        device_free[layer['device']] = False
                        event_queue.push(Event('wakeup', layer['device'], 0, subtype='op', batch=b))

        def run_op(op, backward, start_time, batch=0):
            device = self.device_graph.devices[op['device']].device
            run_time = FlopsProfiler.profile(op, device, backward, batch_size)
            end_time = start_time + run_time
            event_queue.push(Event('op_done', op['device'], start_time,
                                   end_time=end_time, operation=(op, backward, batch), batch=batch))

        def run_transfer(op, backward, comm_channel_id, target_ops, start_time, batch=0):
            parent_device = self.device_graph.devices[op['device']].device
            comm_channel = self.device_graph.comm_channels[comm_channel_id]

            # If we are running the backward step, we are transferring gradients, which are the same size as the
            # output of the target op
            transferred_ops = target_ops if backward else [op]

            transfer_time = 0
            for transferred_op in transferred_ops:
                transfer_time += TransferProfiler.profile(transferred_op, comm_channel, parent_device,
                                                          backward, batch_size)
            end_time = start_time + transfer_time
            event_queue.push(Event('transfer_done', comm_channel_id, start_time,
                                   operation=((op, backward, batch), target_ops),
                                   end_time=end_time, batch=batch,
                                   from_device=op['device'], to_device=target_ops[0]['device']))

        def can_run(op, backward, batch):
            # TODO Need to somehow check that all tensors are available on the device?
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

            transfers = []
            for child in children:
                if child['device'] != op['device']:
                    op_device = self.device_graph.devices[op['device']]
                    child_device = self.device_graph.devices[child['device']]
                    comm_channel = op_device.neighbours[child_device]

                    # See if a transfer to this device is scheduled already
                    transfer = next((t for t in transfers if t[1][0]['device'] == child['device']), None)
                    if transfer:
                        transfer[1].append(child)
                    else:
                        # Transfer format: (transferred_op, target_ops)
                        transfers.append(((op, backward, batch), [child]))
                    if comm_free[comm_channel.id]:
                        comm_free[comm_channel.id] = False
                        event_queue.push(Event('wakeup', comm_channel.id, event.end_time, subtype='transfer',
                                               batch=batch))
                else:
                    if can_run(child, backward, batch):
                        op_queues[child['device']].append((child, backward, batch))
                        # Don't need to check if device is free, as it is the current device

            for transfer in transfers:
                child = transfer[1][0]
                op_device = self.device_graph.devices[op['device']]
                child_device = self.device_graph.devices[child['device']]
                comm_channel = op_device.neighbours[child_device]

                transfer_queues[comm_channel.id].append(transfer)

            if not backward and not len(children) and include_backward:
                op_queues[op['device']].append((op, True, batch))

            if len(op_queues[event.device]):
                op2, backward2, batch2 = op_queues[event.device].popleft()
                run_op(op2, backward2, event.end_time, batch=batch2)
            else:
                device_free[event.device] = True

        def transfer_done(event):
            (op, backward, batch), target_ops = event.operation
            children = op.inbounds if backward else op.outbounds

            for child in children:
                if can_run(child, backward, batch):
                    op_queues[child['device']].append((child, backward, batch))
                    if device_free[child['device']]:
                        device_free[child['device']] = False
                        event_queue.push(Event('wakeup', child['device'], event.end_time, subtype='op', batch=batch))

            if len(transfer_queues[event.device]):
                (op2, backward2, batch2), target_ops2 = transfer_queues[event.device].popleft()
                run_transfer(op2, backward2, event.device, target_ops2, event.end_time, batch=batch2)
            else:
                comm_free[event.device] = True

        def wakeup(event):
            if event.subtype == 'transfer':
                (op, backward, batch), target_ops = transfer_queues[event.device].popleft()
                run_transfer(op, backward, event.device, target_ops, event.start_time, batch=batch)
            else:
                (op, backward, batch) = op_queues[event.device].popleft()
                run_op(op, backward, event.start_time, batch=batch)

        event_map = {
            'op_done': op_done,
            'transfer_done': transfer_done,
            'wakeup': wakeup
        }

        while not event_queue.empty():
            event = event_queue.pop()
            events.append(event)
            event.handled = True
            event_map[event.type](event)

        if print_event_trace:
            for event in events:
                print(event)

        peak_memory_usage = self.calculate_peak_memory_usage(events)

        if print_memory_usage:
            print('\n')
            mem_strings = []
            print('Device:'.rjust(18), end='')
            for i, device in enumerate(self.device_graph.devices):
                mem_string = f'{round(peak_memory_usage[i] / 2**20, 2)} MiB'
                col_width = max(len(mem_string), len(device.name)) + 5
                mem_string = mem_string.rjust(col_width)
                mem_strings.append(mem_string)

                print(device.name.rjust(col_width), end='')

            print()
            print('Peak memory usage:'.rjust(18), end='')
            print(''.join(mem_strings))

        if return_event_trace:
            return events[-1].end_time, events
        return events[-1].end_time

    def calculate_peak_memory_usage(self, events):
        def calculate_tensor_size(shape, dtype='float32'):
            return np.prod(shape) * np.dtype(dtype).itemsize

        devices = self.device_graph.devices

        memory_usage = np.zeros(len(devices))

        for op in self.computation_graph.topological_order:
            memory_usage[op['device']] += op.operation.weights_in_bytes

        peak_memory_usage = np.copy(memory_usage)

        # We keep a record of what tensors are being saved on which devices (used for both consecutive layers
        # and backpropagation)
        saved_tensors = [[] for i in range(len(devices))]

        for event in events:
            if event.type == 'op_done':
                op = event.operation[0]
                children = op.inbounds if event.backward else op.outbounds
                parents = op.outbounds if event.backward else op.inbounds

                for child in children:
                    saved_op = (op, child) if event.backward else op
                    saved_tensor = next((i for i in saved_tensors[op['device']]
                                         if i[:3] == [saved_op, event.batch, event.backward]), None)
                    if saved_tensor:
                        saved_tensor[3] += 1
                    else:
                        saved_tensors[op['device']].append([saved_op, event.batch, event.backward, 1])
                        saved_op_shape = (saved_op[1] if event.backward else saved_op).operation.outputs
                        tensor_size = calculate_tensor_size(saved_op_shape)
                        memory_usage[op['device']] += tensor_size

                for parent in parents:
                    saved_op = (parent, op) if event.backward else parent
                    saved_tensor = next((i for i in saved_tensors[op['device']]
                                         if i[:3] == [saved_op, event.batch, event.backward]), None)
                    assert saved_tensor, 'All required tensors must be available before operation execution!'

                    # If we are doing the backward pass, this is the last time we need the inputs.
                    if event.backward:
                        saved_tensor[3] -= 1
                        if saved_tensor[3] == 0:
                            saved_op_shape = (saved_tensor[0][1] if event.backward
                                              else saved_tensor[0]).operation.outputs
                            saved_tensors[op['device']].remove(saved_tensor)
                            memory_usage[op['device']] -= calculate_tensor_size(saved_op_shape)

            elif event.type == 'transfer_done':
                transferred_op, target_ops = event.operation[0][0], event.operation[1]
                saved_op = (transferred_op, target_ops[0]) if event.backward else transferred_op

                saved_tensor = next((i for i in saved_tensors[transferred_op['device']]
                                     if i[:3] == [saved_op, event.batch, event.backward]), None)

                saved_tensor[3] -= 1
                if saved_tensor[3] == 0:
                    saved_op_shape = (saved_tensor[0][1] if event.backward
                                      else saved_tensor[0]).operation.outputs
                    saved_tensors[transferred_op['device']].remove(saved_tensor)
                    memory_usage[transferred_op['device']] -= calculate_tensor_size(saved_op_shape)

                for child in target_ops:
                    saved_tensor = next((i for i in saved_tensors[child['device']]
                                         if i[:3] == [saved_op, event.batch, event.backward]), None)

                    if saved_tensor:
                        saved_tensor[3] += 1
                    else:
                        saved_tensors[child['device']].append([saved_op, event.batch, event.backward, 1])
                        saved_op_shape = (saved_op[1] if event.backward else saved_op).operation.outputs
                        tensor_size = calculate_tensor_size(saved_op_shape)
                        memory_usage[child['device']] += tensor_size

            # Check if any devices now use more memory than previous recorded peak
            peak_memory_usage = np.maximum(peak_memory_usage, memory_usage)

        return tuple(peak_memory_usage)


class MinHeap:
    def __init__(self, initial=None):
        if initial:
            self._data = initial[:]
            heapq.heapify(self._data)
        else:
            self._data = []

    def push(self, item):
        heapq.heappush(self._data, item)

    def pop(self):
        return heapq.heappop(self._data)

    def empty(self):
        return len(self._data) == 0


class Event:

    def __init__(self, event_type, device, start_time, end_time=None, operation=None, subtype=None, batch=0,
                 from_device=None, to_device=None):
        self.type = event_type
        self.device = device
        self.start_time = start_time
        if event_type == 'wakeup':
            self.end_time = start_time
        else:
            self.end_time = end_time
        self.operation = operation
        self.subtype = subtype
        self.batch = batch
        self.from_device = from_device
        self.to_device = to_device

        self.handled = False

    @property
    def backward(self):
        if self.operation:
            if self.type == 'op_done' or self.subtype == 'op':
                return self.operation[1]
            else:
                return self.operation[0][1]
        return None

    @property
    def op_name(self):
        if self.operation:
            if self.type == 'op_done' or self.subtype == 'op':
                return self.operation[0].name
            else:
                return self.operation[0][0].name
        return None

    def __str__(self):
        return f'[{self.type.capitalize()}{f"/{self.subtype}" if self.subtype else ""}] ' \
               f'Device: {self.device}   ' \
               f'Batch: {self.batch}   ' \
               f'Start time: {self.start_time}   ' \
               f'{f"End time: {self.end_time}   " if self.end_time else ""}' \
               f'{f"Operation: {self.op_name}   " if self.op_name else ""}' \
               f'{f"From device: {self.from_device}   " if self.from_device else ""}' \
               f'{f"To device: {self.to_device}   " if self.to_device else ""}' \
               f'{f"Backward: {self.backward}   " if self.backward is not None else ""}'

    def __gt__(self, other):
        self.end_time > other.end_time
