from enum import Enum


class DeviceType(Enum):
    GPU = 'gpu'
    CPU = 'cpu'


class CommunicationType(Enum):
    ETHERNET = 'ethernet'
    INFINIBAND = 'infiniband'
    PCIE3 = 'pcie3'


class Device:

    def __init__(self, model, clock, peak_gflop, type=DeviceType.CPU):
        self.model = model
        self.clock = clock
        self.peak_gflop = peak_gflop
        self.type = type


class CommunicationChannel:

    def __init__(self, type, bandwidth):
        self.type = type
        self.bandwidth = bandwidth


class DeviceNode:

    def __init__(self, device):
        self.device = device
        self.neighbours = {}

    def add_neighbour(self, device_node, comm_channel):
        self.neighbours[comm_channel] = device_node


class DeviceGraph:

    def __init__(self):
        self.devices = []

    def load_from_file(self, path):
        with open(path) as f:
            pass