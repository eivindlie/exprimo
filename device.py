from enum import Enum


class DeviceType(Enum):
    GPU = 'gpu'
    CPU = 'cpu'


class CommunicationType(Enum):
    ETHERNET = 'ethernet'
    INFINIBAND = 'infiniband'
    PCIE3 = 'pcie3'
    SLI = 'sli'


class Device:

    def __init__(self, model, clock, peak_gflops, memory, mem_bandwidth=68, type=DeviceType.CPU, id=None):
        """
        Creates a new device.
        :param model: Model string for the device. E.g. 'V100' or 'Titan X'.
        :param clock: Clock speed of the device, in MHz.
        :param peak_gflops: Peak GFLOPS (Floating Operations Per Second) of the device
        :param memory: Memory available to the device, in GB.
        :param mem_bandwidth: The bandwidth available to the device when reading from device memory, in GB/s.
        :param type: Device type; available types are given in the DeviceType enum. (CPU or GPU)
        :param id: The hardware ID of the device if available on the local computer.
        """

        self.model = model
        self.clock = clock
        self.peak_gflops = peak_gflops
        self.memory = memory
        self.mem_bandwidth = mem_bandwidth
        self.type = type
        self.id = id


class CommunicationChannel:

    def __init__(self, type, bandwidth):
        """
        Creates a new communication channel.
        :param type:    The type of the channel; available types are given in the CommunicationType enum.
                        E.g. ethernet or Infiniband.
        :param bandwidth: The bandwidth of the channel, given in Gb/s.
        """
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