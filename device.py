from enum import Enum
import json

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

    @staticmethod
    def load_from_file(path):
        device_graph = DeviceGraph()
        with open(path) as f:
            graph = json.loads(f.read())

            # First, we load all devices
            for device in graph['devices']:
                args = [device['model'], device['clock'], device['peak_gflops'], device['memory']]
                kwargs = {}

                for kw in ('mem_bandwidth', 'type', 'id'):
                    if kw in device:
                        kwargs[kw] = device[kw]

                d = Device(*args, **kwargs)
                device_graph.devices.append(DeviceNode(d))

            # Then, we resolve neighbours
            for i, device in enumerate(device_graph.devices):
                json_device = graph['devices'][i]

                for neighbour in json_device['neighbours']:
                    json_comm_channel = graph['comm_types'][neighbour['comm_channel']]
                    comm_channel = CommunicationChannel(json_comm_channel['type'], json_comm_channel['bandwidth'])

                    device.add_neighbour(device_graph.devices[neighbour['device']], comm_channel)
        return device_graph
