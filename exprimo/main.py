from graph import ComputationGraph
from device import DeviceGraph
from simulator import Simulator

if __name__ == '__main__':
    graph = ComputationGraph('../nets/resnet50.json')
    device_graph = DeviceGraph.load_from_file('../device_graphs/cluster1.json')
    simulator = Simulator(graph, device_graph)
    run_time = simulator.simulate(batch_size=128, batches=2)

    print()
    print(f'Total batch run time: {run_time:.2f}ms')
