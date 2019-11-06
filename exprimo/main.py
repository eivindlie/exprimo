from graph import ComputationGraph
from device import DeviceGraph
from simulator import Simulator
from plotting import plot_event_trace

if __name__ == '__main__':
    graph = ComputationGraph('../nets/resnet50.json')
    device_graph = DeviceGraph.load_from_file('../device_graphs/cluster2.json')
    simulator = Simulator(graph, device_graph)
    run_time, events = simulator.simulate(batch_size=128, batches=2, return_event_trace=True)

    plot_event_trace(events)

    print()
    print(f'Total batch run time: {run_time:.2f}ms')
