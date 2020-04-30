import os
import json

from exprimo import ComputationGraph, DeviceGraph, Simulator

directory = '~/logs/experiment_results/sim_real_comp/resnet/nets'
output_file = '~/logs/experiment_results/sim_real_comp/resnet/2020-04-30_scores.csv'

device_graph_path = 'device_graphs/malvik.json'

comp_penalization = 0.9
comm_penalization = 0.25

device_assignment = {}

directory = os.path.expanduser(directory)
output_file = os.path.expanduser(output_file)

with open(output_file, 'w') as f:
    f.write('')

for file in os.listdir(directory):
    if file.endswith('.json'):
        print(f'Processing file {file}... ', end='')
        with open(os.path.join(directory, file)) as f:
            # net_dict = json.load(f)
            net_string = f.read()

        # net_dict = json.loads(net_string)
        # for layer_name, layer in net_dict['layers'].items():
        #     if layer['device'] != 0:
        #         layer['device'] -= 1
        # net_string = json.dumps(net_dict)

        graph = ComputationGraph()
        graph.load_from_string(net_string)
        device_graph = DeviceGraph.load_from_file(device_graph_path)

        simulator = Simulator(graph, device_graph)
        run_time = simulator.simulate(batch_size=128, batches=1, pipeline_batches=1, print_event_trace=False,
                                      print_memory_usage=False, comp_penalization=comp_penalization,
                                      comm_penalization=comm_penalization)

        with open(output_file, 'a') as f:
            f.write(f'{file.replace("gen_", "").replace(".json", "")}, {run_time}\n')

        print(f'{run_time:.2f}ms')
