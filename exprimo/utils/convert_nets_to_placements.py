import os
import json

directory = 'experiment_results/sim_real_comp/nets'
save_directory = 'experiment_results/sim_real_comp/device_assignments'

device_assignment = {}

for file in os.listdir(directory):
    if file.endswith('.json'):
        with open(os.path.join(directory, file)) as f:
            net_dict = json.load(f)

            for layer_name in net_dict['layers'].keys():
                layer = net_dict['layers'][layer_name]

                device_assignment[layer_name] = layer['device']

                if layer['type'] == 'Block':
                    for sublayer_name in layer['layers'].keys():
                        sublayer = layer['layers'][sublayer_name]
                        device_assignment[f'{layer_name}/{sublayer_name}'] = layer['device']

            with open(os.path.join(save_directory, file), 'w') as f:
                json.dump(device_assignment, f, indent=4)
