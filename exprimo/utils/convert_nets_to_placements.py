import os
import json

directory = 'experiment_results/sim_real_comp/inception/nets'
save_directory = 'experiment_results/sim_real_comp/inception/device_assignments'


def convert_to_placement(net_directory, placement_directory):
    if not os.path.exists(placement_directory):
        os.makedirs(placement_directory)

    for file in os.listdir(net_directory):
        device_assignment = {}
        if file.endswith('.json'):
            with open(os.path.join(net_directory, file)) as f:
                net_dict = json.load(f)

                for layer_name in net_dict['layers'].keys():
                    layer = net_dict['layers'][layer_name]

                    try:
                        device_assignment[layer_name] = layer['device']
                    except KeyError:
                        device_assignment[layer_name] = 0

                    if layer['type'] == 'Block':
                        for sublayer_name in layer['layers'].keys():
                            sublayer = layer['layers'][sublayer_name]
                            try:
                                device_assignment[f'{layer_name}/{sublayer_name}'] = sublayer['device']
                            except KeyError:
                                device_assignment[f'{layer_name}/{sublayer_name}'] = 0

                with open(os.path.join(placement_directory, file), 'w') as f:
                    json.dump(device_assignment, f, indent=4)


if __name__ == '__main__':
    convert_to_placement(directory, save_directory)
