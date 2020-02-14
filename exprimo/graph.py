"""
Graph representation of a Deep Neural Network.

This code is based off of corresponding code in the Paleo library. See license file in submodule.
Changes are made in order to support specific device assignment of layers.
"""

import json
from copy import deepcopy
import collections

from paleo import layers
from paleo.graph import GraphWalker


class LayerSpec:

    def __init__(self, name, params):
        self.name = name
        self.params = dict(params)
        self.operation = None
        self.parents = []
        self.inbounds = []
        self.outbounds = []

    def attach_op(self, operation):
        self.operation = operation

    def __repr__(self):
        return self.name

    def __hash__(self):
        return hash(self.name)

    def __getitem__(self, key):
        return self.params[key]

    def get(self, key, default):
        return self.params.get(key, default)


class ComputationGraph:

    def __init__(self, path=None, attach_ops=True, force_device=None):
        self.nested_list = None
        self.topological_order = None
        self.attach_ops = attach_ops
        self.force_device = force_device

        if path:
            self.load(path)

    def load(self, path):
        with open(path) as f:
            net = json.load(f)
        self._build(net)

    def load_from_string(self, string):
        net = json.loads(string)
        self._build(net)

    def _create_topological_order(self):

        def flatten(layer):
            if isinstance(layer, (tuple, list)):
                _layer_list = []

                for l in layer:
                    _layer_list.extend(flatten(l))
                return _layer_list
            return [layer]

        if self.topological_order is None:
            self.topological_order = []
            for layer in self.nested_list:
                self.topological_order.extend(flatten(layer))

    def _attach_layer_op(self):
        names_to_specs = dict()

        for layer_spec in self.topological_order:
            if len(layer_spec['parents']) == 1:
                parent_name = layer_spec['parents'][0]
                inputs = names_to_specs[parent_name].operation.outputs
            else:
                inputs = []
                try:
                    for parent_name in layer_spec['parents']:
                        inputs.append(names_to_specs[parent_name].operation.outputs)
                except KeyError:
                    raise KeyError(f'Cannot find parent {parent_name} of {layer_spec.name}')

            try:
                layer = None
                if layer_spec['type'] == 'Input':
                    layer = layers.Input(layer_spec.name, layer_spec['tensor'])
                elif layer_spec['type'] == 'Convolution':
                    layer = layers.Conv2d(
                        layer_spec.name,
                        inputs,
                        layer_spec['filter'],
                        layer_spec['strides'],
                        layer_spec['padding'],
                        backprop=('data' not in layer_spec['parents']),
                        activation_fn=layer_spec.get('activation_fn', 'relu'),
                        splits=layer_spec.get('splits', None)
                    )
                elif layer_spec['type'] == 'Deconvolution':
                    layer = layers.Deconv2D(
                        layer_spec.name,
                        inputs,
                        layer_spec['filter'],
                        layer_spec['strides'],
                        layer_spec['padding'],
                        layer_spec['output_shape'],
                        backprop=('data' not in layer_spec['parents']),
                        activation_fn=layer_spec.get('activation_fn', 'relu'))
                elif layer_spec['type'] == 'Pooling':
                    layer = layers.Pool2d(
                        layer_spec.name,
                        inputs,
                        layer_spec['ksize'],
                        layer_spec['strides'],
                        layer_spec['padding'],
                        pool_type='max')

                elif layer_spec['type'] == 'UpSampling2D':
                    layer = layers.UpSampling2D(layer_spec.name, inputs,
                                                layer_spec['ksize'])
                elif layer_spec['type'] == 'AvgPool':
                    layer = layers.Pool2d(
                        layer_spec.name,
                        inputs,
                        layer_spec['ksize'],
                        layer_spec['strides'],
                        layer_spec['padding'],
                        pool_type='avg')
                elif layer_spec['type'] == 'Dropout':
                    layer = layers.Dropout(layer_spec.name, inputs,
                                           layer_spec['dropout_keep_prob'])
                elif layer_spec['type'] == 'Concatenate':
                    layer = layers.Concatenate(layer_spec.name, inputs,
                                               layer_spec['dim'])
                elif layer_spec['type'] == 'Reshape':
                    layer = layers.Reshape(layer_spec.name, inputs,
                                           layer_spec['output_shape'])
                elif layer_spec['type'] == 'Elementwise':
                    layer = layers.Elementwise(layer_spec.name, inputs)
                elif layer_spec['type'] == 'Softmax':
                    layer = layers.Softmax(layer_spec.name, inputs,
                                           layer_spec.get('num_classes', None))
                elif layer_spec['type'] == 'Sigmoid':
                    layer = layers.Sigmoid(layer_spec.name, inputs)
                elif layer_spec['type'] == 'InnerProduct':
                    layer = layers.InnerProduct(layer_spec.name, inputs,
                                                layer_spec['num_outputs'])
                else:
                    layer = layers.Generic(layer_spec.name, inputs,
                                           layer_spec['type'])
            except Exception as e:
                raise e

            if layer:
                layer_spec.parents.extend([names_to_specs[p] for p in layer_spec['parents']])
                layer.parents = layer_spec['parents']
                layer_spec.attach_op(layer)
                names_to_specs[layer_spec.name] = layer_spec

    def _build(self, net):
        names_to_specs = dict()
        block_endpoints = dict()

        def _parents(parents):
            # Replace with endpoint if parent is a block.
            transformed_parents = []
            for parent_name in parents:
                transformed_parents.append(block_endpoints.get(parent_name, parent_name))
            return transformed_parents

        sharded_layers = {}

        def _shard(layer_spec, endpoint_block=None):
            devices = layer_spec.params['device']
            assert isinstance(devices, collections.Sequence), 'devices must be a Sequence for sharding to be allowed!'

            dim_vector_name = None
            if layer_spec.params['type'] == 'Convolution':
                dim_vector_name = 'filter'
            elif layer_spec.params['type'] == 'Pooling':
                dim_vector_name = 'ksize'

            channel_sizes = [layer_spec.params[dim_vector_name][-1] // len(devices)] * len(devices)
            i = 0
            while sum(channel_sizes) < layer_spec.params[dim_vector_name][-1]:
                channel_sizes[i] += 1
                i += 1

            shard_names = []
            for i, device in enumerate(devices):
                shard_params = deepcopy(layer_spec.params)
                shard_name = f'{layer_spec.name}_shard{i}'
                shard_params['device'] = device
                shard_params[dim_vector_name][-1] = channel_sizes[i]
                shard_names.append(shard_name)
                shard_spec = LayerSpec(shard_name, shard_params)
                assert shard_name not in names_to_specs, f'Duplicate {shard_name}.'
                names_to_specs[shard_name] = shard_spec

            sharded_layers[layer_name] = shard_names
            if endpoint_block:
                sharded_layers[endpoint_block] = shard_names

        # Transform all specs into LayerSpec objects
        for layer_name, layer_params in net['layers'].items():
            if layer_params.get('type', None) in ['Block']:
                block_name = layer_name
                block_parents = _parents(layer_params['parents'])

                # If block provides an endpoint, subsequent layers can refer to the block name as parent.
                if 'endpoint' in layer_params:
                    block_endpoints[block_name] = f'{block_name}/{layer_params["endpoint"]}'

                for sublayer_name, sublayer_params in layer_params['layers'].items():
                    is_endpoint = 'endpoint' in layer_params and layer_params['endpoint'] == sublayer_name

                    sublayer_name = f'{block_name}/{sublayer_name}'

                    if 'device' not in sublayer_params and 'device' in layer_params:
                        sublayer_params['device'] = layer_params['device']

                    sublayer = LayerSpec(sublayer_name, sublayer_params)

                    # Update parents
                    if len(sublayer_params['parents']) == 0:
                        # Use the parent of the block
                        sublayer_parents = block_parents
                    else:
                        # Add blockname to the parent names
                        sublayer_parents = [f'{block_name}/{n}' for n in sublayer_params['parents']]
                        sublayer_parents = _parents(sublayer_parents)

                    sublayer.params['parents'] = sublayer_parents

                    if 'device' in sublayer.params and isinstance(sublayer.params['device'], collections.Sequence):
                        endpoint_block = layer_name if is_endpoint else None
                        _shard(sublayer, endpoint_block)
                    else:
                        assert sublayer_name not in names_to_specs, f'Duplicate {sublayer_name}.'
                        names_to_specs[sublayer_name] = sublayer

            else:
                layer_params['parents'] = _parents(layer_params['parents'])
                layer = LayerSpec(layer_name, layer_params)

                if 'device' in layer.params and isinstance(layer.params['device'], collections.Sequence):
                    _shard(layer)
                else:
                    assert layer_name not in names_to_specs, f'Duplicate {layer_name}'
                    names_to_specs[layer_name] = layer

        # Update parents list for children of sharded layers
        for layer_name, layer_spec in names_to_specs.copy().items():
            new_parents = []
            for parent in layer_spec['parents']:
                if parent in sharded_layers:
                    conc_layer_params = {
                        'type': 'Concatenate',
                        'parents': sharded_layers[parent],
                        'device': layer_spec.params['device'],
                        'dim': 3
                    }
                    conc_layer_name = f'{layer_name}_conc_{parent}'
                    conc_layer_spec = LayerSpec(conc_layer_name, conc_layer_params)
                    names_to_specs[conc_layer_name] = conc_layer_spec

                    new_parents.append(conc_layer_name)
                else:
                    new_parents.append(parent)
            layer_spec.params['parents'] = new_parents

        # Add edges
        for layer_name, layer_spec in names_to_specs.items():
            for parent_name in _parents(layer_spec['parents']):
                assert parent_name in names_to_specs, f'Parent layer {parent_name} of {layer_name} ' \
                                                      f'does not have a LayerSpec object.'

                names_to_specs[parent_name].outbounds.append(layer_spec)
                layer_spec.inbounds.append(names_to_specs[parent_name])

        # Set default devices for any layer with unspecified device
        for layer_name, layer_spec in names_to_specs.items():
            if 'device' not in layer_spec.params:
                layer_spec.params['device'] = 0
            if self.force_device is not None:
                layer_spec.params['device'] = self.force_device

        graph_walker = GraphWalker(names_to_specs)
        self.nested_list = graph_walker.start(names_to_specs['data'])
        self._create_topological_order()
        if self.attach_ops:
            self._attach_layer_op()


def get_flattened_layer_names(net_string):
    graph = ComputationGraph()
    graph.load_from_string(net_string)

    return graph.topological_order
