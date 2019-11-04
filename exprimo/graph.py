"""
Graph representation of a Deep Neural Network.

This code is based off of corresponding code in the Paleo library. See license file in submodule.
Changes are made in order to support specific device assignment of layers.
"""

import json
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

    def __init__(self, path=None, attach_ops=True):
        self.nested_list = None
        self.topological_order = None
        self.attach_ops = attach_ops

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
                exit()

            if layer:
                layer_spec.parents.extend([names_to_specs[p] for p in layer_spec['parents']])
                layer.parents = layer_spec['parents']
                layer_spec.attach_op(layer)
                names_to_specs[layer_spec.name] = layer_spec

    def _build(self, net):
        names_to_specs = dict()
        block_endpoints = dict()

        layernames_to_splits = dict()

        def _parents(parents, current_split=None):
            # Replace with endpoint if parent is a block.
            transformed_parents = []

            for parent_name in parents:
                # Pointing to a specific split is not supported.
                # Will be replaced by explicit device placement.
                if '@all' in parent_name:
                    parent_name = parent_name.replace('@all', '')
                    splits = layernames_to_splits[parent_name]
                    for s in range(splits):
                        transformed_parents.append(block_endpoints.get(parent_name, parent_name) + f'@{s}')
                elif '@self' in parent_name:
                    parent_name = parent_name.replace('@self', '')
                    assert parent_name in layernames_to_splits, f'Parent {parent_name} is not split.'
                    transformed_parents.append(block_endpoints.get(parent_name, parent_name) + f'@{current_split}')
                else:
                    transformed_parents.append(block_endpoints.get(parent_name, parent_name))
            return transformed_parents

        # Counting splits.
        # TODO Can probably consider removing splits altogether.
        for layer_name, layer_params in net['layers'].items():
            if layer_params.get('type', None) == 'ModelParallel':
                block_name = layer_name
                num_splits = layer_params.get('splits', 1)
                for sublayer_name in layer_params['layers']:
                    layernames_to_splits[f'{block_name}/{sublayer_name}'] = num_splits

        # Transform all specs into LayerSpec objects
        for layer_name, layer_params in net['layers'].items():
            if layer_params.get('type', None) in ['Block', 'ModelParallel']:
                is_model_parallel = (layer_params['type'] == 'ModelParallel')
                block_name = layer_name
                block_parents = _parents(layer_params['parents'])

                # For model parallel, the specified layers are repeated
                num_splits = layer_params.get('splits', 1)

                for s in range(num_splits):
                    for sublayer_name, sublayer_params in layer_params['layers'].items():
                        sublayer_name = f'{block_name}/{sublayer_name}'

                        if is_model_parallel:
                            sublayer_name = f'{sublayer_name}@{s}'
                            sublayer_params['splits'] = num_splits

                        sublayer = LayerSpec(sublayer_name, sublayer_params)

                        # Update parents
                        if len(sublayer_params['parents']) == 0:
                            # Use the parent of the block
                            sublayer_parents = block_parents
                        else:
                            # Add blockname to the parent names
                            sublayer_parents = [f'{block_name}/{n}' for n in sublayer_params['parents']]
                            sublayer_parents = _parents(sublayer_parents, s)

                        assert sublayer_name not in names_to_specs, f'Duplicate {sublayer_name}.'
                        names_to_specs[sublayer_name] = sublayer

                # If block provides an endpoint, subsequent layers can refer to the block name as parent.
                if 'endpoint' in layer_params:
                    block_endpoints[block_name] = f'{block_name}/{layer_params["endpoint"]}'

            else:
                layer_params['parents'] = _parents(layer_params['parents'])
                layer = LayerSpec(layer_name, layer_params)
                assert layer_name not in names_to_specs, f'Duplicate {layer_name}'
                names_to_specs[layer_name] = layer

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

        graph_walker = GraphWalker(names_to_specs)
        self.nested_list = graph_walker.start(names_to_specs['data'])
        self._create_topological_order()
        if self.attach_ops:
            self._attach_layer_op()