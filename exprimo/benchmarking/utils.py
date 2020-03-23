import torch

from exprimo.benchmarking.resnet import resnet50

DEVICE_MAP = {
    0: 'cpu',
    1: 'cpu',
    2: 'cuda:0',
    3: 'cuda:1'
}


def load_model_with_placement(model_type, placement, lr=0.01, classes=1000, device_map=None):
    if device_map is None:
        device_map = DEVICE_MAP

    if model_type.lower() in ['resnet', 'resnet50', 'resnet-50']:
        model_type = 'resnet50'
    elif model_type.lower() in ['inception', 'inception_v3', 'inceptionv3']:
        model_type = 'inception'

    if placement is None:
        placement = 'cpu'
    elif isinstance(placement, dict):
        translated_placement = {}
        for layer_name, device in placement.items():
            translated_placement[layer_name] = device_map[device]
        placement = translated_placement

    if model_type == 'resnet50':
        model = resnet50(pretrained=False, placement=placement, num_classes=classes)
        first_layer = 'conv1'
        last_layer = 'fc1000'
    elif model_type == 'inception':
        first_layer = 'Conv2d_1a_3x3'
        last_layer = 'softmax'
        # model = inception_v3(pretrained=False, placement=placement, num_classes=classes, init_weights=False)

    if isinstance(placement, str):
        input_device = output_device = torch.device(placement)
        model.to(input_device)
    else:
        input_device = placement[first_layer]
        output_device = placement[last_layer]

    criterion = torch.nn.CrossEntropyLoss().to(output_device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    return model, criterion, optimizer, input_device, output_device
