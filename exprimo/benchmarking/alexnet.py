from collections import defaultdict

import torch
import torch.nn as nn

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url


__all__ = ['AlexNet', 'alexnet']


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class AlexNet(nn.Module):

    def __init__(self, num_classes=1000, placement=None):
        super(AlexNet, self).__init__()

        if placement is None:
            self.placement = defaultdict(lambda: 'cpu:0')
        elif isinstance(placement, str):
            self.placement = defaultdict(lambda: placement)
        else:
            self.placement = defaultdict(lambda: 'cpu:0', placement)

        device = torch.device(self.placement['conv1'])
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
        ).to(device)

        device = torch.device(self.placement['pool1'])
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2).to(device)

        device = torch.device(self.placement['conv2'])
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
        ).to(device)

        device = torch.device(self.placement['pool2'])
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2).to(device)

        device = torch.device(self.placement['conv3'])
        self.conv3 = nn.Sequential(
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        ).to(device)

        device = torch.device(self.placement['conv4'])
        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        ).to(device)

        device = torch.device(self.placement['conv5'])
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        ).to(device)

        device = torch.device(self.placement['pool5'])
        self.pool5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.AdaptiveAvgPool2d((6, 6)),
        ).to(device)

        device = torch.device(self.placement['fc6'])
        self.fc6 = nn.Sequential(
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
        ).to(device)

        device = torch.device(self.placement['dropout6'])
        self.dropout6 = nn.Dropout().to(device)

        device = torch.device(self.placement['fc7'])
        self.fc7 = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
        ).to(device)

        device = torch.device(self.placement['dropout7'])
        self.dropout7 = nn.Dropout().to(device)

        device = torch.device(self.placement['fc8'])
        self.fc8 = nn.Linear(4096, num_classes).to(device)

    def forward(self, x):
        x = x.to(torch.device(self.placement['conv1']))
        x = self.conv1(x)
        x = x.to(torch.device(self.placement['pool1']))
        x = self.pool1(x)
        x = x.to(torch.device(self.placement['conv2']))
        x = self.conv2(x)
        x = x.to(torch.device(self.placement['pool2']))
        x = self.pool2(x)
        x = x.to(torch.device(self.placement['conv3']))
        x = self.conv3(x)
        x = x.to(torch.device(self.placement['conv4']))
        x = self.conv4(x)
        x = x.to(torch.device(self.placement['conv5']))
        x = self.conv5(x)
        x = x.to(torch.device(self.placement['pool5']))
        x = self.pool5(x)
        x = torch.flatten(x, 1)
        x = x.to(torch.device(self.placement['fc6']))
        x = self.fc6(x)
        x = x.to(torch.device(self.placement['dropout6']))
        x = self.dropout6(x)
        x = x.to(torch.device(self.placement['fc7']))
        x = self.fc7(x)
        x = x.to(torch.device(self.placement['dropout7']))
        x = self.dropout7(x)
        x = x.to(torch.device(self.placement['fc8']))
        x = self.fc8(x)
        return x


def alexnet(pretrained=False, progress=True, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = AlexNet(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['alexnet'],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model