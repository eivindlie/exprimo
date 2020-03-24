from __future__ import division

from collections import namedtuple, defaultdict
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.jit.annotations import Optional
from torch import Tensor

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

__all__ = ['Inception3', 'inception_v3', 'InceptionOutputs', '_InceptionOutputs']


model_urls = {
    # Inception v3 ported from TensorFlow
    'inception_v3_google': 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth',
}

InceptionOutputs = namedtuple('InceptionOutputs', ['logits', 'aux_logits'])
InceptionOutputs.__annotations__ = {'logits': torch.Tensor, 'aux_logits': Optional[torch.Tensor]}

# Script annotations failed with _GoogleNetOutputs = namedtuple ...
# _InceptionOutputs set here for backwards compat
_InceptionOutputs = InceptionOutputs


def inception_v3(pretrained=False, progress=True, **kwargs):
    r"""Inception v3 model architecture from
    `"Rethinking the Inception Architecture for Computer Vision" <http://arxiv.org/abs/1512.00567>`_.

    .. note::
        **Important**: In contrast to the other models the inception_v3 expects tensors with a size of
        N x 3 x 299 x 299, so ensure your images are sized accordingly.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        aux_logits (bool): If True, add an auxiliary branch that can improve training.
            Default: *True*
        transform_input (bool): If True, preprocesses the input according to the method with which it
            was trained on ImageNet. Default: *False*
    """
    if pretrained:
        if 'transform_input' not in kwargs:
            kwargs['transform_input'] = True
        if 'aux_logits' in kwargs:
            original_aux_logits = kwargs['aux_logits']
            kwargs['aux_logits'] = True
        else:
            original_aux_logits = True
        model = Inception3(**kwargs)
        state_dict = load_state_dict_from_url(model_urls['inception_v3_google'],
                                              progress=progress)
        model.load_state_dict(state_dict)
        if not original_aux_logits:
            model.aux_logits = False
            del model.AuxLogits
        return model

    return Inception3(**kwargs)


class Inception3(nn.Module):

    def __init__(self, num_classes=1000, aux_logits=True, transform_input=False,
                 inception_blocks=None, init_weights=True, placement=None):
        super(Inception3, self).__init__()
        if inception_blocks is None:
            inception_blocks = [
                BasicConv2d, InceptionA, InceptionB, InceptionC,
                InceptionD, InceptionE, InceptionAux
            ]
        assert len(inception_blocks) == 7
        conv_block = inception_blocks[0]
        inception_a = inception_blocks[1]
        inception_b = inception_blocks[2]
        inception_c = inception_blocks[3]
        inception_d = inception_blocks[4]
        inception_e = inception_blocks[5]
        inception_aux = inception_blocks[6]

        if placement is None:
            placement = defaultdict(lambda: 'cpu:0')
        elif isinstance(placement, str):
            old_placement = placement
            placement = defaultdict(lambda: old_placement)
        else:
            placement = defaultdict(lambda: 'cpu:0', placement)

        self.placement = placement

        self.aux_logits = aux_logits
        self.transform_input = transform_input
        self.Conv2d_1a_3x3 = conv_block(3, 32, kernel_size=3, stride=2).to(self.get_device('Conv2d_1a_3x3'))
        self.Conv2d_2a_3x3 = conv_block(32, 32, kernel_size=3).to(self.get_device('Conv2d_2a_3x3'))
        self.Conv2d_2b_3x3 = conv_block(32, 64, kernel_size=3, padding=1).to(self.get_device('Conv2d_2b_3x3'))
        self.Conv2d_3b_1x1 = conv_block(64, 80, kernel_size=1).to(self.get_device('Conv2d_3b_1x1'))
        self.Conv2d_4a_3x3 = conv_block(80, 192, kernel_size=3).to(self.get_device('Conv2d_4a_3x3'))
        self.Mixed_5b = inception_a(192, pool_features=32, placement=placement, name='Mixed_5b')
        self.Mixed_5c = inception_a(256, pool_features=64, placement=placement, name='Mixed_5c')
        self.Mixed_5d = inception_a(288, pool_features=64, placement=placement, name='Mixed_5d')
        self.Mixed_6a = inception_b(288, placement=placement, name='Mixed_6a')
        self.Mixed_6b = inception_c(768, channels_7x7=128, placement=placement, name='Mixed_6b')
        self.Mixed_6c = inception_c(768, channels_7x7=160, placement=placement, name='Mixed_6c')
        self.Mixed_6d = inception_c(768, channels_7x7=160, placement=placement, name='Mixed_6d')
        self.Mixed_6e = inception_c(768, channels_7x7=192, placement=placement, name='Mixed_6e')
        if aux_logits:
            self.AuxLogits = inception_aux(768, num_classes, placement=placement)
        self.Mixed_7a = inception_d(768, placement=placement, name='Mixed_7a')
        self.Mixed_7b = inception_e(1280, placement=placement, name='Mixed_7b')
        self.Mixed_7c = inception_e(2048, placement=placement, name='Mixed_7c')
        self.fc = nn.Linear(2048, num_classes).to(self.get_device('softmax'))
        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    import scipy.stats as stats
                    stddev = m.stddev if hasattr(m, 'stddev') else 0.1
                    X = stats.truncnorm(-2, 2, scale=stddev)
                    values = torch.as_tensor(X.rvs(m.weight.numel()), dtype=m.weight.dtype)
                    values = values.view(m.weight.size())
                    with torch.no_grad():
                        m.weight.copy_(values)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def get_device(self, layer):
        return torch.device(self.placement[layer])

    def _transform_input(self, x):
        if self.transform_input:
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        return x

    def _forward(self, x):
        # N x 3 x 299 x 299
        x = x.to(self.get_device('Conv2d_1a_3x3'))
        x = self.Conv2d_1a_3x3(x)
        # N x 32 x 149 x 149
        x = x.to(self.get_device('Conv2d_2a_3x3'))
        x = self.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = x.to(self.get_device('Conv2d_2b_3x3'))
        x = self.Conv2d_2b_3x3(x)
        # N x 64 x 147 x 147
        x = x.to(self.get_device('MaxPool_3a_3x3'))
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # N x 64 x 73 x 73
        x = x.to(self.get_device('Conv2d_3b_1x1'))
        x = self.Conv2d_3b_1x1(x)
        # N x 80 x 73 x 73
        x = x.to(self.get_device('Conv2d_4a_3x3'))
        x = self.Conv2d_4a_3x3(x)
        # N x 192 x 71 x 71
        x = x.to(self.get_device('MaxPool_5a_3x3'))
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # N x 192 x 35 x 35
        x = self.Mixed_5b(x)
        # N x 256 x 35 x 35
        x = self.Mixed_5c(x)
        # N x 288 x 35 x 35
        x = self.Mixed_5d(x)
        # N x 288 x 35 x 35
        x = self.Mixed_6a(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6b(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6c(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6d(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6e(x)
        # N x 768 x 17 x 17
        aux_defined = self.training and self.aux_logits
        if aux_defined:
            aux = self.AuxLogits(x)
        else:
            aux = None
        # N x 768 x 17 x 17
        x = self.Mixed_7a(x)
        # N x 1280 x 8 x 8
        x = self.Mixed_7b(x)
        # N x 2048 x 8 x 8
        x = self.Mixed_7c(x)
        # N x 2048 x 8 x 8
        # Adaptive average pooling
        x = x.to(self.get_device('AvgPool_1a_8x8'))
        x = F.adaptive_avg_pool2d(x, (1, 1))
        # N x 2048 x 1 x 1
        x = x.to(self.get_device('Dropout_1b'))
        x = F.dropout(x, training=self.training)
        # N x 2048 x 1 x 1
        x = x.to(self.get_device('Conv2d_1c_1x1'))
        x = torch.flatten(x, 1)
        # N x 2048
        x = x.to(self.get_device('softmax'))
        x = self.fc(x)
        # N x 1000 (num_classes)
        return x, aux

    @torch.jit.unused
    def eager_outputs(self, x, aux):
        # type: (Tensor, Optional[Tensor]) -> InceptionOutputs
        if self.training and self.aux_logits:
            return InceptionOutputs(x, aux)
        else:
            return x

    def forward(self, x):
        x = self._transform_input(x)
        x, aux = self._forward(x)
        aux_defined = self.training and self.aux_logits
        if torch.jit.is_scripting():
            if not aux_defined:
                warnings.warn("Scripted Inception3 always returns Inception3 Tuple")
            return InceptionOutputs(x, aux)
        else:
            return self.eager_outputs(x, aux)


class InceptionA(nn.Module):

    def __init__(self, in_channels, pool_features, conv_block=None, placement=None, name=''):
        super(InceptionA, self).__init__()

        self.placement = placement
        self.name = name

        if conv_block is None:
            conv_block = BasicConv2d
    
        # Branch_0
        self.branch1x1 = conv_block(in_channels, 64, kernel_size=1)\
            .to(torch.device(self.placement[f'{self.name}/Branch_0/Conv2d_0a_1x1']))

        # Branch_1
        self.branch5x5_1 = conv_block(in_channels, 48, kernel_size=1)\
            .to(torch.device(self.placement[f'{self.name}/Branch_1/Conv2d_0a_1x1']))
        self.branch5x5_2 = conv_block(48, 64, kernel_size=5, padding=2)\
            .to(torch.device(self.placement[f'{self.name}/Branch_1/Conv2d_0b_5x5']))

        # Branch_2
        self.branch3x3dbl_1 = conv_block(in_channels, 64, kernel_size=1)\
            .to(torch.device(self.placement[f'{self.name}/Branch_2/Conv2d_0a_1x1']))
        self.branch3x3dbl_2 = conv_block(64, 96, kernel_size=3, padding=1)\
            .to(torch.device(self.placement[f'{self.name}/Branch_2/Conv2d_0b_3x3']))
        self.branch3x3dbl_3 = conv_block(96, 96, kernel_size=3, padding=1)\
            .to(torch.device(self.placement[f'{self.name}/Branch_2/Conv2d_0c_3x3']))

        # Branch_3
        self.branch_pool = conv_block(in_channels, pool_features, kernel_size=1)\
            .to(torch.device(self.placement[f'{self.name}/Branch_3/Conv2d_0b_1x1']))

    def _forward(self, x):
        # Branch_0
        branch1x1 = x.to(torch.device(self.placement[f'{self.name}/Branch_0/Conv2d_0a_1x1']))
        branch1x1 = self.branch1x1(branch1x1)

        # Branch_1
        branch5x5 = x.to(torch.device(self.placement[f'{self.name}/Branch_1/Conv2d_0a_1x1']))
        branch5x5 = self.branch5x5_1(branch5x5)
        branch5x5 = branch5x5.to(torch.device(self.placement[f'{self.name}/Branch_1/Conv2d_0b_5x5']))
        branch5x5 = self.branch5x5_2(branch5x5)

        # Branch_2
        branch3x3dbl = x.to(torch.device(self.placement[f'{self.name}/Branch_2/Conv2d_0a_1x1']))
        branch3x3dbl = self.branch3x3dbl_1(branch3x3dbl)
        branch3x3dbl = branch3x3dbl.to(torch.device(self.placement[f'{self.name}/Branch_2/Conv2d_0b_3x3']))
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = branch3x3dbl.to(torch.device(self.placement[f'{self.name}/Branch_2/Conv2d_0c_3x3']))
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        # Branch_3
        branch_pool = x.to(torch.device(self.placement[f'{self.name}/Branch_3/AvgPool_0a_3x3']))
        branch_pool = F.avg_pool2d(branch_pool, kernel_size=3, stride=1, padding=1)
        branch_pool = branch_pool.to(torch.device(self.placement[f'{self.name}/Branch_3/Conv2d_0b_1x1']))
        branch_pool = self.branch_pool(branch_pool)

        concat_device = torch.device(self.placement[f'{self.name}/concat'])
        outputs = [branch1x1.to(concat_device), branch5x5.to(concat_device),
                   branch3x3dbl.to(concat_device), branch_pool.to(concat_device)]
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class InceptionB(nn.Module):

    def __init__(self, in_channels, conv_block=None, placement=None, name=''):
        super(InceptionB, self).__init__()

        self.placement = placement
        self.name = name

        if conv_block is None:
            conv_block = BasicConv2d

        # Branch_0
        self.branch3x3 = conv_block(in_channels, 384, kernel_size=3, stride=2)\
            .to(torch.device(self.placement[f'{self.name}/Branch_0/Conv2d_1a_3x3']))

        # Branch_1
        self.branch3x3dbl_1 = conv_block(in_channels, 64, kernel_size=1)\
            .to(torch.device(self.placement[f'{self.name}/Branch_1/Conv2d_0a_1x1']))
        self.branch3x3dbl_2 = conv_block(64, 96, kernel_size=3, padding=1)\
            .to(torch.device(self.placement[f'{self.name}/Branch_1/Conv2d_0b_3x3']))
        self.branch3x3dbl_3 = conv_block(96, 96, kernel_size=3, stride=2)\
            .to(torch.device(self.placement[f'{self.name}/Branch_1/Conv2d_0b_3x3']))

    def _forward(self, x):
        # Branch_0
        branch3x3 = x.to(torch.device(self.placement[f'{self.name}/Branch_0/Conv2d_1a_3x3']))
        branch3x3 = self.branch3x3(branch3x3)

        # Branch_1̈́
        branch3x3dbl = x.to(torch.device(self.placement[f'{self.name}/Branch_1/Conv2d_0a_1x1']))
        branch3x3dbl = self.branch3x3dbl_1(branch3x3dbl)
        branch3x3dbl = branch3x3dbl.to(torch.device(self.placement[f'{self.name}/Branch_1/Conv2d_0b_3x3']))
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        # Branch_2
        branch_pool = x.to(torch.device(self.placement[f'{self.name}/Branch_2/MaxPool_1a_3x3']))
        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)

        concat_device = torch.device(self.placement[f'{self.name}/concat'])
        outputs = [branch3x3.to(concat_device), branch3x3dbl.to(concat_device), branch_pool.to(concat_device)]
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class InceptionC(nn.Module):

    def __init__(self, in_channels, channels_7x7, conv_block=None, placement=None, name=''):
        super(InceptionC, self).__init__()

        self.placement = placement
        self.name = name

        if conv_block is None:
            conv_block = BasicConv2d

        # Branch_0
        self.branch1x1 = conv_block(in_channels, 192, kernel_size=1)\
            .to(torch.device(self.placement[f'{self.name}/Branch_0/Conv2d_0a_1x1']))

        # Branch_1
        c7 = channels_7x7
        self.branch7x7_1 = conv_block(in_channels, c7, kernel_size=1)\
            .to(torch.device(self.placement[f'{self.name}/Branch_1/Conv2d_0a_1x1']))
        self.branch7x7_2 = conv_block(c7, c7, kernel_size=(1, 7), padding=(0, 3))\
            .to(torch.device(self.placement[f'{self.name}/Branch_1/Conv2d_0b_1x7']))
        self.branch7x7_3 = conv_block(c7, 192, kernel_size=(7, 1), padding=(3, 0))\
            .to(torch.device(self.placement[f'{self.name}/Branch_1/Conv2d_0b_7x1']))

        # Branch_2
        self.branch7x7dbl_1 = conv_block(in_channels, c7, kernel_size=1)\
            .to(torch.device(self.placement[f'{self.name}/Branch_2/Conv2d_0a_1x1']))
        self.branch7x7dbl_2 = conv_block(c7, c7, kernel_size=(7, 1), padding=(3, 0))\
            .to(torch.device(self.placement[f'{self.name}/Branch_2/Conv2d_0b_7x1']))
        self.branch7x7dbl_3 = conv_block(c7, c7, kernel_size=(1, 7), padding=(0, 3))\
            .to(torch.device(self.placement[f'{self.name}/Branch_2/Conv2d_0c_1x7']))
        self.branch7x7dbl_4 = conv_block(c7, c7, kernel_size=(7, 1), padding=(3, 0))\
            .to(torch.device(self.placement[f'{self.name}/Branch_2/Conv2d_0d_7x1']))
        self.branch7x7dbl_5 = conv_block(c7, 192, kernel_size=(1, 7), padding=(0, 3))\
            .to(torch.device(self.placement[f'{self.name}/Branch_2/Conv2d_0e_1x7']))

        # Branch_3
        self.branch_pool = conv_block(in_channels, 192, kernel_size=1)\
            .to(torch.device(self.placement[f'{self.name}/Branch_3/Conv2d_0b_1x1']))

    def _forward(self, x):
        # Branch_0
        branch1x1 = x.to(torch.device(self.placement[f'{self.name}/Branch_0/Conv2d_0a_1x1']))
        branch1x1 = self.branch1x1(branch1x1)

        # Branch_1
        branch7x7 = x.to(torch.device(self.placement[f'{self.name}/Branch_1/Conv2d_0a_1x1']))
        branch7x7 = self.branch7x7_1(branch7x7)
        branch7x7 = branch7x7.to(torch.device(self.placement[f'{self.name}/Branch_1/Conv2d_0b_1x7']))
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = branch7x7.to(torch.device(self.placement[f'{self.name}/Branch_1/Conv2d_0b_7x1']))
        branch7x7 = self.branch7x7_3(branch7x7)

        # Branch_2
        branch7x7dbl = x.to(torch.device(self.placement[f'{self.name}/Branch_2/Conv2d_0a_1x1']))
        branch7x7dbl = self.branch7x7dbl_1(branch7x7dbl)
        branch7x7dbl = branch7x7dbl.to(torch.device(self.placement[f'{self.name}/Branch_2/Conv2d_0b_7x1']))
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = branch7x7dbl.to(torch.device(self.placement[f'{self.name}/Branch_2/Conv2d_0c_1x7']))
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = branch7x7dbl.to(torch.device(self.placement[f'{self.name}/Branch_2/Conv2d_0d_7x1']))
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = branch7x7dbl.to(torch.device(self.placement[f'{self.name}/Branch_2/Conv2d_0e_1x7']))
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

        # Branch_3
        branch_pool = x.to(torch.device(self.placement[f'{self.name}/Branch_3/AvgPool_0a_3x3']))
        branch_pool = F.avg_pool2d(branch_pool, kernel_size=3, stride=1, padding=1)
        branch_pool = branch_pool.to(torch.device(self.placement[f'{self.name}/Branch_3/Conv2d_0b_1x1']))
        branch_pool = self.branch_pool(branch_pool)

        concat_device = torch.device(self.placement[f'{self.name}/concat'])
        outputs = [branch1x1.to(concat_device), branch7x7.to(concat_device),
                   branch7x7dbl.to(concat_device), branch_pool.to(concat_device)]
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class InceptionD(nn.Module):

    def __init__(self, in_channels, conv_block=None, placement=None, name=''):
        super(InceptionD, self).__init__()

        self.placement = placement
        self.name = name

        if conv_block is None:
            conv_block = BasicConv2d

        # Branch_0
        self.branch3x3_1 = conv_block(in_channels, 192, kernel_size=1)\
            .to(torch.device(self.placement[f'{self.name}/Branch_0/Conv2d_0a_1x1']))
        self.branch3x3_2 = conv_block(192, 320, kernel_size=3, stride=2)\
            .to(torch.device(self.placement[f'{self.name}/Branch_0/Conv2d_1a_3x3']))

        # Branch_1
        self.branch7x7x3_1 = conv_block(in_channels, 192, kernel_size=1)\
            .to(torch.device(self.placement[f'{self.name}/Branch_1/Conv2d_0a_1x1']))
        self.branch7x7x3_2 = conv_block(192, 192, kernel_size=(1, 7), padding=(0, 3))\
            .to(torch.device(self.placement[f'{self.name}/Branch_1/Conv2d_0b_1x7']))
        self.branch7x7x3_3 = conv_block(192, 192, kernel_size=(7, 1), padding=(3, 0))\
            .to(torch.device(self.placement[f'{self.name}/Branch_1/Conv2d_0b_7x1']))
        self.branch7x7x3_4 = conv_block(192, 192, kernel_size=3, stride=2)\
            .to(torch.device(self.placement[f'{self.name}/Branch_1/Conv2d_1a_3x3']))

    def _forward(self, x):
        # Branch_0
        branch3x3 = x.to(torch.device(self.placement[f'{self.name}/Branch_0/Conv2d_0a_1x1']))
        branch3x3 = self.branch3x3_1(branch3x3)
        branch3x3 = branch3x3.to(torch.device(self.placement[f'{self.name}/Branch_0/Conv2d_1a_3x3']))
        branch3x3 = self.branch3x3_2(branch3x3)

        # Branch_1
        branch7x7x3 = x.to(torch.device(self.placement[f'{self.name}/Branch_1/Conv2d_0a_1x1']))
        branch7x7x3 = self.branch7x7x3_1(branch7x7x3)
        branch7x7x3 = branch7x7x3.to(torch.device(self.placement[f'{self.name}/Branch_1/Conv2d_0b_1x7']))
        branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
        branch7x7x3 = branch7x7x3.to(torch.device(self.placement[f'{self.name}/Branch_1/Conv2d_0b_7x1']))
        branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
        branch7x7x3 = branch7x7x3.to(torch.device(self.placement[f'{self.name}/Branch_1/Conv2d_1a_3x3']))
        branch7x7x3 = self.branch7x7x3_4(branch7x7x3)

        # Branch_2
        branch_pool = x.to(torch.device(self.placement[f'{self.name}/Branch_2/MaxPool_1a_3x3']))
        branch_pool = F.max_pool2d(branch_pool, kernel_size=3, stride=2)

        concat_device = torch.device(self.placement[f'{self.name}/concat'])
        outputs = [branch3x3.to(concat_device), branch7x7x3.to(concat_device), branch_pool.to(concat_device)]
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class InceptionE(nn.Module):

    def __init__(self, in_channels, conv_block=None, placement=None, name=''):
        super(InceptionE, self).__init__()

        self.placement = placement
        self.name = name

        if conv_block is None:
            conv_block = BasicConv2d


        # Branch_0
        self.branch1x1 = conv_block(in_channels, 320, kernel_size=1)\
            .to(torch.device(self.placement[f'{self.name}/Branch_0/Conv2d_0a_1x1']))

        # Branch_1
        self.branch3x3_1 = conv_block(in_channels, 384, kernel_size=1)\
            .to(torch.device(self.placement[f'{self.name}/Branch_1/Conv2d_0a_1x1']))
        self.branch3x3_2a = conv_block(384, 384, kernel_size=(1, 3), padding=(0, 1))\
            .to(torch.device(self.placement[f'{self.name}/Branch_1/Conv2d_0b_1x3']))
        self.branch3x3_2b = conv_block(384, 384, kernel_size=(3, 1), padding=(1, 0))\
            .to(torch.device(self.placement[f'{self.name}/Branch_1/Conv2d_0b_3x1']))

        # Branch_2
        self.branch3x3dbl_1 = conv_block(in_channels, 448, kernel_size=1)\
            .to(torch.device(self.placement[f'{self.name}/Branch_2/Conv2d_0a_1x1']))
        self.branch3x3dbl_2 = conv_block(448, 384, kernel_size=3, padding=1)\
            .to(torch.device(self.placement[f'{self.name}/Branch_2/Conv2d_0b_3x3']))
        self.branch3x3dbl_3a = conv_block(384, 384, kernel_size=(1, 3), padding=(0, 1))\
            .to(torch.device(self.placement[f'{self.name}/Branch_2/Conv2d_0c_1x3']))
        self.branch3x3dbl_3b = conv_block(384, 384, kernel_size=(3, 1), padding=(1, 0))\
            .to(torch.device(self.placement[f'{self.name}/Branch_2/Conv2d_0d_3x1']))

        # Branch_3
        self.branch_pool = conv_block(in_channels, 192, kernel_size=1)\
            .to(torch.device(self.placement[f'{self.name}/Branch_3/Conv2d_0b_1x1']))

    def _forward(self, x):
        # Branch_0
        branch1x1 = x.to(torch.device(self.placement[f'{self.name}/Branch_0/Conv2d_0a_1x1']))
        branch1x1 = self.branch1x1(branch1x1)

        # Branch_1
        branch3x3 = x.to(torch.device(self.placement[f'{self.name}/Branch_1/Conv2d_0a_1x1']))
        branch3x3 = self.branch3x3_1(branch3x3)
        branch3x3 = [
            self.branch3x3_2a(branch3x3.to(torch.device(self.placement[f'{self.name}/Branch_1/Conv2d_0b_1x3']))),
            self.branch3x3_2b(branch3x3.to(torch.device(self.placement[f'{self.name}/Branch_1/Conv2d_0b_3x1']))),
        ]
        concat_device = torch.device(self.placement[f'{self.name}/Branch_1/concat'])
        branch3x3 = [branch3x3[0].to(concat_device), branch3x3[1].to(concat_device)]
        branch3x3 = torch.cat(branch3x3, 1)

        # Branch_2
        branch3x3dbl = x.to(torch.device(self.placement[f'{self.name}/Branch_2/Conv2d_0a_1x1']))
        branch3x3dbl = self.branch3x3dbl_1(branch3x3dbl)
        branch3x3dbl = branch3x3dbl.to(torch.device(self.placement[f'{self.name}/Branch_2/Conv2d_0b_3x3']))
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl.to(torch.device(self.placement[f'{self.name}/Branch_2/Conv2d_0c_1x3']))),
            self.branch3x3dbl_3b(branch3x3dbl.to(torch.device(self.placement[f'{self.name}/Branch_2/Conv2d_0d_3x1']))),
        ]
        concat_device = torch.device(self.placement[f'{self.name}/Branch_2/concat'])
        branch3x3dbl = [branch3x3dbl[0].to(concat_device), branch3x3dbl[1].to(concat_device)]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        # Branch_3
        branch_pool = x.to(torch.device(self.placement[f'{self.name}/Branch_3/AvgPool_0a_3x3']))
        branch_pool = F.avg_pool2d(branch_pool, kernel_size=3, stride=1, padding=1)
        branch_pool = branch_pool.to(torch.device(self.placement[f'{self.name}/Branch_3/Conv2d_0b_1x1']))
        branch_pool = self.branch_pool(branch_pool)

        concat_device = torch.device(self.placement[f'{self.name}/concat'])
        outputs = [branch1x1.to(concat_device), branch3x3.to(concat_device),
                   branch3x3dbl.to(concat_device), branch_pool.to(concat_device)]
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class InceptionAux(nn.Module):

    def __init__(self, in_channels, num_classes, conv_block=None, placement=None, name=''):
        super(InceptionAux, self).__init__()

        self.placement = placement
        self.name = name

        self.device = torch.device(self.placement['Mixed_6e/concat'])

        if conv_block is None:
            conv_block = BasicConv2d
        self.conv0 = conv_block(in_channels, 128, kernel_size=1).to(self.device)
        self.conv1 = conv_block(128, 768, kernel_size=5).to(self.device)
        self.conv1.stddev = 0.01
        self.fc = nn.Linear(768, num_classes).to(self.device)
        self.fc.stddev = 0.001

    def forward(self, x):
        x = x.to(self.device)
        # N x 768 x 17 x 17
        x = F.avg_pool2d(x, kernel_size=5, stride=3)
        # N x 768 x 5 x 5
        x = self.conv0(x)
        # N x 128 x 5 x 5
        x = self.conv1(x)
        # N x 768 x 1 x 1
        # Adaptive average pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))
        # N x 768 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 768
        x = self.fc(x)
        # N x 1000
        return x


class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, placement=None, name='', **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)
