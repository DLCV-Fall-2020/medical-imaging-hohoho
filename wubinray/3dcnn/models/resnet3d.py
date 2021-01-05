#Ref:https://github.com/kenshohara/video-classification-3d-cnn-pytorch/blob/master/models/resnet.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial

__all__ = ['HemoResNet3d', 'resnet10_3d', 'resnet18_3d', 'resnet34_3d', 'resnet50_3d', 'resnet101_3d', 'resnet152_3d', 'resnet200_3d']


def conv3x3x3(in_planes, out_planes, stride=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(in_planes, out_planes, kernel_size=3,
                     stride=(1,stride,stride), padding=1, bias=False)


def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(out.size(0), planes - out.size(1),
                             out.size(2), out.size(3),
                             out.size(4)).zero_()
    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=(1,stride,stride),
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class HemoResNet3d(nn.Module):

    def __init__(self, block, layers, in_channels=1, hidd_dim=128, shortcut_type='B', num_classes=400, last_fc=False):
        self.last_fc = last_fc
        
        self.inplanes = 64
        super(HemoResNet3d, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=7, stride=(1, 2, 2),
                               padding=(3, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(1,2,2), padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(block, 96, layers[1], shortcut_type, stride=4)
        self.layer3 = self._make_layer(block, 128, layers[2], shortcut_type, stride=2)
        self.layer4 = self._make_layer(block, 128, layers[3], shortcut_type, stride=2)
        #last_duration = math.ceil(sample_duration / 16)
        #last_size = math.ceil(sample_size / 32)
        #self.avgpool = nn.AvgPool3d((last_duration, last_size, last_size), stride=1, padding=1)
        #self.avgpool = nn.AvgPool3d(3, stride=(1,2,2), padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128 * block.expansion, hidd_dim)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=(1,stride,stride))
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=(1,stride,stride), bias=False),
                    nn.BatchNorm3d(planes * block.expansion)
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.permute(0,2,1,3,4) #(b,ch,t,h,w)->(b,t,ch,h,w)
        
        b,t,ch,h,w = x.shape 
        x = self.avgpool(x.contiguous().view(b*t,ch,h,w))
        x = x.view(b,t,ch)
        
        x = x.contiguous().view(b,t,-1)
        x = self.fc(x) #(b,t,hidd_dim)

        return x


def get_fine_tuning_parameters(model, ft_begin_index):
    if ft_begin_index == 0:
        return model.parameters()

    ft_module_names = []
    for i in range(ft_begin_index, 5):
        ft_module_names.append('layer{}'.format(ft_begin_index))
    ft_module_names.append('fc')

    parameters = []
    for k, v in model.named_parameters():
        for ft_module in ft_module_names:
            if ft_module in k:
                parameters.append({'params': v})
                break
        else:
            parameters.append({'params': v, 'lr': 0.0})

    return parameters


def resnet10_3d(**kwargs):
    """Constructs a HemoResNet3d-18 model.
    """
    model = HemoResNet3d(BasicBlock, [1, 1, 1, 1], **kwargs)
    return model

def resnet18_3d(**kwargs):
    """Constructs a HemoResNet3d-18 model.
    """
    model = HemoResNet3d(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model

def resnet34_3d(**kwargs):
    """Constructs a HemoResNet3d-34 model.
    """
    model = HemoResNet3d(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model

def resnet50_3d(**kwargs):
    """Constructs a HemoResNet3d-50 model.
    """
    model = HemoResNet3d(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model

def resnet101_3d(**kwargs):
    """Constructs a HemoResNet3d-101 model.
    """
    model = HemoResNet3d(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model

def resnet152_3d(**kwargs):
    """Constructs a HemoResNet3d-101 model.
    """
    model = HemoResNet3d(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model

def resnet200_3d(**kwargs):
    """Constructs a HemoResNet3d-101 model.
    """
    model = HemoResNet3d(Bottleneck, [3, 24, 36, 3], **kwargs)
    return model