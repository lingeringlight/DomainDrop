from torch.utils import model_zoo
from torchvision.models.resnet import BasicBlock, model_urls, Bottleneck
from torch import nn as nn
from .LayerDiscriminator import LayerDiscriminator
import random

class ResNet(nn.Module):
    def __init__(self, block, layers,
                 device,
                 classes=100,
                 domains=3,
                 network='resnet18',
                 domain_discriminator_flag=0,
                 grl=0,
                 lambd=0.,
                 drop_percent=0.33,
                 dropout_mode=0,
                 wrs_flag=0,
                 recover_flag=0,
                 layer_wise_flag=0,
                 ):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512 * block.expansion, classes)

        if network == "resnet18":
            layer_channels = [64, 64, 128, 256, 512]
        else:
            layer_channels = [64, 256, 512, 1024, 2048]

        self.device = device
        self.domain_discriminator_flag = domain_discriminator_flag
        self.drop_percent = drop_percent
        self.dropout_mode = dropout_mode

        self.recover_flag = recover_flag
        self.layer_wise_flag = layer_wise_flag

        self.domain_discriminators = nn.ModuleList([
            LayerDiscriminator(
                num_channels=layer_channels[layer],
                num_classes=domains,
                grl=grl,
                reverse=True,
                lambd=lambd,
                wrs_flag=wrs_flag,
                )
            for i, layer in enumerate([0, 1, 2, 3, 4])])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def is_patch_based(self):
        return False

    def perform_dropout(self, feature, domain_labels, layer_index, layer_dropout_flag):
        domain_output = None
        if self.domain_discriminator_flag and self.training:
            index = layer_index
            percent = self.drop_percent
            domain_output, domain_mask = self.domain_discriminators[index](
                feature.clone(),
                domain_labels,
                percent=percent,
            )
            if self.recover_flag:
                domain_mask = domain_mask * domain_mask.numel() / domain_mask.sum()
            if layer_dropout_flag:
                feature = feature * domain_mask
        return feature, domain_output

    def forward(self, x, domain_labels=None, layer_drop_flag=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        domain_outputs = []
        for i, layer in enumerate([self.layer1, self.layer2, self.layer3, self.layer4]):
            x = layer(x)
            x, domain_output = self.perform_dropout(x, domain_labels, layer_index=i + 1,
                                                    layer_dropout_flag=layer_drop_flag[i])
            if domain_output is not None:
                domain_outputs.append(domain_output)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)  # B x C
        y = self.classifier(x)
        return y, domain_outputs


def resnet18(pretrained=True, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']), strict=False)
    return model


def resnet50(pretrained=True, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']), strict=False)
    return model

