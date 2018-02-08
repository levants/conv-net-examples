"""
Created on Oct 14, 2017

Implementation of ResNet for fine-tuning, transfer learning, features extractor etc

@author: Levan Tsinadze
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from torch import nn
from torch.utils import model_zoo
from torchvision.models.resnet import (ResNet, Bottleneck, \
                                       BasicBlock, model_urls)


class ResNetModule(ResNet):
    """ResNet extension module"""

    def __init__(self, block, layers, channels=3, num_classes=1000):
        super(ResNetModule, self).__init__(block, layers, num_classes=num_classes)
        self.conv1 = nn.Conv2d(channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

    def features(self, input_tensor):
        """Extracts features from input tensor
          Args:
            input_tensor - input image tensor
          Returns:
            features_tensor - features tensor
        """

        x = self.conv1(input_tensor)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        features_tensor = self.layer4(x)

        return features_tensor

    def extract_features(self, input_tensor):
        """Extracts features for passed tensor
          Args:
            input_tensor - input image tensor
          Returns: 
            extracted features
        """
        return self.features(input_tensor)

    def forward(self, input_tensor):
        x = self.features(input_tensor)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        logits = self.fc(x)

        return logits


def _init_model(core_type=ResNetModule, block=BasicBlock, layers=[2, 2, 2, 2],
                model_key='resnet18', pretrained=False, **kwargs):
    """Initializes appropriated model
      Args:
        core_type - type for model core initialization
        block - block for layers initialization
        layers - model layers
        model_key - key for model URL dictionary
        pretrained - flags for trained weights
        kwargs - additional arguments
      Returns:
        model - network model with weights
    """

    model = core_type(block, layers, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls[model_key]))

    return model


def _init_module(block=BasicBlock, layers=[2, 2, 2, 2],
                 model_key='resnet18', pretrained=False, **kwargs):
    """Initializes appropriated model
      Args:
        block - block for layers initialization
        layers - model layers
        pretrained - flags for trained weights
        kwargs - additional arguments
      Returns:
        network model with weights
    """
    return _init_model(core_type=ResNetModule, block=block, layers=layers, model_key=model_key,
                       pretrained=pretrained, **kwargs)


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
      Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
      Returns:
        network model width weights
    """
    return _init_module(block=BasicBlock, layers=[2, 2, 2, 2], model_key=resnet18.__name__, \
                        pretrained=pretrained, **kwargs)


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
      Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
      Returns:
        network model with weights
    """
    return _init_module(block=BasicBlock, layers=[3, 4, 6, 3], model_key=resnet34.__name__, \
                        pretrained=pretrained, **kwargs)


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
      Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
      Returns:
        network model with weights
    """
    return _init_module(block=Bottleneck, layers=[3, 4, 6, 3], model_key=resnet50.__name__, \
                        pretrained=pretrained, **kwargs)


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
      Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
      Returns:
        network model with weights
    """
    return _init_module(block=Bottleneck, layers=[3, 4, 23, 3], model_key=resnet101.__name__, \
                        pretrained=pretrained, **kwargs)


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
      Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
      Returns:
        network model with weights
    """
    return _init_module(block=Bottleneck, layers=[3, 8, 36, 3], model_key=resnet152.__name__, \
                        pretrained=pretrained, **kwargs)
