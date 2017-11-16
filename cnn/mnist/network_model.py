"""
Created on Nov 7, 2017

Network model for handwritten character recognition

@author: Levan Tsinadze
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict

from torch import nn

import torch.nn.functional as F
from utils.models.layers import Flatten


class LeNetClassic(nn.Module):
  """Network model without flatten layer
   for character recognition"""
  
  def __init__(self):
    super(LeNetClassic, self).__init__()
    self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
    self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
    self.conv2_drop = nn.Dropout2d()
    self.fc1 = nn.Linear(320, 50)
    self.fc2 = nn.Linear(50, 10)
  
  def forward(self, x):
      
    x = F.relu(F.max_pool2d(self.conv1(x), 2))
    x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
    x = x.view(x.size(0), 320)
    x = F.relu(self.fc1(x))
    x = F.dropout(x, training=self.training)
    x = self.fc2(x)
    result = F.log_softmax(x)
    
    return result


class LeNet(nn.Module):
  """Network model with flatten layer
   for character recognition"""
  
  def __init__(self):
    super(LeNet, self).__init__()
    self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
    self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
    self.conv2_drop = nn.Dropout2d()
    self.flatten = Flatten(50)
    self.fc2 = nn.Linear(50, 10)
  
  def forward(self, x):
      
    x = F.relu(F.max_pool2d(self.conv1(x), 2))
    x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
    x = self.flatten(x)
    x = F.relu(x)
    x = F.dropout(x, training=self.training)
    x = self.fc2(x)
    result = F.log_softmax(x)
    
    return result


class LeNetSequential(nn.Module):
  """Network model with flatten layer
   for character recognition"""
  
  def __init__(self):
    super(LeNetSequential, self).__init__()
    self.conv_part = nn.Sequential(nn.Conv2d(1, 10, kernel_size=5),
                                   nn.MaxPool2d(2, 2),
                                   nn.ReLU(),
                                   nn.Conv2d(10, 20, kernel_size=5),
                                   nn.MaxPool2d(2, 2),
                                   nn.ReLU(),
                                   nn.Dropout2d())
    self.flatten = Flatten(50)
    self.fc2 = nn.Linear(50, 10)
  
  def forward(self, x):
      
    x = self.conv_part(x)
    x = self.flatten(x)
    x = F.relu(x)
    x = F.dropout(x, training=self.training)
    x = self.fc2(x)
    result = F.log_softmax(x)
    
    return result
  
  
class LeNetSequentialDict(nn.Module):
  """Network model with flatten layer
   for character recognition"""
  
  def __init__(self):
    super(LeNetSequentialDict, self).__init__()
    self.conv_part = nn.Sequential(OrderedDict([
                                   ('conv1', nn.Conv2d(1, 10, kernel_size=5)),
                                   ('mxpl1', nn.MaxPool2d(2, 2)),
                                   ('relu1', nn.ReLU()),
                                   ('conv2', nn.Conv2d(10, 20, kernel_size=5)),
                                   ('mxol2', nn.MaxPool2d(2, 2)),
                                   ('relu2', nn.ReLU()),
                                   ('drop1', nn.Dropout2d())]))
    self.flatten = Flatten(50)
    self.fc2 = nn.Linear(50, 10)
  
  def forward(self, x):
      
    x = self.conv_part(x)
    x = self.flatten(x)
    x = F.relu(x)
    x = F.dropout(x, training=self.training)
    x = self.fc2(x)
    result = F.log_softmax(x)
    
    return result

