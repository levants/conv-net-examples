"""
Created on Nov 19, 2017

Implementation of flatten layer

@author: Levan Tsinadze
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

from torch import nn
import torch
from torch.nn.parameter import Parameter

import numpy as np
import torch.nn.functional as F


class Flatten(nn.Linear):
  """Flatten layer"""
  
  def __init__(self, out_features, in_features=None, bias=True):
    super(nn.Linear, self).__init__()
    self.in_features = in_features
    self.out_features = out_features
    if in_features is None:
      self.weight = None
      self.register_parameter('weight', None)
    else:
      self.weight = Parameter(torch.Tensor(out_features, in_features))
    if bias:
        self.bias = Parameter(torch.Tensor(out_features))
    else:
        self.register_parameter('bias', None)
    self.reset_parameters()
    
  def reset_parameters(self):
    
    if self.weight is not None:
      stdv = 1. / math.sqrt(self.weight.size(1))
      self.weight.data.uniform_(-stdv, stdv)
      if self.bias is not None:
        self.bias.data.uniform_(-stdv, stdv)
        
  def calculate_total(self, x):
    """Calculates total dimension of tensor
      Args:
        x - tensor to calculate total dimension
    """
    
    if self.weight is None:
      self.in_features = 0 if x is None else np.prod(x.size()[1:])
      self.weight = Parameter(torch.Tensor(self.out_features, self.in_features))
      self.register_parameter('weight', self.weight)
      self.reset_parameters()   

  def forward(self, input_tensor):
    
    self.calculate_total(input_tensor)
    x = input_tensor.view(input_tensor.size(0), self.in_features)
    linear_result = F.linear(x, self.weight, self.bias) 
    
    return linear_result