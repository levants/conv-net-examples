"""
Created on Oct 16, 2017

Flatten layer for PyTorch models

@author: Levan Tsinadze
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from torch import nn

import numpy as np


class Flatten(nn.Module):
  """Module to flatten network layer"""
  
  def __init__(self, out_dim):
    super(Flatten, self).__init__()
    self.out_dim = out_dim
    self.fc_layer = None
    self._apply_fns = []
    
  def _apply(self, fn):
    """Saves passed function for initialized linear layer
      Args:
        fn - functions to apply
      Returns:
        current object instance
    """
    
    self._apply_fns.append(fn)
    _apply_result = super(Flatten, self)._apply(fn)
    
    return _apply_result
  
  def _apply_postfactum(self):
    """Applies functions from module"""
    
    for fn in self._apply_fns:
      self.fc_layer._apply(fn)
  
  def calculate_total(self, x):
    """Calculates total dimension of tensor
      Args:
        x - tensor to calculate total dimension
    """
    
    if self.fc_layer is None:
      self.flatten_dim = 0 if x is None else np.prod(x.size()[1:])
      self.fc_layer = nn.Linear(self.flatten_dim, self.out_dim)
      self._apply_postfactum()
    
  def forward(self, input_tensor):
    """Flattens passed tensor and calculates input dimension
      Args:
        input_tensor - input tensor
      Returns:
        x - flattened tensor
    """
    
    self.calculate_total(input_tensor)
    x = input_tensor.view(input_tensor.size(0), self.flatten_dim)
    x = self.fc_layer(x)
    
    return x
    
  def __repr__(self):
    return self.__class__.__name__ + ' (None -> ' \
        + str(self.out_dim) + ')'
