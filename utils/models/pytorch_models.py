"""
Created on Oct 14, 2017

Utility module for PyTorch network models

@author: Levan Tsinadze
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


def _validate_parameters(layer):
  """Validate if has parameters
    Args:
      layer - layer to freeze parameters
    Returns:
      has_parameters - if layer has parameters
  """
  
  has_parameters = False
  for _ in layer.parameters():
    has_parameters = has_parameters if has_parameters else True
    break
  
  return has_parameters


def _count_children(model, layer_count):
  """Counts layer for given layer and it's children
    Args:
      model - network model
      layer_count - count of freeze layers
      verbose - flag to log process
    Returns:
      ct - count of layers
  """
  
  ct = layer_count
  for child in model.children():
    tmp_ct = _count_children(child, ct)
    if tmp_ct == ct:
      has_parameters = _validate_parameters(child)
      ct = ct + 1 if has_parameters else ct
    else:
      ct = tmp_ct
          
  return ct


def count_layers(model):
  """Counts layers in model
    Args:
      model - model to count layers
  """
  return _count_children(model, 1)


def _freeze_parameters(layer, ct, n_freeze):
  """Freeze layer's parameters
    Args:
      layer - layer to freeze parameters
      ct - count for current layer
      n_freeze - number of first layers to freeze
    Returns:
      has_parameters - if layer has parameters
  """
  
  has_parameters = False
  for param in layer.parameters():
    has_parameters = has_parameters if has_parameters else True
    param.requires_grad = ct > n_freeze
  
  return has_parameters


def _freeze_children(model, layer_count, n_freeze, verbose):
  """Freezes layer for given layer and it's children
    Args:
      model - network model
      layer_count - count of freeze layers
      n_freeze - number of layers to freeze
      verbose - flag for process logging
    Returns:
      ct - count of layers
  """
  
  ct = layer_count
  for child in model.children():
    tmp_ct = _freeze_children(child, ct, n_freeze, verbose)
    if tmp_ct == ct:
      has_parameters = _freeze_parameters(child, ct, n_freeze)
      ct = ct + 1 if has_parameters else ct
      print('Freeze ', ct, ' layer')
    else:
      ct = tmp_ct
          
  return ct


def freeze_layers(model, n_freeze, verbose=False):
  """Freezes layers in model
    Args:
      model - model to freeze layers
      n_freeze - number for first layers to be freeze
      verbose - fag to log process
    Returns:
      trainebles - model parameters (weights) for training
  """
  
  _freeze_children(model, 1, n_freeze, verbose)
  trainebles = [param for param in model.parameters() if param.requires_grad]
  
  return trainebles
  
