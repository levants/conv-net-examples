"""
Created on Sep 28, 2017

Configuration utilities for PyTorch library

@author: Levan Tsinadze
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from  torch import nn
import torch
from torch.nn.parallel.data_parallel import DataParallel


# Key for saved state dictionary (trained model's weights only)
STATE_DICT_KEY = 'state_dict'

def gpu_available():
  """Validates if GPU devices available
    Returns:
      validation result
  """
  return torch.cuda.is_available()


def gpu_seed(cuda, seed):
  """Set CUDA seed
    Args:
      cuda - gpu flag
      seed - seed for gpu
  """
  
  if cuda:
    torch.cuda.manual_seed(seed)


def cuda_seed(flags):
  """Set CUDA seed
    Args:
      flags - training configuration parameters
  """
  gpu_seed(flags.cuda, flags.seed)
    

def _attach_tuple(tnsrs):
  """Attaches tuple on GPU device"""
  
  cuda_vars_list = []
  for tnsr in tnsrs:
    cuda_vars_list.append(tnsr.cuda())
  cuda_vars = tuple(cuda_vars_list)
  
  return cuda_vars


def validate_and_attach_gpu(cuda, tnsrs):
  """Attaches cuda if available
    Args:
      cuda - cuda flag
      tnsrs - variable to attach
    Returns:
      cuda_vars - attached variables tuple
  """

  if cuda:
    if type(tnsrs) == tuple:
      cuda_vars = _attach_tuple(tnsrs)
    else:
      cuda_vars = tnsrs.cuda()
  else:
    cuda_vars = tnsrs
  
  return cuda_vars


def attach_if_cuda(tnsrs):
  """Validates and attaches GPU if available
    Args:
      tnsrs - variables to attach
    Returns:
      cuda_vars - attached variables
  """
  
  cuda = gpu_available()
  cuda_vars = validate_and_attach_gpu(cuda, tnsrs)
  
  return cuda_vars


def validate_and_init_cuda(init_funct):
  """Initializes function and attaches GPU if available
    Args:
      init_funct - model initialization functon
    Returns:
      model - initialized by function
  """
  
  net = init_funct()
  model = net.cuda() if torch.cuda.is_available() else net
        
  return model
  

def attach_cuda(flags, tnsrs):
  """Attaches cuda to variables
    Args:
      flags - configuration flags
      tnsrs - variables to attach
    Returns:
      cuda_vars - attached variables tuple
  """
  return validate_and_attach_gpu(flags.cuda, tnsrs)  


def wrap_multidevice(model, gpu_ids):
  """Attach model to multi-devices
    Args:
      model - network model
      gpu_ids - device identifiers
    Returns:
      attached model
  """ 
  return nn.DataParallel(model, device_ids=gpu_ids) \
          if gpu_ids and len(gpu_ids) > 1 else model
          

def get_original_property(model, model_prop):
  """Get original property if model is wrapped
    Args:
      model - wrapped model
      model_prop - model property
    Returns:
      model property value
  """
  return model.module.model_prop if isinstance(model, DataParallel) else model.model_prop
  
