"""
Created on Oct 14, 2017

Network model configuration for fine-tuning

@author: Levan Tsinadze
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import shutil

from torch import nn
import torch
from torchvision.models import (resnet18, resnet34, resnet50)

import torch.nn.functional as F
from utils.files import file_utils as _files
from utils.models import pytorch_models as pymodels

STATE_DICT_KEY = 'state_dict'


class TransferModel(nn.Module):
  """Transfer learning module"""
  
  def __init__(self, base_model, fc_size,
               keep_prob=.2,
               keep_dense_prob=.5,
               num_classes=None):
    super(TransferModel, self).__init__()
    # Hyper-parameters
    self.inplace = not self.training
    self.keep_prob = keep_prob
    self.keep_dense_prob = keep_dense_prob
    num_fc_features = base_model.fc.in_features
    # Model blocks
    self.base_model = base_model
    self.avgpool = nn.AvgPool2d(7)
    self.fc1 = nn.Linear(num_fc_features, fc_size)
    self.bn_f1 = nn.BatchNorm1d(fc_size)
    self.fc_logits = nn.Linear(fc_size , num_classes)
    
  def forward(self, input_tensor):
    """Forward propagation of network model with additional layers
      Args:
        input_tensor - input tensor
      Returns:
        logits - calculated output logits
    """

    x = self.base_model.features(input_tensor)
    x = F.dropout2d(x, p=self.keep_prob, training=self.training, inplace=self.inplace)
    x = self.avgpool(x)
    x = x.view(x.size(0), -1)
    x = self.fc1(x)
    x = self.bn_f1(x)
    x = F.relu(x, inplace=self.inplace)
    x = F.dropout(x, p=self.keep_dense_prob, training=self.training, inplace=self.inplace)
    logits = self.fc_logits(x)
    
    return logits
  

def get_num_classes(flags, num_classes):
  """Gets number of classes
    Args:
      flags - training configuration flags
      num_classes - number of classes
    Returns:
      number of classes
  """
  return flags.num_classes if num_classes is None else num_classes


def init_model(flags, base_model_type, num_classes=None):
  """Initializes network model
    Args:
      flags - configuration parameters and flags
      base_model_type - base model initializer
      num_classes - number of classes
    Returns:
      model - network model
  """
  
  base_model = base_model_type(pretrained=True)
  num_classes = get_num_classes(flags, num_classes)
  model = TransferModel(base_model, flags.fc_size,
                        keep_prob=flags.keep_prob,
                        keep_dense_prob=flags.keep_dense_prob,
                        num_classes=num_classes)
  
  return model


def init_model_and_weights(flags, base_model_type):
  """Initializes network model and loads appropriated weights
    Args:
      flags - configuration parameters and flags
      base_model_type - type of base network model
    Returns:
      model - network model
  """
  
  model = init_model(flags, base_model_type, num_classes=flags.num_classes)
  checkpoint = torch.load(flags.weights, map_location=lambda storage, loc: storage)
  state_dict = checkpoint[STATE_DICT_KEY]
  model.load_state_dict(state_dict)
  
  return model


def init_eval_model(flags, base_model_type):
  """Initializes model and loads weights for evaluation
    Args:
      flags - configuration parameters
      base_model_type - type of base network model
    Returns:
      model - network model
  """
  
  model = init_model_and_weights(flags, base_model_type)
  model.eval()
  
  return model
 

def fine_tune_model(flags, model, num_freeze_layers=None):
  """Setups network model for fine tuning
    Args:
      flags - command line arguments
      model - existed network model
      num_freeze_layers - number of layer to freeze
  """
  
  n_freeze = flags.num_freeze_layers if num_freeze_layers is None else num_freeze_layers
  pymodels.freeze_layers(model, n_freeze, flags.verbose)
      
        
def init_num_of_freeze_layers(base_model):
  """Initializes number of layers to freeze
    Args:
      base_model - network model
    Returns:
      num_freeze_layers - number of network's first layers to freeze
  """
  return len(pymodels.count_layers(base_model)) - 1 \
         if type(base_model) in (resnet18, resnet34, resnet50) \
         else None


def init_model_and_freeze_layers(network_parameters):
  """Initializes network model and freezes layers
    Args:
      network_parameters - network parameters tuple
    Returns:
      tuple of -
        renset_model - network model
        trainables - parameters for training
  """
  
  (flags, model_type, num_classes, trainable_layers) = network_parameters
  renset_model = init_model(flags, model_type, num_classes)
  layer_count = pymodels.count_layers(renset_model)
  layers_to_freeze = (layer_count - trainable_layers)
  trainables = pymodels.freeze_layers(renset_model, layers_to_freeze, verbose=flags.verbose)
  
  return (renset_model, trainables)


def save_checkpoint(flags, state, is_best, filename='checkpoint.pth.tar'):
  """Saves checkpoint to disk
    Args:
      flags - configuration parameters
      state - state parameter
      is_best - flag if checkpoint has best accuracy
      filename - checkpoint file name
  """
  
  directory = flags.model_dir
  filename = _files.join(directory, filename)
  torch.save(state, filename)
  if is_best:
    shutil.copyfile(filename, _files.join(directory, flags.weights))

