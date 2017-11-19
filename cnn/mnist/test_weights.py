"""
Created on Nov 19, 2017

Test module for weights loading

@author: Levan Tsinadze
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest

from cnn.mnist.cnn_files import files as _files
from cnn.mnist.network_model import LeNet
import torch


class TestWeightsLoader(unittest.TestCase):
  """Test cases for directory data loader"""
  
  def setUp(self):
    self.weights = _files.model_file('mnist_weights.pth.tar')
      
  def test_load_state_dict(self):
    """Test case for loading weights in models with Flatten layer"""
    
    model = LeNet()
    weights_dict = torch.load(self.weights, map_location=lambda storage, loc: storage)
    model.load_state_dict(weights_dict)
