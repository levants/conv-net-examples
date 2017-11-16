"""
Created on Nov 15, 2017

Network model interface for letres

@author: Levan Tsinadze
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from torchvision import transforms

transform_func = transforms.Compose([
                         transforms.ToTensor(),
                         transforms.Normalize((0.1307,), (0.3081,))
                     ])


def run_model(model, input_image):
  
  x = transform_func(input_image)
  x = x.unsqueeze(1)
  res = model(x)
  
  return res
