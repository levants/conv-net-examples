"""
Created on Nov 7, 2017

Training configuration for character detection

@author: Levan Tsinadze
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import torch

from cnn.mnist.cnn_files import files as _files


def configure():
  """Configuration parameters
    Returns:
      flags - configuration parameters
  """
  
  parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
  parser.add_argument('--batch-size',
                      type=int,
                      default=64,
                      metavar='N',
                      help='input batch size for training (default: 64)')
  parser.add_argument('--test-batch-size',
                      type=int,
                      default=1000,
                      metavar='N',
                      help='input batch size for testing (default: 1000)')
  parser.add_argument('--epochs',
                      type=int,
                      default=10,
                      metavar='N',
                      help='number of epochs to train (default: 10)')
  parser.add_argument('--lr',
                      type=float,
                      default=0.01,
                      metavar='LR',
                      help='learning rate (default: 0.01)')
  parser.add_argument('--momentum',
                      type=float,
                      default=0.5,
                      metavar='M',
                      help='SGD momentum (default: 0.5)')
  parser.add_argument('--no-cuda',
                      action='store_true',
                      default=False,
                      help='disables CUDA training')
  parser.add_argument('--seed',
                      type=int,
                      default=1,
                      metavar='S',
                      help='random seed (default: 1)')
  parser.add_argument('--log-interval',
                      type=int,
                      default=10,
                      metavar='N',
                      help='how many batches to wait before logging training status')
  parser.add_argument('--use_classic',
                      action='store_true',
                      default=False,
                      help='Use classic network without flatten layers')
  parser.add_argument('--attach_cpu',
                      action='store_true',
                      default=False,
                      help='Attach model to CPU device')
  # Save training model
  parser.add_argument('--weights',
                      type=str,
                      default=_files.model_file('mnist_weights.pth.tar'),
                      help='Where to save trained weights')
  # Host and port for http service
  parser.add_argument('--host',
                      type=str,
                      default='0.0.0.0',
                      help='Host name for HTTP service.')
  parser.add_argument('--port',
                      type=int,
                      default=8080,
                      help='Port number for HTTP service.')
  flags = parser.parse_args()
  flags.cuda = not flags.no_cuda and torch.cuda.is_available()
  
  return flags
