"""
Created on May 20, 2017

Flags and command line arguments for training

@author: Levan Tsinadze
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

from cnn.transfer.cnn_files import files as _files


MODEL_LABELS = 'model_labels.json'


def read_common_parameters(arg_parser):
  """Reads command line arguments
    Args:
      arg_parser - command line arguments parser
  """
  
  # Network architecture
  arg_parser.add_argument('--fc_size',
                          type=int,
                          default=1024,
                          help='Fully connected layer size')
  arg_parser.add_argument('--num_classes',
                          type=int,
                          help='Number of classes')
  arg_parser.add_argument('--trainable_layers',
                          type=int,
                          default=5,
                          help='Number of layers to train')
  # Model configuration
  arg_parser.add_argument('--model_dir',
                          type=str,
                          default=_files.model_dir,
                          help='Path to network models directory.')
  arg_parser.add_argument('--weights',
                          type=str,
                          default=_files.model_file('resnet18.pth.tar'),
                          help='Where to save the trained data.')
  arg_parser.add_argument('--save_model',
                          dest='save_model',
                          action='store_true',
                          help='Prints data set file names and labels.')
  # Logging configuration
  arg_parser.add_argument('--verbose',
                          dest='verbose',
                          action='store_true',
                          help='Log steps.')
  arg_parser.add_argument('--plot',
                          dest='plot',
                          action='store_true',
                          help='Flag to plot training.')


def read_interface_parameters():
  """Parses and retrieves parameters to run the interface
    Returns:
      arg_parser - argument parses
      flags - configuration flags
  """
  
  arg_parser = argparse.ArgumentParser()
  
  read_common_parameters(arg_parser)
  arg_parser.add_argument('--image',
                          type=str,
                          help='Path to image for prediction.')
  arg_parser.add_argument('--url',
                          type=str,
                          help='URL to image for prediction.')
  arg_parser.add_argument('--labels',
                          type=str,
                          help='Path to labels with indices JSON file.')
  # Host and port for http service
  arg_parser.add_argument('--host',
                          type=str,
                          default='0.0.0.0',
                          help='Host name for HTTP service.')
  arg_parser.add_argument('--port',
                          type=int,
                          default=50050,
                          help='Port number for HTTP service.')
  (flags, _) = arg_parser.parse_known_args()
  
  return (flags, arg_parser)


def read_training_parameters():
  """Reads command line arguments
    Returns:
      flags - training configuration flags
  """
  
  arg_parser = argparse.ArgumentParser()
  
  
  read_common_parameters(arg_parser)
  # Fine tune or transfer learning
  arg_parser.add_argument('--fine_tune',
                          dest='fine_tune',
                          action='store_true',
                          help='Flag to choose between transfer and fine-tune model.')
  # Training hyper parameters configuration
  arg_parser.add_argument('--optimizer',
                          type=str,
                          default='adam',
                          help='Const function')
  arg_parser.add_argument('--batch_size',
                          type=int,
                          default=32,
                          help='Training batch size')
  arg_parser.add_argument('--epochs',
                          type=int,
                          default=15,
                          help='Number of training epochs')
  arg_parser.add_argument('--keep_prob',
                          type=float,
                          default=0.2,
                          help='Dropout keep probability')
  arg_parser.add_argument('--keep_dense_prob',
                          type=float,
                          default=0.5,
                          help='Dropout keep probability for fully connected layers')
  arg_parser.add_argument('--learning_rate',
                          type=float,
                          default=0.0001,
                          help='Learning rate')
  arg_parser.add_argument('--momentum',
                          type=float,
                          default=0.9,
                          help='Learning momentum')
  arg_parser.add_argument('--weight_decay',
                          type=float,
                          default=1e-4,
                          help='weight decay (default: 1e-4)')
  # Data set configuration
  arg_parser.add_argument('--dataset_dir',
                          type=str,
                          default=_files.data_file('dataset'),
                          help='Path to folders of labeled images for pre-processing and training.') 
  arg_parser.add_argument('--train_dir',
                          type=str,
                          default=_files.data_file('training'),
                          help='Path to folders of labeled images for training.')
  arg_parser.add_argument('--val_dir',
                          type=str,
                          default=_files.data_file('validation'),
                          help='Path to folders of labeled images for validation.')
  arg_parser.add_argument('--val_precentage',
                          type=float,
                          default=0.2,
                          help='Percentage of validation data.')
  arg_parser.add_argument('--test_dir',
                          type=str,
                          default=_files.data_file('test'),
                          help='Path to folders of labeled images for testing.')
  arg_parser.add_argument('--test_precentage',
                          type=float,
                          default=0.2,
                          help='Percentage of test data.')
  # System configuration
  arg_parser.add_argument('--num_workers',
                          type=int,
                          default=4,
                          help='Number of data loader workers')
  # Logging configuration
  arg_parser.add_argument('--print_dataset',
                          dest='print_dataset',
                          action='store_true',
                          help='Prints data set file names and labels.')
  (flags, _) = arg_parser.parse_known_args()
  
  return flags

