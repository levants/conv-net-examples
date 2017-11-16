"""
Created on Oct 14, 2017

Fine-tune trained model

@author: Levan Tsinadze
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from torch.optim import lr_scheduler
import torchvision

from cnn.resnet.resnet import resnet18
from cnn.transfer import (dataset_config as datasets,
                          network_model as networks,
                          training_flags as config,
                          train_model as training)
import torch.nn as nn
import torch.optim as optim


def run_training(flags):
  """Runs fine-tuning of network model
    Args:
      flags - configuration parameters
    Returns:
      tuple of -
        class_names - class names
        class_to_idx - class indices
  """
  
  # import matplotlib.pyplot as plt
  # import numpy as np
  # plt.ion()  # interactive mode
  # Data augmentation and normalization for training
  # Just normalization for validation
  (dataloders, dataset_sizes, class_names, class_to_idx) = datasets.init_datasets(flags)
  
  
  # Get a batch of training data
  (inputs, _) = next(iter(dataloders[datasets.TRAIN_PHASE]))
  
  # Make a grid from batch
  _ = torchvision.utils.make_grid(inputs)
  
  # imshow(out, title=[class_names[x] for x in classes])
  # Network implementation
  network_parameters = (flags, resnet18, len(class_names) , flags.trainable_layers)
  (renset_model, trainables) = networks.init_model_and_freeze_layers(network_parameters)
  
  criterion = nn.CrossEntropyLoss()
  
  # Observe that all parameters are being optimized
  optimizer = optim.SGD(trainables, lr=flags.learning_rate,
                        momentum=flags.momentum, weight_decay=flags.weight_decay)
  
  # Decay LR by a factor of 0.1 every 7 epochs
  scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
  
  # Initialize training parameters
  training_args = (renset_model, criterion, optimizer, scheduler,
                   dataloders, dataset_sizes, flags)
  training.train_model(training_args)
  training.test_model(training_args)
  # visualize_model(renset_model)
  
  # plt.ioff()
  # plt.show()
  
  return (class_names, class_to_idx)

  
if __name__ == '__main__':
  """Train model"""
  flags = config.read_training_parameters()
  run_training(flags)

