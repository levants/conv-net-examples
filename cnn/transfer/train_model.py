"""
Created on Oct 16, 2017

Training module for network

@author: Levan Tsinadze
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import torch

from cnn.transfer import (network_model as networks,
                          dataset_config as datasets)

STATE_DICT_KEY = 'state_dict'


def set_flag(model, phase, scheduler):
  """Sets appropriated flag (training, evaluation) to model
    Args:
      model - network model
      phase - training / validation phase indicator
      scheduler - learning rate scheduler
  """

  if phase == datasets.TRAIN_PHASE:
    scheduler.step()
    model.train()  # Set model to training mode
  else:
    model.eval()  # Set model to evaluate mode


def train_phase(training_args, phase, use_gpu):
  """Trains instant epoch
    Args:
      training_args - training parameters
      phase - training / validation phase
      use_gpu - flag to use GPU devices
    Returns:
      epoch_acc - model accuracy per epoch
  """

  (model, criterion, optimizer,
   scheduler, dataloders, dataset_sizes, _) = training_args  
  
  set_flag(model, phase, scheduler)

  running_loss = 0.0
  running_corrects = 0

  # Iterate over data.
  for data in dataloders[phase]:

      (inputs, labels) = datasets.read_variables(data, use_gpu)
      # zero the parameter gradients
      optimizer.zero_grad()

      # forward
      outputs = model(inputs)
      (_, preds) = torch.max(outputs.data, 1)
      loss = criterion(outputs, labels)

      # backward + optimize only if in training phase
      if phase == datasets.TRAIN_PHASE:
        loss.backward()
        optimizer.step()

      # statistics
      running_loss += loss.data[0]
      running_corrects += torch.sum(preds == labels.data)

  epoch_loss = running_loss / dataset_sizes[phase]
  epoch_acc = running_corrects / dataset_sizes[phase]

  print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase,
                                             epoch_loss,
                                             epoch_acc))

  return epoch_acc
  

def train_model(training_args):
  """Trains network model
    Args:
      training_args - training arguments
  """
    
  (model, _, _, _, _, _, flags) = training_args
  since = time.time()

  best_model_wts = model.state_dict()
  use_gpu = torch.cuda.is_available()
  model = model.cuda() if use_gpu else model
  epochs = flags.epochs
  best_acc = 0.0
  for epoch in range(epochs):
    print('Epoch {}/{}'.format(epoch, epochs - 1))
    print('-' * 10)
    # Each epoch has a training and validation phase
    for phase in [datasets.TRAIN_PHASE, datasets.VAL_PHASE]:
      epoch_acc = train_phase(training_args, phase, use_gpu)
      is_best = phase == datasets.VAL_PHASE and epoch_acc > best_acc
      # deep copy the model
      (best_acc, best_model_wts) = (epoch_acc, model.state_dict())\
      if is_best else (best_acc, best_model_wts)
      networks.save_checkpoint(flags, {'epoch': epoch + 1,
                                       STATE_DICT_KEY: model.state_dict(),
                                       'best_prec1': best_acc},
                               is_best,
                               filename='checkpoint.pth.tar')
    print()

  time_elapsed = time.time() - since
  print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60,
                                                      time_elapsed % 60))
  print('Best val Acc: {:4f}'.format(best_acc))

  # load best model weights
  model.load_state_dict(best_model_wts)


def test_model(training_args):
  """Test trained model
    Args:
      training_args - training parameters
    Returns:
      epoch_acc - epoch accuracy
  """
  
  (model, criterion, optimizer,
   scheduler, dataloders, dataset_sizes, _) = training_args 
  
  set_flag(model, datasets.TEST_PHASE, scheduler)

  running_loss = 0.0
  running_corrects = 0

  use_gpu = pyconf.gpu_available()
  
  # Iterate over data.
  for data in dataloders[datasets.TEST_PHASE]:

      (inputs, labels) = datasets.read_variables(data, use_gpu)
      # zero the parameter gradients
      optimizer.zero_grad()

      # forward
      outputs = model(inputs)
      (_, preds) = torch.max(outputs.data, 1)
      loss = criterion(outputs, labels)

      # statistics
      running_loss += loss.data[0]
      running_corrects += torch.sum(preds == labels.data)

  epoch_loss = running_loss / dataset_sizes[datasets.TEST_PHASE]
  epoch_acc = running_corrects / dataset_sizes[datasets.TEST_PHASE]

  print('{} Loss: {:.4f} Acc: {:.4f}'.format(datasets.TEST_PHASE,
                                             epoch_loss,
                                             epoch_acc))

  return epoch_acc

