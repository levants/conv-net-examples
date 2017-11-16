"""
Created on Jun 3, 2017

Utility module fot network configuration

@author: Levan Tsinadze
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras import backend as K
from keras.layers import Dropout
from keras.losses import categorical_crossentropy
from keras.optimizers import (SGD, Adam)

from utils.logging import logger


SAME_PADDING = 'same'
RELU = 'relu'
SOFTMAX = 'softmax'
SIGMOID = 'sigmoid'
LINEAR = 'linear'

def init_bn_axis():
  """Initializes batch normalization axis
    Returns:
      bn_axis - batch normalization axis
  """
  
  if K.image_data_format() == 'channels_first':
    bn_axis = 1
  else:
    bn_axis = -1
  
  return bn_axis


def init_input_shape(image_size, channels=3):
  """Initializes network input shape
    Args:
      image_size - image width and height
    Returns:
      input_shape - input shape for network model
  """
  
  (img_rows, img_cols) = image_size
  if K.image_data_format() == 'channels_first':
    input_shape = (channels, img_rows, img_cols)
  else:
    input_shape = (img_rows, img_cols, channels)
  
  return input_shape


def init_input_shape_and_bn_axis(image_size):
  """Initializes network input shape and batch normalization axis
    Args:
      image_size - image width and height
    Returns:
      tuple of -
        input_shape - input shape for network model
        bn_axis - batch normalization axis
  """
  
  (img_rows, img_cols) = image_size
  if K.image_data_format() == 'channels_first':
    input_shape = (1, img_rows, img_cols)
    bn_axis = 1
  else:
    input_shape = (img_rows, img_cols, 1)
    bn_axis = 3
  
  return (input_shape, bn_axis)


def dropout(keep_prob, net, is_training):
  """Adds "dropout" layer to model
    Args:
      keep_prob - keep probability
      net - network model
      is_training - training flag
    Returns:
      net - network model
  """
  return Dropout(keep_prob)(net) if is_training else net


def retrieve_optimizer(flags):
  """Initializes training cost function
    Args:
      flags - training configuration flags
    Returns:
      optimizer - optimizer function
  """
  
  lrate = flags.learning_rate
  momentum = flags.momentum
  decay = lrate / flags.epochs
  if flags.optimizer == 'adam':
    optimizer = Adam(lr=lrate, decay=decay)
    logger.print_directly(flags, 'ADAM optimizer was configured')
  else:
    optimizer = SGD(lr=lrate, momentum=momentum, decay=decay, nesterov=False)
    logger.print_directly(flags, 'SGD optimizer was configured')
  
  return optimizer


def compile_network_model(model, optimizer, loss_func):
  """Compiles network for training
    Args:
      model - network model for compilation
      optimizer - cost function
      loss_func - loss function
  """
  model.compile(optimizer=optimizer,
                loss=loss_func, metrics=['accuracy'])
  

def compile_network(model, optimizer):
  """Compiles network for training
    Args:
      model - network model for compilation
      optimizer - cost function
  """
  compile_network_model(model, optimizer, categorical_crossentropy)
  
  
def read_training_history(history):
  """Reads training history
    Args:
      history - training history
    Returns:
      tuple of -
        acc - training accuracy by epochs
        val_acc - validation accuracy by epochs
        loss - training loss by epochs
        val_loss validation loss by epochs
        epochs = range of epochs
  """
  
  acc = history.history['acc']
  val_acc = history.history['val_acc']
  loss = history.history['loss']
  val_loss = history.history['val_loss']
  epochs = range(len(acc))
  
  return (acc, val_acc, loss, val_loss, epochs)

