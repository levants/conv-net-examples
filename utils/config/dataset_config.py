"""
Created on May 17, 2017

Configures data set generators

@author: Levan Tsinadze
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json

from keras.preprocessing.image import ImageDataGenerator


def init_data_generator(config_tuple, data_dir):
  """Data set generator training data reading and labeling
    Args:
      config_tuple - tuple of
        _preprocess_function - input preprocessor function,
        flags - command-line arguments
      data_dir - data set directory
    Returns:
      train_datagen - training data generator
      
  """
  
  (_preprocess_function, flags) = config_tuple
  rescale = 1. / 255 if _preprocess_function is None else None
  image_sizes = (flags.image_width, flags.image_height)
  batch_size = flags.batch_size
  # Configure test generator
  train_datagen = ImageDataGenerator(
      preprocessing_function=_preprocess_function,
      rescale=rescale,
      rotation_range=30,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True
  )
  # Configure test data flow
  train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=image_sizes,
    batch_size=batch_size,
  )
  
  return train_generator

def get_data_generators(config_tuple):
  """Data set generator for training and validation
    Args:
      config_tuple - tuple of
        _preprocess_function - input preprocessor function,
        flags - command-line arguments
    Returns:
      tuple of -
       train_generator - training data generator
       validation_generator - validation data generator
       test_generator - test data generator
  """
  
  (_, flags) = config_tuple
  # Configure training data flow
  train_generator = init_data_generator(config_tuple, flags.train_dir)
  # Configure validation data flow
  validation_generator = init_data_generator(config_tuple, flags.val_dir)
  # Configure test data flow
  test_generator = init_data_generator(config_tuple, flags.test_dir)
  
  return (train_generator, validation_generator, test_generator)

def dump_labels(label_indices, labels_path):
  """Saves labels in JSON file
    Args:
      label_indices - labels and associated indices
      labels_path - path to save labels
  """
  
  with open(labels_path, 'w') as json_file:
    json.dump(label_indices, json_file)

def retrieve_label_indices(json_path):
  """Retrieves label indices from passed data generator
    Args:
      json_path - path to retrieve labels
    Returns:
      index_labels - label indices dictionary
  """
  
  with open(json_path) as data_file:    
    label_indices = json.load(data_file)
  index_labels = {v: k for (k, v) in label_indices.iteritems()}
  
  return index_labels
  
  
