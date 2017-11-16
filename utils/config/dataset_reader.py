"""
Created on May 17, 2017

Configures data set generators

@author: Levan Tsinadze
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.applications.inception_v3 import preprocess_input

from cnn.transfer import training_flags
from utils.config import dataset_config as datasets
from utils.files import file_utils as files


def _init_preprocess_input(_preprocess_input):
  """Initializes pre-process input function
    Args:
      _preprocess_input - function for input data pre-processing 
                          before neural network
    Returns:
      preprocess_func - initialized input pre-processing function
  """
  
  if _preprocess_input is None:
    preprocess_func = preprocess_input
  else:
    preprocess_func = _preprocess_input
    
  return preprocess_func

def get_labels(flags, _preprocess_input=None):
  """Gets data from labels
    Args:
      flags - configuration flags
    Returns:
      dictionary of indices with associated labels
  """
  
  preprocess_func = _init_preprocess_input(_preprocess_input)
  config_tuple = (preprocess_func, flags)
  data_generator = datasets.init_data_generator(config_tuple, flags.test_dir)
  
  return data_generator.class_indices

def get_num_classes(flags):
  """Gets number of classes of network model
    Args:
      flags - configuration flags
    Returns:
      num_classes - number of classes for model
  """
  class_indices = get_labels(flags)
  num_classes = 0 if class_indices is None else len(class_indices)
  
  return num_classes

def save_labels(flags, label_indices):
  """Saves labels ijn to the JSON file
    Args:
      flags - configuration flags
      label_indices - dictionary of labels and indices
  """
  
  weights_dir_path = files.dirname(flags.weights)
  labels_path = files.join(weights_dir_path, training_flags.MODEL_LABELS)
  files.clear_existing_data(labels_path)
  datasets.dump_labels(label_indices, labels_path)
  
def _init_labels_file(flags):
  """Initializes labels JSON file path
    Args:
      flags - configuration flags
    Returns:
      label JSON file path
  """
  return files.join(files.dirname(flags.weights), training_flags.MODEL_LABELS)

def read_label_indices(flags):
  """Retrieves labels and indices from JSON file
    Args:
      flags - configuration flags
    Returns:
      index_labels - dictionary of indices and labels
  """
  
  json_path = flags.labels if flags.labels else _init_labels_file(flags)
  index_labels = datasets.retrieve_label_indices(json_path)
  
  return index_labels
  
if __name__ == '__main__':
  """Explore data generator and labeling"""
  
  flags = training_flags.read_training_parameters()
  class_indices = get_labels(flags)
  if class_indices is not None and len(class_indices) > 0:
    for (class_name, class_index) in  class_indices.iteritems():
      print(class_name, class_index)
