"""
Created on Jul 6, 2016

Utility module for evaluation files and directories

@author: Levan Tsinadze
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from __builtin__ import property
from os import walk
import os
import shutil
import types

import requests

# General parent directory for files
DATAS_DIR_NAME = 'datas'

# Files and directory constant parameters
PATH_FOR_PARAMETERS = 'trained_data'
PATH_FOR_TRAINING = 'training_data'
PATH_FOR_TESTING = 'test_data'
PATH_FOR_LOGGING = 'logs'
PATH_FOR_TEMPORARY = 'temp'
WEIGHTS_FILE = 'output_graph.pb'
LABELS_FILE = 'output_labels.txt'
TEST_IMAGES_DIR = 'test_images'
TEST_IMAGE_NAME = 'test_image'


def basename(full_file_path):
  """Gets base file name from full path
    Args:
      full_file_path
    Returns:
      base file name
  """
  return os.path.basename(full_file_path)


def file_exists(_file_path):
  """Validates if file exits
    Args:
      _file_path - file path
    Returns:
      validation result
  """
  return os.path.exists(_file_path)


def _init_dir(_dir_name):
  """Makes sure the folder exists on disk.

  Args:
    dir_name: Path string to the folder we want to create.
  """
  
  if not os.path.exists(_dir_name):
    os.makedirs(_dir_name)


def ensure_dir_exists(*dir_names):
  """Makes sure the folder exists on disk.
  Args:
    dir_name: Path string to the folder we want to create.
  """
  
  if dir_names:
    for _dir_name in dir_names:
      _init_dir(_dir_name)

      
def dirname(_file_path):
  """Gets containing directory for file
    Args:
      _file_path - path to file
    Returns:
      enclosing directory name
  """
  return os.path.dirname(_file_path)


def ensure_file_dir_exists(_file_path):
  """Makes sure the file enclosing folder exists on disk.
   Args:
    dir_name: Path string to the folder we want to create.
  """
  
  _dir_path = dirname(_file_path)
  ensure_dir_exists(_dir_path)

    
def join(path1, *path2):
  """Joins passed passes
    Args:
      path1 - first (parent) path
      path2 - second (child) path
    Returns:
      joined pathFS
  """
  return os.path.join(path1, *path2)


def move(src, dst):
  """Moves file from source directory to destination directory
    Args:
      src - source directory
      dst - destination directory
  """
  os.rename(src, dst)
 
  
def clear_existing_data(_dir_path):
  """Removes existing directory recursively
    Args:
      _dir_path - path to directory
  """
  
  if os.path.exists(_dir_path):
    if os.path.isdir(_dir_path):
      shutil.rmtree(_dir_path)
    else:
      os.unlink(_dir_path)


def rmtree(_dir_path):
  """Removes existing directory recursively
    Args:
      _dir_path - path to directory
  """
  clear_existing_data(_dir_path)
  

def rm(_file_path):
  """Removes existing file or directory recursively
    Args:
      _file_path - path to file
  """
  
  if os.path.exists(_file_path):
    if os.path.isfile(_file_path):
      os.remove(_file_path)
    else:
      clear_existing_data(_file_path)


def is_appropriated_file(file_name, file_exts=[]):
  """Validates file extension
    Args:
      file_name - file name
      file_exts - list of file extensions
    Returns:
      val_result - validation result
  """
  
  if len(file_exts) > 0:
    val_result = False
    for file_ext in file_exts:
      if file_name.endswith(file_ext):
        val_result = True
  else:
    val_result = True
  
  return val_result
      

def filter_files(file_names, file_exts=[]):
  """Filters files by extensions
    Args:
      file_names - list of file names
      file_exts - list of file extensions
    Returns:
      filtered_names - filtered file names
  """
  
  filtered_names = []
  
  if len(file_exts) > 0:
    filtered_names = [file_name for file_name in file_names \
                      if is_appropriated_file(file_name, file_exts)]
  else:
    filtered_names = file_names
  
  return filtered_names


def list_files(dir_path, file_exts=[]):
  """Lists all files from directory
    Args:
      dir_path - directory path
      file_exts - list of file extensions
    Returns:
      file_names - file from directories names
  """
  
  file_names = []
  for (_, _, filenames) in walk(dir_path):
    filtered_names = filter_files(filenames, file_exts=file_exts)
    file_names.extend(filtered_names)
  
  return file_names


def walk(dir_path, topdown=True, onerror=None, followlinks=False):
  """Walks through files and directories from passed path
    Args:
      dir_path - directory path to walk
      topdown - walk from top to down
      oneerr - action on error
      followlinks - follow symbolic links
    Returns:
      iteration tuple of -
        dirpath - directory path
        dirnames - sub directory names
        filenames - file names in path
  """
  return os.walk(dir_path, topdown=topdown, onerror=onerror, followlinks=followlinks)


def list_subdirs(dir_path):
  """Lists directory path for sub - directories
    Args:
     dir_path - directory path
    Returns:
      sub_dirs - list of sub - directories
  """
  
  sub_dirs = []
  
  for sub_dir_path in os.listdir(dir_path):
    full_sub_path = join(dir_path, sub_dir_path)
    if os.path.isdir(full_sub_path):
      sub_dirs.append((sub_dir_path, full_sub_path))
      
  return sub_dirs


def list_subfiles(dir_path, file_exts=[]):
  """Lists sub files
    Args:
      dir_path - directory path
      file_exts - file extensions
    Returns:
      sub_files - sub files with extension
  """
  
  sub_files = []
  
  if os.path.isdir(dir_path):
    for sub_file_path in os.listdir(dir_path):
      full_sub_path = join(dir_path, sub_file_path)
      if os.path.isfile(full_sub_path) and is_appropriated_file(sub_file_path, file_exts=file_exts):
        sub_files.append((sub_file_path, full_sub_path))
  
  return sub_files


def delete(fname):
  """Removes passed file from system
    Args:
      fname - file name
  """
  
  if os.path.exists(fname):
    os.remove(fname)


class files_and_path_utils(object):
  """Utility class for file management"""
  
  def __init__(self, parent_cnn_dir):
    self.path_to_cnn_directory = join(DATAS_DIR_NAME, parent_cnn_dir)
    
  def join_path(self, path_inst, *other_path):
    """Joins passed file paths
      Args:
        paths_inst - function to get path string
                     or path string itself
        *other_path - paths to join varargs
      Returns:
        result - joined paths
    """
    
    if isinstance(path_inst, types.StringType):
      init_path = path_inst
    else:
      init_path = path_inst()
    result = join(init_path, *other_path)
    
    return result

  def init_file_or_path(self, file_path):
    """Creates file if not exists
      Args:
        file_path - file path
      Returns:
        file_path - the same path
    """
    
    ensure_dir_exists(file_path)
    return file_path

  def join_and_init_path(self, path_inst, *other_path):
    """Joins and creates file or directory paths
      Args:
        path_inst - image path or function 
                    returning path
        other_path - varargs for other paths
                     or functions
      Returns:
        result - joined path
    """
    
    result = self.join_path(path_inst, *other_path)
    self.init_file_or_path(result)
    
    return result
  
  def get_current(self):
    """Gets current directory of script
      Returns:
        current_dir - project data files parent directoryFS
    """
      
    current_dir = os.path.dirname(os.path.realpath(__file__))
    
    dirs = os.path.split(current_dir)
    dirs = os.path.split(dirs[0])
    current_dir = dirs[0]
    
    return current_dir

  def get_data_general_directory(self):
    return self.join_path(self.get_current, self.path_to_cnn_directory)

  @property
  def general_dir(self):
    """Gets or creates directory for training / validation / test / trained 
      parameters
      Returns:
        path of training / validation / test / trained data and parameters
    """
    return self.get_data_general_directory()


class cnn_file_utils(files_and_path_utils):
  """Utility class for network files management"""
  
  def __init__(self, parent_cnn_dir):
    super(cnn_file_utils, self).__init__(parent_cnn_dir)
    
  def init_temp_directory(self):
    """Gets or creates directory for trained parameters
      Returns:
        path of training / validation / test / trained data and parameters
    """
      
    current_dir = self.join_path(self.get_data_general_directory, PATH_FOR_TEMPORARY)
    
    if not os.path.exists(current_dir):
        os.makedirs(current_dir)
    
    return current_dir  
  
  def init_files_directory(self):
    """Gets or creates directory for trained parameters
      Returns:
        path of training / validation / test / trained data and parameters
    """
      
    current_dir = self.join_path(self.get_data_general_directory, PATH_FOR_PARAMETERS)
    
    if not os.path.exists(current_dir):
        os.makedirs(current_dir)
    
    return current_dir
  
  def get_or_init_files_path(self):
    """Initializes trained files path
      Returns:
        trained files path
    """
    return self.join_path(self.init_files_directory, WEIGHTS_FILE)
      
  def get_or_init_labels_path(self):
    """Gets training data  / parameters directory path
      Returns:
        training data / parameters directory path
    """
    return self.join_path(self.init_files_directory, LABELS_FILE)

  def get_or_init_test_dir(self):
    """Gets directory for test images
      Returns:
        test image directory
    """
    
    current_dir = self.join_path(self.get_data_general_directory, TEST_IMAGES_DIR)
    
    if not os.path.exists(current_dir):
      os.mkdir(current_dir)  
    
    return current_dir
    
  def get_or_init_test_path(self):
    """Gets or initializes test image
      Returns:
        test image full path
    """
    return self.join_path(self.get_or_init_test_dir, TEST_IMAGE_NAME)
  
  def get_file_bytes_to_recognize(self, file_url):
    """Reads data binary from URL address
      Args:
        file_url - file URL address
      Returns:
        buff - file bytes buffer
    """
    
    response = requests.get(file_url, stream=True)
    buff = response.raw.read()
    del response
    
    return buff
    
  def get_file_to_recognize(self, file_url):
    """Downloads file from passed URL address
      Args:
        file_url - file URL address
      Returns:
        response - file as byte array
    """
    
    response = requests.get(file_url, stream=True)
    test_img_path = self.get_or_init_test_path()
    with open(test_img_path, 'wb') as out_file:
      shutil.copyfileobj(response.raw, out_file)
    del response
    
  @property
  def temp_dir(self):
    """Gets or creates directory for temporary files
      Returns:
        directory for temporary files
    """
    return self.init_temp_directory()
  
  def temp_file(self, _file_path):
    """Joins temporary directory path to passed file path
      Args:
        _file_path - temporary file path
      Returns:
        joined temporary directory and temporary file path
    """
    return self.join_path(self.temp_dir, _file_path)
  
  @property
  def model_dir(self):
    """Gets or creates directory for trained parameters
      Returns:
        current_dir - directory for trained parameters
    """
    return self.init_files_directory()

  def model_file(self, _file_path):
    """Joins models directory path to passed file path
      Args:
        _file_path - model file path
      Returns:
        joined models directory and model file path
    """
    return self.join_path(self.model_dir, _file_path)
  
  def get_training_directory(self):
    """Gets training data directory
    Returns:
      training data directory path
    """
    return self.join_path(self.get_data_general_directory, PATH_FOR_TRAINING)

  def get_data_directory(self):
    """Gets directory for training set and parameters
      Returns:
        directory for training set and parameters
    """
    return self.join_path(self.get_training_directory, self.path_to_training_photos)
  
  def get_or_init_data_directory(self):
    """Creates directory for training set and parameters"""
    
    dir_path = self.get_data_directory()
    ensure_dir_exists(dir_path)
  
  @property
  def data_dir(self):
    """Creates directory for training set and parameters
      Returns:
        _data_dir - data directory path
    """
    
    _data_dir = self.get_training_directory()
    ensure_dir_exists(_data_dir)
    
    return _data_dir
  
  def data_file(self, _file_path):
    """Joins data directory path to passed file path
      Args:
        _file_path - data file path
      Returns:
        joined data directory and data file path
    """
    return self.join_path(self.data_dir, _file_path)
  
  @property
  def test_dir(self):
    """Creates directory for application testing
      Returns:
        _test_dir - testing directory path
    """
    return self.join_path(self.get_data_general_directory, PATH_FOR_TESTING)
  
  @property
  def log_dir(self):
    """Creates directory for application logging
      Returns:
        dir_path - logging directory path
    """
    
    dir_path = self.join_path(self.get_data_general_directory, PATH_FOR_LOGGING)
    _init_dir(dir_path)
    
    return dir_path
    
