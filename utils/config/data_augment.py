"""
Created on Aug 3, 2017

Utility module for data augmentation

@author: Levan Tsinadze
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from imgaug import augmenters as iaa


def augment_dir(src_path, dest_path):
  """Augments files in directory
    Args:
      src_path - source directory path
      dest_path - destination directory path
  """
  
  
