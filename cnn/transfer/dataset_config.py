"""
Created on Oct 16, 2017

Module for training data configuration

@author: Levan Tsinadze
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import (datasets, transforms)
from torchvision.datasets.folder import find_classes

# Data directories names
TRAIN_PHASE = 'train'
VAL_PHASE = 'val'
TEST_PHASE = 'test'
DATA_PHASES = [TRAIN_PHASE, VAL_PHASE, TEST_PHASE]


def read_labels(dir_path):
    """Reads labels and label indices from directory
      Args:
        dir_path - path to data directory
      Returns:
        tuple of -
          classes - array of labels
          class_to_idx - dictionary of indices keyed by classes
    """
    return find_classes(dir_path)


def read_variables(data, use_gpu):
    """Reads data tuple as variables for training / validation batch
      Args:
        data - data tuple
        use_gpu - system GPU flag
      Returns:
        tuple of -
          inputs - input tensors
          labels - label tensors
    """

    # get the inputs
    (inputs, labels) = data

    # wrap them in Variable
    if use_gpu:
        (input_vars, label_vars) = (Variable(inputs.cuda()), Variable(labels.cuda()))
    else:
        (input_vars, label_vars) = (Variable(inputs), Variable(labels))

    return (input_vars, label_vars)


def init_validation_transform():
    """Initializes transformation function
      Returns:
        chain of transformation functions
    """
    return transforms.Compose([transforms.Scale(256),
                               transforms.CenterCrop(224),
                               transforms.ToTensor(),
                               transforms.Normalize([0.485, 0.456, 0.406],
                                                    [0.229, 0.224, 0.225])])


validation_trandorms = init_validation_transform()


def init_datasets(flags):
    """Initializes training  / test / validation data
      Args:
        flags - configuration parameters
      Returns:
        tuple of -
          dataloders - training / validation / test data loaders
          dataset_sizes - training and validation data size
          class_names - class names
    """

    train_transforms = transforms.Compose([transforms.RandomSizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])
    image_datasets = {TRAIN_PHASE: datasets.ImageFolder(flags.train_dir, train_transforms),
                      VAL_PHASE: datasets.ImageFolder(flags.val_dir, validation_trandorms),
                      TEST_PHASE: datasets.ImageFolder(flags.test_dir, validation_trandorms)}
    dataloders = {x: DataLoader(image_datasets[x], batch_size=flags.batch_size,
                                shuffle=x == TRAIN_PHASE, num_workers=flags.num_workers)
                  for x in DATA_PHASES}
    dataset_sizes = {x: len(image_datasets[x]) for x in DATA_PHASES}
    class_names = image_datasets[TRAIN_PHASE].classes
    class_to_idx = image_datasets[TRAIN_PHASE].class_to_idx

    return (dataloders, dataset_sizes, class_names, class_to_idx)
