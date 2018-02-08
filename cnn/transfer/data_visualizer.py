"""
Created on Oct 16, 2017

Model and data visualizer

@author: Levan Tsinadze
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import torch

from cnn.transfer import dataset_config  as datasets


def imshow(inp, title=None):
    """Show image for Tensor."""

    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


def visualize_model(model, dataloders, class_names, num_images=6):
    """Visualizes data
      Args:
        model - network model
        class_names - label names
        use_gpu - flag for GPU devices
        num_images - number of images
    """

    images_so_far = 0
    _ = plt.figure()

    for (_, data) in enumerate(dataloders[datasets.TEST_PHASE]):

        use_gpu = torch.cuda.is_available()
        (inputs, _) = datasets.read_variables(data, use_gpu)

        outputs = model(inputs)
        (_, preds) = torch.max(outputs.data, 1)

        for j in range(inputs.size()[0]):
            images_so_far += 1
            ax = plt.subplot(num_images // 2, 2, images_so_far)
            ax.axis('off')
            ax.set_title('predicted: {}'.format(class_names[preds[j]]))
            imshow(inputs.cpu().data[j])

            if images_so_far == num_images:
                return
