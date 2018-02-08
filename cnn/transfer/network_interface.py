"""
Created on Oct 17, 2017

Interface for transfer / fine-tuned model

@author: Levan Tsinadze
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from io import BytesIO

from PIL import Image
from torch.autograd import Variable

from cnn.transfer.dataset_config import validation_trandorms as transforms


def preprocess_and_run(model, img):
    """Prepares image and evaluated network model
      Args:
        model - network model
        img - binary image
      Returns:
        predictions - network output
    """

    image_tensor = transforms(img)
    input_batch = image_tensor.unsqueeze(0)
    input_var = Variable(input_batch, volatile=True)
    predictions = model(input_var)

    return predictions


def run_on_path(model, image_path):
    """Runs model on image path
      Args:
        model - network model
        image_path - path to image file
      Returns:
        predictions - predictions on image
    """

    with Image.open(image_path) as img:
        predictions = preprocess_and_run(model, img)

    return predictions


def run_model(model, image_data):
    """Runs model for image data
      Args:
        model - network model
        image_data - image data
      Returns:
        predictions - network outputs
    """

    with Image.open(BytesIO(image_data)) as img:
        predictions = preprocess_and_run(model, img)

    return predictions
