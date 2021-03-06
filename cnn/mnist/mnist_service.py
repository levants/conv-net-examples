"""
Created on Nov 15, 2017

Service for model interface

@author: Levan Tsinadze
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io

import numpy as np
import torch
from PIL import Image, ImageOps
from flask import (Flask, request, render_template, json)

from cnn.mnist import (training_flags as config, mnist_interface as interface)
from cnn.mnist.cnn_files import files as _files
from cnn.mnist.network_model import LeNet

# from torch.autograd import Variable

IMAGE_SIZE = 28

n_input = 784
border_color = 'black'

# Initializes web container
app = Flask(__name__)


def recpgnize():
    image_data = request.data
    with Image.open(io.BytesIO(image_data)) as img:
        img = img.convert("L")  # convert into greyscale
        img = img.point(lambda i: i < 150 and 255)  # better black and white
        img = ImageOps.expand(img, border=8, fill=border_color)  # add padding
        img.thumbnail((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)  # resize back to the same size
        img.save(_files.data_file('http_img.png'))
        img = Image.fromarray(np.array(img), mode='L')
        resp = interface.run_model(model, img)

    return json.dumps(resp)


@app.route('/', methods=['GET', 'POST'])
def cnn_recognize():
    """Web method for recognition
      Returns:
        resp - recognition response
    """

    if request.method == 'POST':
        recpgnize()
    elif request.method == 'GET':
        resp = render_template("index.html")

    return resp


if __name__ == "__main__":
    flags = config.configure()

    global model
    model = LeNet()
    model.load_state_dict(torch.load(flags.weights, map_location=lambda storage, loc: storage))
    model.eval()
    app.run(host=flags.host, port=flags.port, threaded=True)
