"""
Created on Jun 17, 2016
Image processing before recognition
@author: Levan Tsinadze
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io

from PIL import Image, ImageOps
from cnn.mnist.cnn_files import files as _files

IMAGE_SIZE = 28

n_input = 784
border_color = 'black'


def read_image(image_file_path=None, image_data=None):
	"""Reads image from binaries or path
		Args:
			image_file_path - image file path
			image_data - binary image
		Retunrs:
			img - image matrix
	"""
	if image_data:
		img = Image.open(io.BytesIO(image_data))
	else:
		img = Image.open(image_file_path)
	
	return img


def read_input_file(image_data=None):
	"""Reads image file to tensor
		Args:
			image_data - binary image
		Returns:
			img_array - array of image pixels
	"""
	with Image.open(io.BytesIO(image_data)) as img:
		img = img.convert("L")  # convert into greyscale
		img = img.point(lambda i: i < 150 and 255)  # better black and white
		img = ImageOps.expand(img, border=8, fill=border_color)  # add padding
		img.thumbnail((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)  # resize back to the same size
		img.save(_files.data_file('http_img.jpg'))
		resp = interface.run_model(model, img)
		
		return img

