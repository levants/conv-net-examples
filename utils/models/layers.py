"""
Created on Oct 16, 2017

Flatten layer for PyTorch models

@author: Levan Tsinadze
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter


class Flatten(nn.Module):
    """Module to flatten network layer"""

    def __init__(self, out_dim, input_dim=None):
        super(Flatten, self).__init__()
        self.out_dim = out_dim
        self.input_dim = input_dim
        self.fc_layer = nn.Linear(1, 1)
        self._no_weights_adjasted = True
        self._apply_fns = []
        if self.input_dim:
            input_rand = Variable(torch.randn(1, input_dim))
            self.forward(input_rand)

    def _apply(self, fn):
        """Saves passed function for initialized linear layer
          Args:
            fn - functions to apply
          Returns:
            current object instance
        """

        self._apply_fns.append(fn)
        _apply_result = super(Flatten, self)._apply(fn)

        return _apply_result

    def _apply_postfactum(self):
        """Applies functions from module"""

        for fn in self._apply_fns:
            self.fc_layer._apply(fn)

    def calculate_total(self, x):
        """Calculates total dimension of tensor
          Args:
            x - tensor to calculate total dimension
        """

        if self._no_weights_adjasted:
            self.flatten_dim = 0 if x is None else np.prod(x.size()[1:])
            self.fc_layer = nn.Linear(self.flatten_dim, self.out_dim)
            self._apply_postfactum()
            self._no_weights_adjasted = False

    def forward(self, input_tensor):
        """Flattens passed tensor and calculates input dimension
          Args:
            input_tensor - input tensor
          Returns:
            x - flattened tensor
        """

        self.calculate_total(input_tensor)
        x = input_tensor.view(input_tensor.size(0), self.flatten_dim)
        x = self.fc_layer(x)

        return x

    def load_state_dict(self, state_dict):
        """Copies parameters and buffers from :attr:`state_dict` into
        this module and its descendants. The keys of :attr:`state_dict` must
        exactly match the keys returned by this module's :func:`state_dict()`
        function.

        Arguments:
            state_dict (dict): A dict containing parameters and
                persistent buffers.
        """
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                raise KeyError('unexpected key "{}" in state_dict'
                               .format(name))
            if isinstance(param, Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
                if name == 'fc_layer':
                    self.fc_layer = nn.Linear(self.flatten_dim, self.out_dim)

            try:
                own_state[name].copy_(param)
            except:
                print('While copying the parameter named {}, whose dimensions in the model are'
                      ' {} and whose dimensions in the checkpoint are {}, ...'.format(
                    name, own_state[name].size(), param.size()))
                raise

        missing = set(own_state.keys()) - set(state_dict.keys())
        if len(missing) > 0:
            raise KeyError('missing keys in state_dict: "{}"'.format(missing))

    def __repr__(self):
        return self.__class__.__name__ + ' (None -> ' \
               + str(self.out_dim) + ')'
