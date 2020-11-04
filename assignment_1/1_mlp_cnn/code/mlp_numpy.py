"""
This module implements a multi-layer perceptron (MLP) in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from typing import List, Tuple

from modules import *

import numpy as np
import numpy.random as rnd

class MLP(object):
    """
    This class implements a Multi-layer Perceptron in NumPy.
    It handles the different layers and parameters of the model.
    Once initialized an MLP object can perform forward and backward.
    """

    def __init__(self, n_inputs, n_hidden, n_classes):
        """
        Initializes MLP object.

        Args:
          n_inputs: number of inputs.
          n_hidden: list of ints, specifies the number of units
                    in each linear layer. If the list is empty, the MLP
                    will not have any linear layers, and the model
                    will simply perform a multinomial logistic regression.
          n_classes: number of classes of the classification problem.
                     This number is required in order to specify the
                     output dimensions of the MLP

        TODO: Implement initialization of the network.
        """

        # TEST: asdasd
        self.hidden_layers = self._init_hidden(n_inputs, n_hidden)

        last_dim: int = n_inputs if len(self.hidden_layers) == 0 else self.hidden_layers[-1].params['weights'].shape[1]
        self.hidden2out = LinearModule(last_dim, n_classes)

        # Activations
        self.elu = ELUModule()
        self.softmax = SoftMaxModule()

    def _init_hidden(self, input_dim: int, n_hidden: List[int]) -> List[LinearModule]:
        """Initializes a list of tuples: each tuple contains weights and gradients"""
        hidden_layers: List[LinearModule] = []

        for hidden_dim in n_hidden:
            # Input size is either the input_dim, or the previous layer's input size
            in_size: int = input_dim if len(hidden_layers) == 0 else hidden_layers[-1].params['weights'].shape[1]
            hidden_module = LinearModule(in_size, hidden_dim)
            hidden_layers.append(hidden_module)                      

        return hidden_layers


    def forward(self, x):
        """
        Performs forward pass of the input. Here an input tensor x is transformed through
        several layer transformations.

        Args:
          x: input to the network
        Returns:
          out: outputs of the network
        """
        # For consistency, let's treat `x` let the matrix it probably is `X`
        X = x 

        for hidden_weights in self.hidden_layers:
          X = hidden_weights.forward(X)
          X = self.elu.forward(X)
        
        X = self.hidden2out.forward(X)
        out = self.softmax.forward(X)

        return out

    def backward(self, dout):
        """
        Performs backward pass given the gradients of the loss.

        Args:
          dout: gradients of the loss

        TODO:
        Implement backward pass of the network.
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################
        raise NotImplementedError
        ########################
        # END OF YOUR CODE    #
        #######################

        return
