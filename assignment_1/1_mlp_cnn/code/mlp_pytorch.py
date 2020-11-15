"""
This module implements a multi-layer perceptron (MLP) in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from typing import List, Union

import torch.nn as nn
from torch.nn.modules.activation import ELU

filter_linear_layers = lambda layers: [layer for layer in layers[::-1] if isinstance(layer, nn.Linear)] 


class MLP(nn.Module):
    """
    This class implements a Multi-layer Perceptron in PyTorch.
    It handles the different layers and parameters of the model.
    Once initialized an MLP object can perform forward.
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
    
        TODO:
        Implement initialization of the network.
        """
        super(MLP, self).__init__()
        hidden_layers = self._init_hidden(n_inputs, n_hidden)
        last_dim: int = n_inputs if len(hidden_layers) == 0 else filter_linear_layers(hidden_layers)[0].weight.shape[1]

        self.layers = nn.Sequential(
          *hidden_layers,
          nn.Linear(last_dim, n_classes),
          nn.Softmax(1)
        )
        
    def _init_hidden(self, input_dim: int, n_hidden: List[int]) -> List[Union[nn.Module, ELU]]:
        """Initializes a list of tuples: each tuple contains weights and gradients"""
        hidden_layers: List[Union[nn.Module, ELU]] = []

        for hidden_dim in n_hidden:
            # Input size is either the input_dim, or the previous layer's input size
            in_size: int = input_dim if len(hidden_layers) == 0 else filter_linear_layers(hidden_layers)[0].weight.shape[1]
            hidden_module = nn.Linear(in_size, hidden_dim)
            hidden_layers.append(hidden_module)
            hidden_layers.append(ELU())

        return hidden_layers

    
    def forward(self, x):
        """
        Performs forward pass of the input. Here an input tensor x is transformed through
        several layer transformations.
        
        Args:
          x: input to the network
        Returns:
          out: outputs of the network
        
        TODO:
        Implement forward pass of the network.
        """
        
        out = self.layers.forward(x)

        return out
