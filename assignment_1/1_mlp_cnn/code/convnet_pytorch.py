"""
This module implements a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn

class ResnetBlock(nn.Module):
  def __init__(self, channel_in, channel_out, kernel, stride, padding) -> None:
    super(ResnetBlock, self).__init__()
    self.in_size = channel_in
    self.out_size = channel_out
    self.kernel = kernel
    self.stride = stride
    self.padding = padding

    self.batchnorm = nn.BatchNorm2d(channel_in)
    self.relu = nn.ReLU()
    self.conv = nn.Conv2d(channel_in, channel_out, kernel, stride, padding)

  def forward(self, input):
    X = input
    res_X = self.batchnorm(X)
    res_X = self.relu(res_X)
    res_X = self.conv(res_X)

    return X + res_X



class ConvNet(nn.Module):
    """
    This class implements a Convolutional Neural Network in PyTorch.
    It handles the different layers and parameters of the model.
    Once initialized an ConvNet object can perform forward.
    """
    
    def __init__(self, n_channels, n_classes):
        """
        Initializes ConvNet object.
        
        Args:
          n_channels: number of input channels
          n_classes: number of classes of the classification problem
          
        
        TODO:
        Implement initialization of the network.
        """
        super(ConvNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        
        self.layers = nn.Sequential(
          nn.Conv2d(n_channels, 64, (3, 3), 1, 1),
          ResnetBlock(64, 64, (3,3), 1, 1),
          nn.Conv2d(64, 128, (1,1), 1, 0),
          nn.MaxPool2d((3,3), 2, 1),
          ResnetBlock(128, 128, (3,3), 1, 1),
          ResnetBlock(128, 128, (3,3), 1, 1),
          nn.Conv2d(128, 256, (1,1), 1, 0),
          nn.MaxPool2d((3,3), 2, 1),
          ResnetBlock(256, 256, (3,3), 1, 1),
          ResnetBlock(256, 256, (3,3), 1, 1),
          nn.Conv2d(256, 512, (1,1), 1, 0),
          nn.MaxPool2d((3,3), 2, 1),
          ResnetBlock(512, 512, (3,3), 1, 1),
          ResnetBlock(512, 512, (3,3), 1, 1),
          nn.MaxPool2d((3,3), 2, 1),
          ResnetBlock(512, 512, (3,3), 1, 1),
          ResnetBlock(512, 512, (3,3), 1, 1),
          nn.MaxPool2d((3,3), 2, 1),
          nn.BatchNorm2d(512),
          nn.ReLU(),
          nn.Flatten(1),
          nn.Linear(512, 10)
        )
    
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
        
        out = self.layers(x)
        return out
