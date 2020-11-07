"""
This module implements various modules of the network.
You should fill in code into indicated sections.
"""
import numpy as np


class LinearModule(object):
    """
    Linear module. Applies a linear transformation to the input data.
    """
    
    def __init__(self, in_features, out_features):
        """
        Initializes the parameters of the module.
    
        Args:
          in_features: size of each input sample
          out_features: size of each output sample
    
        Also, initialize gradients with zeros.
        """

        self.in_features = in_features
        self.out_features = out_features

        # TEST
        self.params = {
          'weights': np.random.standard_normal((in_features, out_features)), # type: ignore
          'bias': np.zeros(out_features)
        }

        # TEST
        self.grads = {
          'weights': np.zeros((out_features, in_features)),
          'bias': np.zeros(out_features)
        }

        self.intermediary_activations_input = None
        self.intermediary_activations = None
    
    def forward(self, x):
        """
        Forward pass.
    
        Args:
          x: input to the module
        Returns:
          out: output of the module
        
        # TEST
        
        """
        self.intermediate_inputs = x
        out = x @ self.params['weights']
        self.intermediate_activations = out
        
        return out

    def zero_grad(self):
        """Clears out gradient (and intermediate activations)"""
        self.gradients = np.zeros((self.in_features, self.out_features))
        self.intermediate_inputs = None
  
    def backward(self, dout):
        """
        Backward pass.
    
        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module
    
        # TEST
        """
        self.grads['weights'] = dout @ self.intermediate_inputs
        self.grads['bias'] = dout
        grad_X = dout @ self.params['weights']
        dx = dout @ grad_X
        return dx

class SoftMaxModule(object):
    """
    Softmax activation module.
    """

    def __init__(self) -> None:
      self.intermediary_activations = None
    
    def forward(self, x) -> np.array:
        """
        Forward pass.
        Args:
          x: input to the module
        Returns:
          out: output of the module
        """

        # Tiled version of input
        x_max = x.max(1)
        x_max_tricked = x - np.tile(x_max, (x.shape[1], 1)).T
        x_totals = np.exp(x_max_tricked).sum(1) # type: ignore
        x_probs = np.exp(x_max_tricked) / np.tile(x_totals, (x.shape[1], 1)).T # type: ignore
        
        out = x_probs
        self.intermediary_activations = out
        return out

    def backward(self, dout):
        """
        Backward pass.
        Args:
          dout: gradients of the previous modul
        Returns:
          dx: gradients with respect to the input of the module
    
        # TEST
        """
        # Get activations from forward pass
        s = self.intermediary_activations
        batch_identity = np.dstack([np.identity(s.shape[0])] * 3).resize(s.shape[0], s.shape[1], s.shape[1])
        s_tiled = np.tile(s, (s.shape[1], 1, 1)).reshape(s.shape[s], s.shape[1], s.shape[1])
        applied_s = batch_identity - s_tiled

        grad_s = s @ applied_s

        dx = dout @ grad_s
        return dx


class CrossEntropyModule(object):
    """
    Cross entropy loss module.
    """
    
    def forward(self, x, y):
        """
        Forward pass.
        Args:
          x: input to the module
          y: labels of the input
        Returns:
          out: cross entropy loss
    
        TODO:
        Implement forward pass of the module.
        """

        # TEST
        elementwise_prod = - np.log(x) * y # type: ignore
        batch_out: np.array = elementwise_prod.sum(1) #TEST
        out = batch_out.mean(0)
        
        return out
    
    def backward(self, x, y):
        """
        Backward pass.
        Args:
          x: input to the module
          y: labels of the input
        Returns:
          dx: gradient of the loss with the respect to the input x.
    
        Implement backward pass of the module.
        # TEST
        """
        dx= -y * 1 / x
        return dx


class ELUModule(object):
    """
    ELU activation module.
    """

    def __init__(self) -> None:
      self.intermediary_activations_input = None
      self.intermediary_activations = None
    
    def forward(self, x: np.array):
        """
        Forward pass.

        Args:
          x: input to the module
        Returns:
          out: output of the module

        # TEST
        """
        inp = x.copy()
        self.intermediary_activations_input = x.copy()

        idxs = np.where(x < 0)

        out = inp
        out[idxs] = np.exp(inp[idxs]) - 1 # type: ignore
        
        self.intermediary_activations = out

        return out
    
    def backward(self, dout):
        """
        Backward pass.
        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module

        # TEST
        """
        idxs = np.where(self.intermediary_activations_input < 0)

        x = np.ones_like(self.intermediary_activations_input)
        x[idxs] = np.exp(self.intermediary_activations_input[idxs]) # type: ignore

        dx = dout * x
        return dx
