"""
This module implements various modules of the network.
You should fill in code into indicated sections.
"""
import numpy as np

DIVISION_CONST = 0.00001

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

        self.params = {
          'weights': np.random.normal(loc=0, scale=0.0001, size=(in_features, out_features)), # type: ignore
          'bias': np.zeros(out_features)
        }

        self.grads = {
          'weights': np.zeros((out_features, in_features)),
          'bias': np.zeros(out_features)
        }

        self.intermediary_activations_input = None
        self.intermediary_activations = None

        self.test = ''
    
    def forward(self, x):
        """
        Forward pass.
    
        Args:
          x: input to the module
        Returns:
          out: output of the module
        
        """
        self.intermediate_inputs = x.copy()
        out = x @ self.params['weights']
        self.intermediate_activations = out.copy()
        
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
        """
        self.grads['weights'] = (dout.T @ self.intermediate_inputs).T
        self.grads['bias'] = dout.T @ np.ones(dout.shape[0])
        grad_X = dout @ self.params['weights'].T
        dx = grad_X
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
    
        # TODO: Convert / understand the einsum better
        """
        s = self.intermediary_activations
        self_activations = np.einsum('ij,ik->ijk', s, s)
        batched_input = np.einsum('ij,jk->ijk', s, np.eye(s.shape[1], s.shape[1]))
        ds = batched_input - self_activations
        dx = np.einsum('ijk,ik->ij', ds, dout)
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
        """
        elementwise_prod = - np.log(x + DIVISION_CONST) * (y) # type: ignore
        batch_out: np.array = elementwise_prod.sum(1) 
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
        """
        dx= -y / ((x * x.shape[0]) + DIVISION_CONST)
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
        """
        idxs = np.where(self.intermediary_activations_input < 0)

        x = np.ones_like(self.intermediary_activations_input)
        x[idxs] = np.exp(self.intermediary_activations_input[idxs]) # type: ignore

        dx = dout * x
        return dx
