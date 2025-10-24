# Basic functionality of all classes. All Layers from the networks will override and inherit from the class.
import numpy as np 
import cupy as cp

class Layer:

    # Base Layer class. All layers in a network should inhgerit from this class.

    def __init__(self, gpu: bool = False):
        
        self.setModule(gpu)



    def forward(self, input: np.ndarray | cp.ndarray) -> np.ndarray | cp.ndarray:

        # Forward method defines the actions to be taken on the forward pass

        return input

    def backward(self, error_grad: np.ndarray | cp.ndarray , learning_rate: float) -> np.ndarray | cp.ndarray:

        # Backward method defines the actions to be taken in backpropagation of the network

        return error_grad
    
    def update(self, learning_rate: float = 0.01):

        # Used for updating parammeters

        return 
    
    def setModule(self, gpu: bool = False):

        if gpu:

            self.module = cp

        else:

            self.module = np