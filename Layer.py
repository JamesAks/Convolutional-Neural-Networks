# Basic functionality of all classes. All Layers from the networks will override and inherit from the class.
import numpy as np 
import cupy as cp

class Layer:

    def __init__(self, gpu: bool = False):
        
        self.setModule(gpu)



    def forward(self, input: np.ndarray | cp.ndarray) -> np.ndarray | cp.ndarray:

        return input

    def backward(self, error_grad: np.ndarray | cp.ndarray , learning_rate: float) -> np.ndarray | cp.ndarray:
        
        return error_grad
    
    def update(self, learning_rate: float = 0.01):

        # Used for updating parammeters

        return 
    
    def setModule(self, gpu: bool = False):

        if gpu:

            self.module = cp

        else:

            self.module = np