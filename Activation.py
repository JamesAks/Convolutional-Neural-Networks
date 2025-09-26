import numpy as np
import cupy as cp
from Layer import Layer

#TO-DO: Annontations

class Activation(Layer):

    def __init__(self, activation, activation_prime,):

        # Activation function for propragation.
        self.activation = activation
        # Backwards propagation activation is the inverse of the forward which means they have to be individually defined.
        self.activation_prime = activation_prime

        super().__init__()


    def forward(self, input: np.ndarray | cp.ndarray):

        self.input = input

        return self.activation(self.input)
    

    def backward(self, error_grad: np.ndarray | cp.ndarray, learning_rate: float):
        
        return self.activation_prime(error_grad)



