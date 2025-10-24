import numpy as np
import cupy as cp
from Layer import Layer


class BatchNorm(Layer):

    # Keeps the activations within a manageble range and stops vaninshing or exploding gradients. 
    # By normalizing data internal covariate shift is addressed and allows the models to stabilize the training and speed up the convergence.

    def __init__(self, gpu: bool = False):
        super().__init__(gpu)

    def forward(self, input: np.ndarray | cp.ndarray):

        self.input = input

        mean = float(self.module.mean(self.input))
        var = float(self.module.var(self.input))

        # Normalization = (input - mean)/ sqare_root( variance^2 + e) where e is a small value to handle if the denominator becomes zero

        self.output = (self.input - mean)/self.module.sqrt((var**2) + (1*10**-5))

        return self.output
    
    # TO:DO implement backward

        











   