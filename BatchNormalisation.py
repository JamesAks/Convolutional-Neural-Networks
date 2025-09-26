import numpy as np
import cupy as cp
from Layer import Layer
import math

class BatchNorm(Layer):

    def __init__(self, gpu: bool = False):
        super().__init__(gpu)

    def forward(self, input: np.ndarray | cp.ndarray):

        self.input = input

        #print(f"Batch input: {self.input[0:3]}")

        mean = float(self.module.mean(self.input))
        #print(f"Mean: {mean}")
    
        var = float(self.module.var(self.input))

        self.output = (self.input - mean)/self.module.sqrt((var**2) + (1*10**-5))

        #print(f"Batch output{self.output[0:3]}")

        return self.output

        











   