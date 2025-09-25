import numpy as np
import cupy as cp
from Layer import Layer

class Flatten(Layer):

    def __init__(self, gpu: bool = False):
        super().__init__(gpu)


    def forward(self, input: np.ndarray | cp.ndarray) -> np.ndarray | cp.ndarray:


        self.batch_size,self.input_depth, self.input_width, self.input_height = input.shape
        

        return self.module.reshape(input,(self.batch_size, self.input_depth * self.input_width * self.input_height))
    
    def backward(self, error_grad: np.ndarray | cp.ndarray, learning_rate: float) -> np.ndarray | cp.ndarray:


        return self.module.reshape(error_grad,(self.batch_size, self.input_depth, self.input_width, self.input_height))


