import numpy as np
from Layer import Layer


class Softmax(Layer):

    # The network needs to turn the float values its outputted into a prediction
    # Softmax utilises exponentials to do this

    def __init__(self, gpu: bool = False):
        super().__init__(gpu)

    def forward(self, input: np.ndarray) -> np.ndarray:

        print(f"input : { input.shape}")

        print()

        self.input = input

        exp_input = self.module.exp(input)

        self.output = exp_input/ exp_input.sum(axis = 1, keepdims= True)

        return self.output
    
    # def backward(self,error_grad: np.ndarray, learning_rate: float):

    #     return error_grad

    #     # n = np.size(self.output)

    #     # return np.dot((np.identity(n) - self.output.T) * self.output, error_grad)


    

