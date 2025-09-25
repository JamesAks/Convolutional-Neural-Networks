import numpy as np
from Layer import Layer

class DOLayer(Layer):

    def __init__(self, probabilty: float = 0.5, train = True):

        self.prob = probabilty
        self.train = train
        super().__init__()

    
    def forward(self, input: np.ndarray) -> np.ndarray:

        if self.train:

            #Turning off nerurons in order to reduce overfitting. Essentially evaluating smaleer subtrees of network

            temp = np.random.rand(*input.shape)

            temp[temp < self.prob] = 0
            temp[temp >= self.prob] = 1

            self.index = temp

            return self.index * input
        
        else:

            # When testing we instead use the average by mulitplying the wieghts by the probabilty. Because the equation for fully connected is y = xw + b we can write it as
            # y = x(p*w) + b. We can then bring the p out and see that its the same as multiplying the input by the probabilty since its done uniformly.
            return input * self.prob

        
    def backward(self, error_grad: np.ndarray, learning_rate: float) -> np.ndarray:

        if self.train:

            return error_grad * self.index
        
        return error_grad
    





        
 