import numpy as np
from Layer import Layer
from Pooling import AveragePooling
from FullyConnected import FCLayer
from Activations import RELU6, Sigmoid
from BatchNormalisation import BatchNorm
from Flatten import Flatten

class SqueezeExciteLayer(Layer):

    def __init__(self, reduction_ratio: int = 4):

        # Certain channels may be more important (contribute more than the rest) than others.
        # We can use gloabl average pooling to find the coefficient for the channels .
        # And then use a fully connected layer to find out which channel is more important than the others.
        # This way the model may pay more attention to the more performative channels in learning

        self.reduction_ratio = reduction_ratio

        self.layers = ()

    def forward(self,input: np.ndarray):

        output = input
        self.input = input
        self.batch_size, self.inp_depth, self.inp_width, self.inp_height = input.shape

        if len(self.layers) == 0:

            self.layers = [

            AveragePooling(glob=True),
            Flatten(),
            FCLayer(int(self.inp_depth / self.reduction_ratio)),
            RELU6(),
            BatchNorm(),
            FCLayer(int(self.inp_depth)),
            Sigmoid()

        ]

        for layer in self.layers:

            output = layer.forward(output)

        self.scalars = output 
            
        for n in range(self.batch_size):

            for c in range(self.batch_size):

                input[n,c] *= self.scalars[n,c] 

        return input
    
    
    def backward(self, error_grad: np.ndarray, learning_rate: float) -> np.ndarray:

        output_grad = np.zeros_like(error_grad)
        for n in range(self.batch_size):
            for c in range(self.inp_depth):

                output_grad[n,c] = error_grad[n,c] * self.scalars[n,c]

        print(f"SE input: {self.input.shape }")
        print(f"SE error: {error_grad.shape}")

        sum_error = np.sum(error_grad * self.input, axis= (2,3))

        for layer in reversed(self.layers):


            sum_error = layer.backward(sum_error, learning_rate)

        output_grad += sum_error/(self.inp_width * self.inp_height)

        return output_grad
    
    def update(self, learning_rate: float = 0.09):

        for layer in self.layers:

            layer.update(learning_rate)
