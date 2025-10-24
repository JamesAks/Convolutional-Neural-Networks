import numpy as np
import cupy as cp
from Layer import Layer

# TO-DO: Pretty much the same as 1x1 convolution could maybe remove

class FCLayer(Layer):

    # Dense or fully-connected Layer connects each input neuron to an output neuron in the hidden layerby a weight.
    # The output at each output neuron is equivalant to the sum of all the input neurons multiplied by the respective weights that connect them to that output neuron plus a bias
    # Y = X . W + B

    def __init__(self, output_size: int, momentum: float = 0.09, gpu: bool = False):

        
        self.output_size = output_size

        self.biases = np.array([])
        self.weights = np.array([])

        self.momentum = momentum

        self.velocity_biases =np.array([])
        self.velocity_weights = np.array([])

        super().__init__(gpu)

    def forward(self, input: np.ndarray | cp.ndarray):

        self.input = input
        self.input_shape = input.shape
        # self.input = input.reshape((input.size,1))
       # print(f"input size {input.size}")
        #print(f"Full input: {input}")
        #print(f" Input shape: {input.shape}")
        

        if len(self.weights) == 0 or len(self.biases == 0):

            self.weights = self.module.random.rand( self.input_shape[1],self.output_size)
            self.biases = self.module.random.rand(1,self.output_size)

            self.velocity_weights = self.module.zeros(self.weights.shape)
            self.velocity_biases = self.module.zeros(self.biases.shape)

        # No need to flatten as technically done by the final conolutional layer

        out = self.module.dot(self.input,self.weights)
        
        #print(f"dot: {self.output.size}")
        #print(self.output.shape)
        # print(f"Weights: {self.weights[0]}")
        # print(f"Biases: {self.biases}")
        # print(f"Output: {self.output}")

        #self.output = np.dot(self.weights, self.input) + self.biases
        #print(f" Full output: {self.output}")

        return out + self.biases
    
    def backward(self,error_grad: np.ndarray, learning_rate: float):

        self.weights_gradient = self.module.dot(self.input.T, error_grad)
        
        input_gradient = self.module.dot(error_grad, self.weights.T)

        #print(input_gradient.shape)

        self.biases_gradient = self.module.sum(error_grad, axis = 0) * learning_rate

        return input_gradient
    

    def update(self, learning_rate: float = 0.09):

        self.velocity_weights = (self.momentum * self.velocity_weights) - (self.weights * learning_rate)
        self.velocity_biases = (self.momentum * self.velocity_biases) - (self.biases_gradient * learning_rate)

        self.weights += self.velocity_weights
        self.biases += self.velocity_biases






        







        
