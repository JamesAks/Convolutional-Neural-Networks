import numpy as np
import cupy as cp
from Activation import Activation

# Activations are applied to the network to add non-linearity. This allows the netwrok to learn more complex patterns
# For most of the network layers we us a RELU6 Activation that clamps the values between 0 and 6.

class Tanh(Activation):

    def __init__(self):
            
        def tanh(x):
                
             return np.tanh(x)
        
        def inverse_tanh(x):

            return

            # Super method to intialise the Activation with the functions
        super().__init__(tanh,inverse_tanh)

        def test():

            return

    


class RELU6(Activation):

    def __init__(self, gpu: bool = False):

        self.setModule(gpu)
        self.matrix = []
        
        def activation(input: np.ndarray | cp.ndarray):

            # RELU6 returns 0 is the input is less than 0 and clamps the max value to 6
            input[input < 0] = 0
            input[input > 6] = 6

            self.input = input

            return self.input

        def activation_prime(output_gradient: np.ndarray | cp.ndarray):

            # For backprop the function need to check if the input given was below 0. If the inpput is below 0 times the input by zero if anything else multiply by 1  m

            self.input[self.input > 0] = 1

            output = self.module.multiply(output_gradient, self.input)


            return output

        super().__init__(activation, activation_prime)


class Swish(Activation):

    # Swish function is a modification of the sigmoid function
    # Swish(x) = x * sigmoidal(bx) where b is ascalable and trainable parameter

    def __init__(self, gpu: bool = False):
        
        def activation(input: np.ndarray | cp.ndarray):

            # TO:DO becasue b is a trainable parameter Swish needs to be made into a layer class rather than an activation

            input *= 1/1+np.exp(-input)

            return input

        def activation_prime(output_gradient: np.ndarray | cp.ndarray):

            output = ((output_gradient + self.module.sinh(output_gradient)) / (4 * (self.module.cosh(output_gradient)**2))) + 0.5

            return output

        super().__init__(activation, activation_prime)


class Sigmoid(Activation):

    # Sigmoidal(x) = 1 / 1 + e^-x

    def __init__(self, gpu: bool = False):

        def activation(input: np.ndarray | cp.ndarray):

            self.input = input

            print(f"Sig In: {self.input.shape}")

            output = 1/(1 + self.module.exp(-self.input))

            return output
        
        def activation_prime(output_gradient: np.ndarray | cp.ndarray):

            # return output_gradient * self.input * (1 - self.input)
            return
        
        super().__init__(activation, activation_prime)

        
            
        



