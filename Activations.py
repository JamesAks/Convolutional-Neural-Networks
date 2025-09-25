import numpy as np
import cupy as cp
from Activation import Activation

    

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


    # TO-DO: Need to implement the new activations for Inception and Effecicient. Sigmoidal?
    
class Act(Activation):

    def __init__(self):

        def act_forward(x):
                
            return x
            
        def act_backward(x):

            return x
            
        super().__init__(act_forward,act_backward)

# TO-DO implement RELU6 activation


class RELU6(Activation):

    def __init__(self):

        def activation(input: np.ndarray | cp.ndarray):

            self.setModule(type(input))

            # RELU6 returns 0 is the input is less than 0 and clamps the max value to 6
            input[input < 0] = 0
            input[input > 6] = 6

            self.input = input

            return self.input

        def activation_prime(output_gradient: np.ndarray | cp.ndarray):

            # For backprop the function need to check if the input given was below 0. If the inpput is below 0 times the input by zero if anything else multiply by 1  m

            self.input[self.input > 0] = 1

            output = self.module.multiply(output_gradient, self.input)

 
            #print(f"RELU6 Input shape : {self.input.shape}")

            #print(np.multiply(output_gradient,x))

            return output

        super().__init__(activation, activation_prime)


class Swish(Activation):

    def __init__(self):
        
        def activation(input: np.ndarray | cp.ndarray):

            self.setModule(type(input))
            
            input *= 1/1+np.exp(-input)

            print("Swish")

            return input

        def activation_prime(output_gradient: np.ndarray | cp.ndarray):

            output = ((output_gradient + self.module.sinh(output_gradient)) / (4 * (self.module.cosh(output_gradient)**2))) + 0.5

            print("Swish Done")

            return output

        super().__init__(activation, activation_prime)

class Sigmoid(Activation):

    def __init__(self):

        def activation(input: np.ndarray | cp.ndarray):

            self.input = input

            self.setModule(type(self.input))

            print(f"Sig In: {self.input.shape}")

            output = 1/(1 + self.module.exp(-self.input))

            return output
        
        def activation_prime(output_gradient: np.ndarray | cp.ndarray):

            return output_gradient * self.input * (1 - self.input)
        

        super().__init__(activation, activation_prime)

        
            
        



