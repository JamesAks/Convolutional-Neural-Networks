import numpy as np
import cupy as cp
from Layer import Layer
from Conv import pad3D



class MaxPooling(Layer):

    def __init__(self, window_shape: tuple, stride: int, padding: int = 0, gpu: bool = False):

        self.window = window_shape
        self.stride = stride
        
        if padding < 0:

            print("Padding must be more than or equal to 0")
            raise SystemExit()
        
        self.padding = padding
        super().__init__(gpu)


    def forward(self,input: np.ndarray | cp.ndarray):

        self.input = input

        self.input = self.module.pad(self.input,((0,0), (0,0), (self.padding, self.padding), (self.padding, self.padding)))

        self.batch,self.input_depth, self.input_width, self.input_height = self.input.shape

        
        if self.window[0] > self.input_width or self.window[1] > self.input_height:

            self.window = (self.input_width, self.input_height)

        # ((input size - pooling window size) / stride ) + 1 Assuming no padding
        self.output_width = ((self.input_width - self.window[0]) // self.stride) + 1
        self.output_height = ((self.input_height - self.window[1]) // self.stride) + 1

        self.output = self.module.zeros((self.batch,self.input_depth, self.output_width, self.output_height))
        

        for w in range(self.output_width):

            for h in range(self.output_height):
                    
                # Finding the max in an n x m sized window at depth d
                    
                self.output[:,:,w,h] = input[:,:,w*self.stride:w*self.stride + self.window[0], h*self.stride: h*self.stride + self.window[1]].max(axis = (2,3))

        #print(f"Max Pooling: {self.output}")
                    
        return self.output
    
    
    def backward(self, error_grad: np.ndarray | cp.ndarray, learning_rate: float):

        # Gradient only matter only on the positions where the max was found everything else is left the same so pad out with zero.
        #Initialise an array of zeroes similar to the input
        #print(f"pool input shape {self.input.shape}")
        backprop = self.module.zeros(self.input.shape)

        #print(f"pool error {error_grad.shape}")

        for h in range(self.output_height):

            for w in range(self.output_width):
                    
                # Finding the max in an n x m sized window at depth d
                # Adds the error gradient to the array only where the max is found. Boolean at the end determines whether its is added or not as the max is either found there (1) or not (0).
                    
                backprop[:, :, w*self.stride:w*self.stride + self.window[0], h*self.stride: h*self.stride + self.window[1]] += error_grad[:, :, w:w+1 ,h:h+1] * (self.output[: ,: ,w:w+1,h:h+1] == self.input[:, :, w*self.stride:w*self.stride + self.window[0], h*self.stride: h*self.stride + self.window[1]])
        
        return backprop[:, :, self.padding : self.input_width - self.padding, self.padding : self.input_height - self.padding]
    
    
class AveragePooling(Layer):


    def __init__(self, window_shape: tuple = (2,2), stride: int = 1, glob: bool = False, gpu: bool = False):

        
        self.window_width, self.window_height = window_shape
        self.glob = glob
        self.stride = stride

        super().__init__(gpu)
        

    def forward(self, input: np.ndarray | cp.ndarray):

   
        self.input = input
        self.input_shape = input.shape
        self.batch_size, self.input_depth, self.input_width, self.input_height = self.input_shape


        
        if self.glob == True:

            self.window_width = self.input_width
            self.window_height = self.input_height


        self.output_width = int((self.input_width - self.window_width) / self.stride) + 1
        self.output_height = int((self.input_height - self.window_height) / self.stride) + 1

        self.output_shape = (self.batch_size, self.input_depth, self.output_width, self.output_height)



        self.output = self.module.zeros(self.output_shape)

        for n in range(self.batch_size):

            for d in range(self.input_depth):

                for h in range(self.output_height):

                    for w in range(self.output_width):

                        if self.glob == True:

                            self.output[n,d,w,h] = self.module.average(self.input[n,d, :, :])

                        else:

                            self.output[n,d,w,h] = self.module.average(self.input[n, d, w*self.stride:w*self.stride + self.window_width, h*self.stride: h*self.stride + self.window_height])


        return self.output
    
    
    def backward(self, error_grad: np.ndarray | cp.ndarray, learning_rate: float):

        backprop = self.module.zeros(self.input_shape)

        for n in range(self.batch_size):

            for d in range(self.input_depth):

                for h in range(self.output_height):

                    for w in range(self.output_width):

                        # Because we averaged over the inputs in the window the error is equally spread across the inputs to the pooling window
                        backprop[n, d, w*self.stride : w*self.stride + self.window_width, h*self.stride : h*self.stride + self.window_height] += (error_grad[n,d,h,w] / max(1,(self.window_width * self.window_height))) 

        return backprop
    

# class GlobalAveragePooling(Layer):

#     def __init__(self):

#         super().__init__()


#     def forward(self, input: np.ndarray) -> np.ndarray:

#         self.input = input
#         self.input_shape = input.shape
#         self.batch, self.inp_depth, self.inp_width, self.inp_height = self.input_shape

#         self.output = np.zeros((1,self.inp_depth,1,1))

#         for i in range(self.inp_depth):
            
#             self.output[i] = np.mean(input)

#         return self.output
    
    
#     def backward(self, error_grad: np.ndarray, learning_rate: float) -> np.ndarray:

#         inp_grads = self.input

#         for i in range (self.inp_depth):

#              inp_grads += error_grad[i]/ (self.inp_width * self.inp_height)


#         return inp_grads

