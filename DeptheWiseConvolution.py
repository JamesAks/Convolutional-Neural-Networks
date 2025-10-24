import numpy as np
import cupy as cp
from Layer import Layer
from Conv import crossCorrelation3D,conv_back, im2col, dialate,col2im, depth_conv, depth_conv_back, im2col_depthwise, col2im_depthwise



class DWConvolution(Layer):

    def __init__(self, kernel_size: int, stride: int, padding: int = 0, mode: str = "Normal", momentum: float = 0.09, gpu: bool = False):

        self.setModule(gpu)

        self.mode = mode
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = kernel_size // 2 #Same padding. Dimensionality only affected by stride

        self.biases = self.module.array([])
        self.kernels = self.module.array([])

        self.velocity_kernel = self.module.array([]) 
        self.velocity_biases = self.module.array([])

        self.momentum = momentum
    


    def forward(self, input: np.ndarray):

        # Input is an numpy Array made from the images
        self.input = input

        print(f"Convolution Input {self.input.shape}")

    
        self.input = np.pad(input,((0,0),(0,0),(self.padding,self.padding), (self.padding,self.padding)) )

        self.input_shape = self.input.shape
        self.batch_size, self.input_channels, self.input_width, self.input_height = self.input.shape

        if len(self.kernels) == 0 or len(self.biases) == 0:

            if self.kernel_size > self.input_height or self.kernel_size > self.input_width:

                self.kernels = np.random.randn(1,self.input_channels, self.input_width, self.input_height)
                self.kernels_shape = self.kernels.shape
                self.kernel_size = self.input_width

            else:

                self.kernels = np.random.randn(1, self.input_channels, self.kernel_size, self.kernel_size)
                self.kernels_shape = self.kernels.shape

            self.biases = self.module.random.randn(self.input_channels, 1)

            self.velocity_biases = self.module.zeros(self.biases.shape)
            self.velocity_kernel = self.module.zeros(self.kernels_shape)

        if self.mode == "Normal":
 
            self.output = depth_conv(self.input, self.kernels, self.stride)

            self.output_shape = self.output.shape
                
            if len(self.biases) == 0:

                self.biases = np.random.randn(self.input_channels, 1)

            self.output += self.biases

            print(f"Convolution Finished. Output Shape:{self.output_shape}")

            return self.output 
        
        elif self.mode == "im2col":

            output_width = ((self.input_width - self.kernel_size) // self.stride) + 1
            output_height = ((self.input_height - self.kernel_size) // self.stride) + 1

            self.input_colums = im2col_depthwise(self.input, self.kernels_shape, 1,0)
            self.kernel_col = self.kernels.reshape(self.input_channels, -1, 1)

            output = np.matmul(self.kernel_col.transpose(0, 2, 1), self.input_colums)
            output = output.reshape(self.batch_size, self.input_channels, output_width, output_height) + self.biases

            


            return output
    

    def backward(self, error_grad, learning_rate):

        #Initialising the matrices


        self.kernels_gradient = np.zeros(self.kernels_shape)

        input_gradient = np.zeros(self.input_shape)

        self.bias_gradient = np.sum(error_grad, axis = (0,2,3)).reshape(1,self.input_channels, 1, 1)

        batch, chan, error_width, error_height = error_grad.shape

        # The kernel error is the input cross correlated with the error given the output
        if self.mode == "Normal":

            if self.stride > 1:

                error_grad = dialate(error_grad,self.stride)

            self.kernels_gradient, input_gradient = depth_conv_back(self.input, self.kernels, error_grad, 1) 

            
            print("Back")

            print(input_gradient.shape)

            # End by undoing padding 

            return input_gradient[:,:, self.padding : self.input_width - self.padding, self.padding : self.input_height - self.padding]
        
        elif self.mode == "im2col":

            rotated_err = self.module.rot90(error_grad).transpose(1,2,3,0).reshape(self.batch_size, self.input_channels, -1)

            error_reshape = error_grad.transpose(1,2,3,0).reshape(self.batch_size,self.input_channels, -1)

            self.kernels_gradient = np.zeros((self.input_channels, self.kernels_shape[0]*self.kernels_shape[1]))
            
            for n in range(self.batch_size):

                self.kernels_gradient += np.matmul(error_reshape[n][:, None, :], error_reshape[n].transpose(0,2,1)).sum(axis=0)

            self.kernels_gradient = self.kernels_gradient.reshape(1,self.input_channels, self.kernel_size , self.kernel_size)

            self.bias_gradient = error_reshape.sum(axis=(0, 2, 3)).reshape(1,self.input_channels, 1,1)

            error_cols = self.module.zeros(self.input_colums.shape
                                           )
            for n in range(self.batch_size):
                # Multiply per channel
                for c in range(self.input_channels):
                # (KH*KW, OH*OW)
                    error_cols[self.batch_size, self.input_channels] = self.module.matmul(self.kernel_col[c] , error_reshape[n, c][None, :])

            im_gradient = col2im_depthwise(error_cols, self.input_shape,(self.kernel_size,self.kernel_size), self.stride, self.padding)

            return im_gradient
    
    
    def update(self, learning_rate: float = 0.01):

        self.velocity_kernel = (self.momentum * self.velocity_kernel) - (self.kernels_gradient * learning_rate)
        self.velocity_biases = (self.momentum * self.velocity_biases) - (self.bias_gradient * learning_rate)

        self.kernels += self.velocity_kernel
        self.biases += self.velocity_biases