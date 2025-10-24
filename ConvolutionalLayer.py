import numpy as np
import cupy as cp
from Layer import Layer
from Conv import crossCorrelation3D,conv_back, im2col, dialate,col2im, depth_conv, depth_conv_back, im2col_depthwise, col2im_depthwise


class ConvolutionLayer(Layer):

    def __init__(self, depth: int, kernel_size: int, stride: int = 1, padding: int = 0, mode: str = "Normal", momentum: float = 0.9, gpu: bool = False):

        # kernel_Size represents the dimensions of each matrix in a kernel
        # depth represents how many kernels

        super().__init__(gpu)

        self.mode = mode
        self.depth = depth
        self.stride = stride
        self.padding = padding

        self.kernel_size = kernel_size

        self.kernels = self.module.array([])
        self.biases = self.module.array([])

        if padding < 0 or stride < 0:

            print(" Padding and stride must be greater than or equal to 0")
            raise SystemExit()
        
        self.momentum = momentum

        self.velocity_kernels = self.module.array([]) 
        self.velocity_biases = self.module.array([])
        
    def forward(self, input):

        # Input is an numpy Array made from the images

        self.input = self.module.pad(input,((0,0),(0,0),(self.padding,self.padding), (self.padding,self.padding)))

        self.batch_size, self.input_channels, self.input_width, self.input_height = self.input.shape

        self.input_shape = self.input.shape

        print(f"Convolution Input {self.input.shape} and Type {type(input)}")
            
        #(input[0])

        if len(self.kernels) == 0 or len(self.biases) == 0:

            if self.kernel_size > self.input_height or self.kernel_size > self.input_width:

                self.kernels = self.module.random.randn(self.depth,self.input_channels, self.input_width, self.input_height)
                self.kernels_shape = self.kernels.shape
                self.kernel_size = self.input_width
                

            else:

                self.kernels = self.module.random.randn(self.depth, self.input_channels, self.kernel_size, self.kernel_size)
                self.kernels_shape = self.kernels.shape
                
            self.biases = self.module.random.randn(1, self.depth, 1, 1)

            self.velocity_kernel = self.module.zeros(self.kernels_shape)
            self.velocity_biases = self.module.zeros(self.biases.shape)

            print(f"V Shape{ self.velocity_biases.shape} " )


        if self.mode == "im2col":

            output_width = ((self.input_width - self.kernel_size) // self.stride) + 1
            output_height = ((self.input_height - self.kernel_size) // self.stride) + 1

            self.output_shape = (self.batch_size, self.depth, output_width, output_height)
    
            #Turning nD Input and nD Weights into columns

            self.input_col = im2col(input, (self.kernel_size,self.kernel_size), self.stride, self.padding).astype(self.module.float16)
            print(self.input_col.shape)
            self.kernel_col = self.kernels.reshape(self.depth, -1).astype(self.module.float16)

            bytes = 150 * 1024 * 1024
            col_num = bytes / ((self.kernel_size ** 2) * self.depth)

            output = self.module.dot(self.kernel_col, self.input_col).astype(self.module.float32)

            output = output.reshape(self.batch_size, self.depth, output_width, output_height)

            output += self.biases

            # Setting Biases only on first pass
            print(f"Output Type: {type(output)}")

            return output #bias
        
        elif self.mode == "Normal":

            # The amount of output matrices should be the same as number of kernels
                
            self.output = crossCorrelation3D(self.input, self.kernels, self.stride)

            self.output_shape = self.output.shape

            self.output += self.biases

            print(f"Convolution Finished. Output Shape:{self.output_shape}")

            return self.output
        
        else:

            print("Only two mode for Convolution: 'Normal' and 'im2col'. Please pick one.")
            raise SystemExit


    def backward(self, error_grad: np.ndarray, learning_rate: float):

        #Initialising the matrices
        self.kernels_gradient = self.module.zeros(self.kernels_shape)
        self.bias_gradient = self.module.zeros(self.biases.shape)


        input_gradient = self.module.zeros(self.input_shape)

        batch, chan, error_width, error_height = error_grad.shape

        if self.mode == "im2col":

            print(f"X shape: {self.input.shape}")

            print(f"Input SHape: {self.input_shape}, Output shape: {self.output_shape}")

            # if self.stride > 1:

            #     dil_error = dialate(error_grad, self.stride)

            rotated_err = self.module.rot90(error_grad).transpose(1,2,3,0).reshape(self.depth, -1)

            error_col = error_grad.transpose(1,2,3,0).reshape(self.depth, -1)

            self.kernels_gradient = self.module.dot(error_col, self.input_col.T).reshape(self.kernels_shape)

            self.bias_gradient = error_grad.sum(axis = (0,2,3)).reshape(1,self.depth, 1, 1)

            input_gradient = self.module.dot(self.kernel_col.T, rotated_err)


            im_gradient = col2im(input_gradient, self.input_shape, (self.kernel_size, self.kernel_size), self.stride, self.padding)

            print(f"inp_grad shape: {im_gradient.shape}")

            return im_gradient
        
        elif self.mode == "Normal":

            self.bias_gradient = error_grad.sum(axis = (0, 2, 3)).reshape(1, self.depth, 1, 1)

            if self.stride > 1:

                error_grad = dialate(error_grad,self.stride)

            self.kernels_gradient, input_gradient = conv_back(self.input, self.kernels, error_grad, 1) 

            # End by undoing padding 

            return input_gradient[:,:, self.padding : self.input_width - self.padding, self.padding : self.input_height - self.padding]
        

    def update(self, learning_rate: float = 0.01):

        self.velocity_kernel = (self.momentum * self.velocity_kernel) - (self.kernels_gradient * learning_rate)
        self.velocity_biases = (self.momentum * self.velocity_biases) - (self.bias_gradient * learning_rate)

        print(f"Velocity Bias Sahpe {self.velocity_biases.shape}")

        self.kernels += self.velocity_kernel
        self.biases += self.velocity_biases


    



      










