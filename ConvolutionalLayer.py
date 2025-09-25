import numpy as np
import cupy as cp
from Layer import Layer
from Conv import crossCorrelation3D,conv_back, im2col, dialate,col2im, depth_conv, depth_conv_back, im2col_depthwise, col2im_depthwise


class ConvolutionLayer(Layer):

    def __init__(self, depth: int, kernel_size: int, stride: int = 1, padding: int = 0, mode: str = "Normal", momentum: float = 0.9, gpu: bool = False):

        # kernel_Size represents the dimensions of each matrix in a kernel
        # depth represents how many kernels

        self.setModule(gpu)

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

        print(f"Convolution Input {self.input.shape}")
            
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

            self.input_col = im2col(input, (self.kernel_size,self.kernel_size), self.stride, self.padding)
            self.kernel_col = self.kernels.reshape(self.depth, -1)

            output = self.module.dot(self.kernel_col, self.input_col)

            output = output.reshape(self.batch_size, self.depth, output_width, output_height)

            output += self.biases

            # Setting Biases only on first pass

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







class GPUConvLayer(Layer):

    def __init__(self, output_channels: int, kernel_shape: tuple, stride: int = 1, padding: int = 0):

        self.out_channels = output_channels
        self.kernel_width, self.kernel_height = kernel_shape
        self.stride = stride
        self.padding = padding
        self.kernels = np.array([])
        self.biases = np.array([])

    def forward(self, input: np.ndarray) -> np.ndarray:

        self.input = input
        self.input_shape = self.input.shape
        #print(f"X shape: {self.input.shape}")
        self.batch, self.input_depth, self.input_width, self.input_height = self.input.shape

        # Setting Kernel weights only on first pass

        if len(self.kernels) == 0:

            if self.kernel_height > self.input_height or self.kernel_width > self.input_width:

                self.kernels = np.random.randn(self.out_channels,self.input_depth, self.input_width, self.input_height)
                self.kernels_shape = self.kernels.shape

            else:

                self.kernels = np.random.randn(self.out_channels, self.input_depth, self.kernel_width, self.kernel_height)
                self.kernels_shape = self.kernels.shape
        
        
        # Output shape

        output_width = ((self.input_width - self.kernel_width + (2 * self.padding)) // self.stride) + 1
        output_height = ((self.input_height - self.kernel_height + (2 * self.padding)) // self.stride) + 1
        self.output_shape = (self.batch, self.out_channels, output_width, output_height)
    
        #Turning nD Input and nD Weights into columns

        self.input_col = im2col(input, (self.kernel_width,self.kernel_height), self.stride, self.padding)
        self.kernel_col = self.kernels.reshape(self.out_channels, -1)

        output = np.dot(self.kernel_col, self.input_col)

        # self.col_input, output = colConv(self.input, self.kernels, self.stride, self.padding)
        output = output.reshape(self.out_channels, output_width, output_height, self.batch)
        output = output.transpose(3,0,1,2)

        #print(f"Output Shape: {output.shape}")

        # Setting Biases only on first pass

        if len(self.biases) == 0:

            self.biases = np.random.randn(self.out_channels)

        return output #bias
    
    
    def backward(self, error_grad: np.ndarray, learning_rate: float) -> np.ndarray:


        print(f"X shape: {self.input.shape}")

        print(f"Input SHape: {self.input_shape}, Output shape: {self.output_shape}")

        # if self.stride > 1:

        #     dil_error = dialate(error_grad, self.stride)

        rotated_err = np.rot90(error_grad).transpose(1,2,3,0).reshape(self.out_channels, -1)

        error_col = error_grad.transpose(1,2,3,0).reshape(self.out_channels, -1)

        kernels_gradient = np.dot(error_col, self.input_col.T).reshape(self.kernels_shape)

        input_gradient = np.dot(self.kernel_col.T, rotated_err)

        

        im_gradient = col2im(input_gradient, self.input_shape, (self.kernel_width, self.kernel_height), self.stride, self.padding)

        # kernels_gradient = colConv(self.input, error_grad)
        # input_gradient = colConv(self.kernels, np.rot90(np.rot90(error_grad)), mode = "full")
        print(f"inp_grad shape: {im_gradient.shape}")

        self.kernels -= kernels_gradient * learning_rate
        self.biases -= error_grad.sum(axis = (0,2,3)) * learning_rate

        return im_gradient

        

      










