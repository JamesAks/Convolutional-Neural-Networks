import numpy as np
from Layer import Layer
from ConvolutionalLayer import ConvolutionLayer
from DeptheWiseConvolution import DWConvolution 
# from SeperableConvolution import DWConvolution
from Activations import RELU6
from BatchNormalisation import BatchNorm
from SELayer import SqueezeExciteLayer



class InvResBlock(Layer):

    # Inverted Residual Block is a method used to reduce the amount of parameters while keeping all other properties of the network.
    # Instead of doing a normal convolution with an output of D channels, first  a depthwise convolution is done where each input depth has its own filter (no mixing)
    # After a 1x1 convolution is done to expand the output to D depth.

    def __init__(self, input_channels: int, expansion_ratio: int, output_channels: int, kernel_size: int = 3, stride: int = 1, mode = "Normal", gpu: bool = False):

        self.padding = kernel_size // 2
        self.stride = stride

        expanded_input = input_channels * expansion_ratio
        layers = []
        # If the number of input channels and the number of output channels are the same then we add the input to the output.
        self.res = input_channels == output_channels and self.stride == 1

        # 3 Phases

        # Expansion phase(Pointwise Convolution)
        # Firat layer should be a 1 by 1 point wise convolution of input depth only if the expanse ratio is not 1
        # With RELU6 activation

        if expansion_ratio != 1:

            
            layers.extend([ConvolutionLayer(expanded_input ,1, 1, mode = mode),
                           BatchNorm(),
                           RELU6()
                          ])
        
        layers.extend(

            # Depthwise Convolution phase
            # Second Layer should be a 3x3 depth-wise convolution of 1 depth
            # With RELU6 activation
    
            [DWConvolution(kernel_size, stride = self.stride),
            RELU6(),
            BatchNorm(),
            SqueezeExciteLayer(),

            # Projection Phase (Pointwise Convolution)
            # Third layer should be a 1x1 Convolution of n depth

            ConvolutionLayer(output_channels, 1, 1, mode = mode),
            BatchNorm()]
            )
        
        self.layers = layers

        super().__init__(gpu)


    def forward(self,input: np.ndarray):


        output = input

        for layer in self.layers:
           
           output = layer.forward(output)

        if self.res:

            output += input

        return output
    
    def backward(self, error_grad: np.ndarray, learning_rate: float) -> np.ndarray:

        inp = error_grad

        for layer in reversed(self.layers):

            inp = layer.backward(inp, learning_rate)

        return inp 
    
    
    def update(self,learning_rate: float = 0.09):

        for layer in self.layers:

            layer.update(learning_rate)



        
    
