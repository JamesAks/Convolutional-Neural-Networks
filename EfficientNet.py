
from ConvolutionalLayer import ConvolutionLayer
from InvResBlock import InvResBlock
from Pooling import AveragePooling
from BatchNormalisation import BatchNorm
from FullyConnected import FCLayer
from Softmax import Softmax
from Activations import Swish
from Pooling import AveragePooling
from InvResBlock import InvResBlock
from BatchNormalisation import BatchNorm
from Flatten import Flatten
from Model import Model


class EfficientNetB0(Model):

    #This is the original Efficient B0 model without the comput scaling factor

    def __init__(self, mode: str = "Normal"):

        self.mode = mode

        self.network = [
            
            ConvolutionLayer(32,3,2),
            BatchNorm(),
            Swish(),

            # 1 input_channels - 2 expansion_ratio - 3 output_channels - 4 kernel_size - 5 stride

            InvResBlock(32,1,16,3,1),
        
            InvResBlock(16,6,24,3,2),
            InvResBlock(24,6,24,3,2),

            InvResBlock(24,6,40,5,),
            InvResBlock(40,6,40,5,),

            InvResBlock(40,6,80,3,),
            InvResBlock(80,6,80,3,),
            InvResBlock(80,6,80,3,),

            InvResBlock(80,6,112,5,),
            InvResBlock(112,6,112,5,),
            InvResBlock(112,6,112,5,),

            InvResBlock(112,6,192,5,),
            InvResBlock(192,6,192,5,),
            InvResBlock(192,6,192,5,),
            InvResBlock(192,6,192,5,),

            InvResBlock(192,6,320,3,),

            ConvolutionLayer(320,1280,1,1),
            BatchNorm(),
            Swish(),
            AveragePooling(glob = True),
            Flatten(),
            FCLayer(8),
            Softmax(),

        ]


# TO: DO implement EfficientNet compound sclaing method



