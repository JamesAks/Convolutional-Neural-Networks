
from ConvolutionalLayer import ConvolutionLayer, GPUConvLayer
from Pooling import MaxPooling
from FullyConnected import FCLayer
from Softmax import Softmax
from Learning import train
from Losses import logLoss,logLossPrime
from DataLoader import DataLoader
from Activations import RELU6
from BatchNormalisation import BatchNorm
from Flatten import Flatten
from Model import Model

def ConstructVGG16(settings = None, mode: str = "Normal", classes: int = 8) -> list:

    network = []

    if settings == None:

        settings = [

            [64,3,1,1,2],
            [128,3,1,1,2],
            [256,3,1,1,3],
            [512,3,1,1,3],
            [512,3,1,1,3]

        ]

    for c, k, s, p, n in settings:

        for i in range(n):

            network.extend([ConvolutionLayer(c, k, s, p, mode),
                            RELU6()])
            
        network.append(MaxPooling((2,2),2))

    network.extend(
            [
                Flatten(),
                FCLayer(4096),
                BatchNorm(),
                RELU6(),
                FCLayer(4096),
                BatchNorm(),
                RELU6(),
                FCLayer(classes),
                Softmax()
            ])
    
    return network

vgg16 = ConstructVGG16()

print(len(vgg16))
         