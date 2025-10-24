from Model import Model
from Inception import InceptionLayer
from ConvolutionalLayer import ConvolutionLayer
from Pooling import MaxPooling, AveragePooling
from Softmax import Softmax
from FullyConnected import FCLayer
from Activations import RELU6
from Flatten import Flatten

class InceptionNet(Model):

    # The original InceptionNet / GooglenET Architecture set out in the original paper

    def __init__(self, mode: str = "Normal"):
        
        self.network = [

            ConvolutionLayer(16,3,1, mode = mode),
            RELU6(),

            InceptionLayer( branches = [
                
                #Branch 1
                [
                ConvolutionLayer(8,1,1, mode = mode),
                RELU6()
                ],
 
                #Branch 2
                [
                ConvolutionLayer(8,1,1, mode = mode),
                RELU6(),
                ConvolutionLayer(16,3,1, padding = 1, mode = mode),
                RELU6()
                ],
                
                #Branch 3
                [
                ConvolutionLayer(4,1,1, mode = mode),
                RELU6(),
                ConvolutionLayer(8,5,1,padding = 2, mode = mode),
                RELU6(),
                ],

                #Branch 4
                [
                MaxPooling((3,3),1,1),
                ConvolutionLayer(8,1,1, mode = mode),
                RELU6()
                ]
            ]),

            MaxPooling((2,2),2),

            InceptionLayer( branches=[ 

                #Branch 1
                [
                ConvolutionLayer(16,1,1, mode = mode),
                RELU6(),
                ],

                #Branch 2
                [
                ConvolutionLayer(12,1,1, mode = mode),
                RELU6(),
                ConvolutionLayer(16,3,1, padding = 1, mode = mode),
                RELU6(),
                ],

                #Branch 3
                [
                ConvolutionLayer(4,1,1, mode = mode),
                RELU6(),
                ConvolutionLayer(8,5,1, padding = 2, mode = mode),
                RELU6(),
                ],

                #Branch 4
                [
                MaxPooling((3,3),1, padding = 1)
                ]
            ]),

            AveragePooling(glob=True),
            Flatten(),
            FCLayer(8),
            Softmax()
            
        ]





    

