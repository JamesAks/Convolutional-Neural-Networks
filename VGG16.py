
from ConvolutionalLayer import ConvolutionLayer
from Pooling import MaxPooling
from FullyConnected import FCLayer
from Softmax import Softmax
from Activations import RELU6
from BatchNormalisation import BatchNorm
from Flatten import Flatten
from Model import Model


class VGG16(Model):

    # The VGG16 model as specified in its original paper

    def __init__(self,model_name: str = "VGG16", settings = None, mode: str = "Normal", num_classes: int = 8, gpu: bool = False):

        super().__init__(model_name,gpu)

        if mode != "Normal" and mode != "im2col":

            print("Only two mode for Convolution: 'Normal' and 'im2col'. Please pick one.")
            raise SystemExit
        
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

                self.network.extend([ConvolutionLayer(c, k, s, p, mode, gpu = gpu),
                                RELU6(gpu=gpu)])
            
            self.network.append(MaxPooling((2,2),2, gpu = gpu))

        self.network.extend(
            [
                Flatten(gpu = gpu),
                FCLayer(128, gpu = gpu),
                BatchNorm(gpu = gpu),
                RELU6(gpu = gpu),
                FCLayer(64, gpu = gpu),
                BatchNorm(gpu = gpu),
                RELU6(gpu = gpu),
                FCLayer(num_classes, gpu = gpu),
                Softmax(gpu = gpu)
            ])
        
        




    
         














    


