
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


# b1 = [

#     ConvolutionLayer(8,3,2,1),
#     # 1 input_channels - 2 expansion_ratio - 3 output_channels - 4 kernel_size - 5 stride
#     InvResBlock(8,1,8,3,1),
#     InvResBlock(8,6,16,3,2),
#     InvResBlock(16,6,24,5,2),
#     ConvolutionLayer(32,1,1), 
#     BatchNorm(),
#     Swish(),
#     AveragePooling(glob= True),
#     Flatten(),
#     FCLayer(8),
#     Softmax()

# ]

# b_0 = EffecientNetB0("Normal")


# img_paths = []
# exp_paths = []

# image_path = "C:/Users/james/Documents/Masters/Dissertation/Preprocessed_Train/Train_images"
# train_exp_path = "C:/Users/james/Documents/Masters/Dissertation/Preprocessed_Train/Train_expected"
# test_image_paths = ""
# test_expected_paths =""

# train_data = DataLoader(image_path, train_exp_path, 100, targetSize= (32,32))
# #x_test, y_test = DataLoader([*test_image_paths],[*test_expected_paths],50)


# #train("Normal_EfficientNet",b_0.network, logLoss, logLossPrime, train_data)

      
