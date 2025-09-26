
from ConvolutionalLayer import ConvolutionLayer, GPUConvLayer
from Pooling import MaxPooling
from FullyConnected import FCLayer
from Softmax import Softmax
from Activations import RELU6
from BatchNormalisation import BatchNorm
from Flatten import Flatten
from Model import Model


class VGG16(Model):

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
        
        
        # self.network = [

        #     ConvolutionLayer(64, 3, 1, 1, mode= mode),
        #     RELU6(),
        #     ConvolutionLayer(64, 3, 1, 1,mode= mode),
        #     RELU6(),
        #     MaxPooling((2,2),2),
        #     ConvolutionLayer(128, 3, 1, 1, mode= mode),
        #     RELU6(),
        #     ConvolutionLayer(128, 3, 1, 1, mode= mode),
        #     RELU6(),
        #     MaxPooling((2,2),2),
        #     ConvolutionLayer(256, 3, 1, 1, mode= mode),
        #     RELU6(),
        #     ConvolutionLayer(256, 3, 1, 1, mode= mode),
        #     RELU6(),
        #     ConvolutionLayer(256, 3, 1, 1, mode= mode),
        #     RELU6(),
        #     MaxPooling((2,2),2),
        #     ConvolutionLayer(512,3,1,1, mode= mode),
        #     RELU6(),
        #     ConvolutionLayer(512,3,1,1, mode= mode),
        #     RELU6(),
        #     ConvolutionLayer(512,3,1,1, mode= mode),
        #     RELU6(),
        #     MaxPooling((2,2),2),
        #     ConvolutionLayer(512,3,1,1, mode= mode),
        #     RELU6(),
        #     ConvolutionLayer(512,3,1,1, mode= mode),
        #     RELU6(),
        #     ConvolutionLayer(512,3,1,1, mode= mode),
        #     RELU6(),
        #     MaxPooling((2,2),2),
        #     Flatten(),
        #     FCLayer(4096),
        #     BatchNorm(),
        #     RELU6(),
        #     FCLayer(4096),
        #     BatchNorm(),
        #     RELU6(),
        #     FCLayer(8),
        #     Softmax()
        # ]


        
network = [

    ConvolutionLayer(3,3,2,1, mode = "im2col"),
    RELU6(),
    ConvolutionLayer(2,3,1,1, mode = "im2col"),
    RELU6(),
    MaxPooling((2,2),2),
    Flatten(),
    FCLayer(100),
    BatchNorm(),
    RELU6(),
    FCLayer(8),
    Softmax()

]

GPnetwork = [

    GPUConvLayer(3,(3,3),2),
    RELU6(),
    GPUConvLayer(2,(3,3),1),
    RELU6(),
    MaxPooling((2,2),2),
    Flatten(),
    FCLayer(100),
    BatchNorm(),
    RELU6(),
    FCLayer(8),
    Softmax()

]

# img_paths = []
# exp_paths = []

# image_path = "C:/Users/james/Documents/Masters/Dissertation/Preprocessed_Train/Train_images"
# train_exp_path = "C:/Users/james/Documents/Masters/Dissertation/Preprocessed_Train/Train_expected"
# test_image_paths = ""
# test_expected_paths =""

# train_data = DataLoader(image_path, train_exp_path, 16, targetSize=(244,244), numb_samples= 10000) 
# gpu_train = DataLoader(image_path, train_exp_path, 100, targetSize=(32,32), numb_samples= 10000, gpu = True)

# #x_test, y_test = DataLoader([*test_image_paths],[*test_expected_paths],50)

# norm_vgg16 = VGG16()
# im2col_vgg16 = VGG16( mode = "im2col")
# GPU_vgg16 = VGG16( mode = "im2col")


# # train("Normal VGG16", norm_vgg16.network, logLoss, logLossPrime, train_data)
# #train("Im2Col VGG16", im2col_vgg16.network, logLoss, logLossPrime, train_data)

# train("GPU VGG16", im2col_vgg16.network, logLoss, logLossPrime, gpu_train)







    
         














    


