from EfficientNet import EfficientNetB0
from InceptionNet import InceptionNet
from VGG16 import VGG16
from Display import EvalLogger
from DataLoader import DataLoader
from Losses import logLoss, logLossPrime

# Constants hold the paths towards datasets

TRAIN_PATH = "C:/Users/james/Documents/Masters/Dissertation/Preprocessed_Train/Train_images"
TRAIN_EXP_PATH = "C:/Users/james/Documents/Masters/Dissertation/Preprocessed_Train/Train_expected"

TEST_PATH = ""
TEST_EXP_PATH = ""

CV_PATH = ""


vgg_16 = VGG16(mode = "im2col", gpu = True)

data = DataLoader(TRAIN_PATH, TRAIN_EXP_PATH, 1, targetSize= (32,32), numb_samples = 1000)
eval_log = EvalLogger()

vgg_16.train(logLoss, logLossPrime, data, eval_log,)
