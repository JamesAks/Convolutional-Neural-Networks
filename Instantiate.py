from VGG16 import VGG16
from InceptionNet import InceptionNet
from EfficientNet import EfficientNetB0

def instantiateModels(mode: str = "Normal"):

    vgg16 = VGG16(mode)
    efficientNet = EfficientNetB0(mode = "Normal")
    inceptionNet = InceptionNet(mode)

    models = [vgg16, efficientNet, inceptionNet]
    model_names = ["VGG16" , "EfficientNet", "InceptionNet"]

    return models, model_names