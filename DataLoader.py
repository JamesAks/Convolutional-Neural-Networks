import os
import numpy as np
import cupy as cp
from PIL import Image

class DataLoader:


    def __init__(self, imagePath: str, expPath: str, batchSize: int, shuffle: bool = False, normalisation: bool = True, targetSize: tuple = (244,244), numb_samples = 0):

        self.imagePaths = self.getPaths(imagePath)
        self.expPaths = self.getPaths(expPath)
        self.batchSize = batchSize
        self.shuffle = shuffle
        self.normalisation = normalisation
        self.targetSize = targetSize
        self.currentIdx = 0
        self.gpu = False
        self.module = np

        if numb_samples == 0:

            self.numb_samples = len(self.imagePaths)

        else:

            self.numb_samples = numb_samples

        self.index = range(self.numb_samples)

        if self.shuffle == True:

            np.random.shuffle(self.index)
        

    def __iter__(self):

        self.currentIdx = 0

        if self.shuffle == True:

            np.random.shuffle(self.index)
        
        return self
    

    def __next__(self):

        images = []
        labels = []

        if self.currentIdx >= self.numb_samples:
            raise StopIteration

        start = self.currentIdx
        end = self.currentIdx + self.batchSize

        batch = self.index[start:end]

        for idx in batch:

            img = Image.open(f"{self.imagePaths[idx]}")
            img = img.resize(self.targetSize)
            img_array = self.module.reshape((self.module.asarray(img)),(3,*self.targetSize))

            if self.normalisation == True:

                img_array = img_array/255

            images.append(img_array)
            label = np.load(self.expPaths[idx], allow_pickle= True)
            
            labels.append(label)
            
        #One hot encode
        images = self.module.array(images)
        labels = self.one_hot_encode(labels)


        self.currentIdx = end

        return images, labels
    


    def one_hot_encode(self, labels: list,numclasses: int = 8) -> np.ndarray | cp.ndarray:

        encoded_list = []

        for label in labels:

            encoded = self.module.identity(numclasses)[int(label)]
            encoded_list.append(encoded)

        encoded_list = self.module.array(encoded_list)
        #encoded = oneHotLabs[np.array(labels)]

        return encoded_list
    
    def getPaths(self, path: str) -> list:

        paths = []

        for root, dirs, files in os.walk(path):

            for p in files:

                p = root + "/" + p
                paths.append(p)
                
        return paths
    
    def setModule(self, gpu: bool):
        
        if gpu: 

            self.gpu = True
            self.module = cp

        else:

            self.gpu = False
            self.module = np


    



   


