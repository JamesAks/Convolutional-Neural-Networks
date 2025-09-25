from numpy import asarray
import numpy as np
import cv2
from PIL import Image
import os
import math

import torch

FILE_PATH ="1_exp.npy"
PADDING = 20
INPUT_WIDTH = 640
INPUT_HEIGHT = 480


# import numpy as np
# a = np.arange(12).reshape((3,2,2))
# b = np.copy(a)
# np.random.shuffle(b)

# b = torch.from_numpy(b)
# a = torch.from_numpy(a)

# a += b[0]

# print(a)
# print(b)

# class Test():

#     def __init__(self,num_1, num_2, num_3):

#         self.num_1 = num_1

#         self.num_2 = num_2

#         self.num_3 = num_3

#         self.num_1 = self.num_2 + self.num_3 


#     def forward(self):

#         self.num_4 = self.num_3 * 2

#         print(self.num_1) 



# class Tester(Test):

#     def __init__(self,num_1,num_2,num_3):

#         #self.num_1 = 1

#         super().__init__(num_1,num_2, num_3)

#         self.num_1 = self.num_1 + self.num_2

#     def forward(self):

#         self.num_1 = self.num_4
        
#         print(self.num_1)

        

        





# Test(1,2,3).forward()

# Tester(4,5,6).forward()


# # KEEP - POSSIBLE OPTIMIZATION
# #print(a)           # Maximum of the flattened array
# #print(np.max(a))
# #print(a[1,1:2,0:2])
# answ = np.max(a[0:3,0:3,:], axis = 2)
# answer = [*answ]
# #print(answ)
# #print(answer)
# #print(*answer) # Maxima along the first axis
# #print(np.max(a, axis=1))   # Maxima along the second axis

# #print(np.max(a, where=[False, True], initial=-1, axis=0))

# def counter(number):
    
#     a = 0
#     for i in range(number):
        
#         a += i

#     print(a)

# #counter(5)

#     #cv2.imshow("Hello", img)
# def createFile(dir):
      
#       #Creating a the list of files for where the image files are stored
#       fileList = ""
#       for root,dirs, files in os.walk(dir, topdown = False):
#             for name in files:
#                   fullName = os.path.join(root, name)

#       return fileList

# print(os.listdir())

# #data = np.load("0.jpg")

# image = cv2.imread("0.jpg")
# image = np.copy(image)
# print(image.shape)

# x = "hello"

# #if condition returns True, then nothing happens:
# assert x == "hello"

#if condition returns False, AssertionError is raised:
#assert x == "goodbye"


     
           
           

           

#r_img = img[:,:,0]
#g_img = img[:,:,1]
#b_img = img[:,:,2]

#merge_img = cv2.merge([r_img,g_img,b_img])

#print(img.shape)

#cv2.imshow("New Image", merge_img)
#cv2.imshow("Red", r_img)
#cv2.imshow("Blue", b_img)
#cv2.imshow("Green", g_img)

#print(r_img[100:200,100:200].max())



#Amazon Test

# input = [4,3,2,1,4,2,1,1,1,3,2,2]

# def operations(input):

#     stack = []
#     ops = 0

#     for loc in input:

#         if len(stack) >= 1:

#             if loc not in stack:
                
#                 for i in stack:

#                     if loc != i:

#                         stack.remove(i)
#                         ops = ops + 1
#                         break
#             else:
#                 stack.append(loc)
            
#         else: 
#             stack.append(loc)

#     ops += len(stack)

#     return ops

# print(operations(input))

arr = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])

arr3d = np.array([[[1,2,3],[4,21,6]],[[7,8,9],[10,11,12]]])

arr2 = np.array([[1,2,7],[3,4,8]])

arr3 = np.array([[2,2],[1,1]])

arr0 = np.array(None)

padarr = np.pad(arr, ((2,2),(1,1)))
#print(padarr)

# def dilate(array, stride):
#     H, W = array.shape
#     s = stride
#     # New dimensions: (H - 1) * s + 1
#     dilated = np.zeros(((H - 1) * s + 1, (W - 1) * s + 1), dtype=array.dtype)
#     dilated[::s, ::s] = array
#     return dilated


# result = np.multiply(arr2, arr2)

for i in range(1,1):

    print(i)

mult1 = np.array([[3,4],[4,6]])
mult2 = np.array(([[1,2],[3,4]],[[5,6],[7,8]],[[9,10],[11,12]]))
size = (244,244)

try1 = np.zeros((3,0,0))
con1 = np.array([[1,2],[3,4]])
con2 = np.array([[5,6],[7,8]])
con3 = np.array([[9,10],[11,12]])
list = np.array([con1,con2,con3])
con_list = [con1,con2]
tr1 = np.concatenate(con_list)

zero_array = np.array([])

#padded_con1 = np.pad(con1,((0,0),(1,1),(1,1)))

#con1_width, con1_height = padded_con1.shape

rev= [1,2,3,4,5,6]
rev.reverse()

padMult2 = np.pad(mult2,((0,0),(1,1),(1,1)))
np.pad(mult1, ((0,1),(0,1)))

#concat = np.concatenate(list)

s = np
smult1 = s.split(mult2,[1])

tuple = (3,5)

low, high = tuple

# mult2 = mult2.reshape(2,2,3)


# def forward(input: np.ndarray):

#         mean = np.mean(input,1)
#         var = np.var(input,1)

#         output = (input - mean)/np.sqrt((var**2) + 0.001)
#         print(mean)
#         print(var)

#         return output


#print((np.pad(mult1,((1,1),(1,2)))))
#print(np.pad(con1,(1,1)))
#print(padded_con1[1:con1_width - 1 ,1: con1_height -1])
# print(mult2.shape)



#print(mult2.shape)

# labs = [1,2,3,4,5]

# labels = np.sum(mult1 - mult2, axis = 0)
# labels = np.reshape(labels,2)
# label = np.array([1,2])

#print(label.shape)
# print(arr.shape)


            


def crossCorrelation3D(input: np.ndarray, kernel: np.ndarray, stride: int, mode:str = "valid"):

    inpt = input
    inp_depth, inp_width, inp_height = input.shape
    kernel_depth, kernel_width, kernel_height  = kernel.shape
    #print(f"Shape: {input.shape}")

    w_pad = 0
    h_pad = 0

    if mode == "full":

        #print(f"full kernels {kernel.shape}")
        #print(f"full input{input.shape}")

        w_pad = kernel_width - 1
        h_pad = kernel_height - 1

        padded_input = []

        for i in range(inp_depth):

            padded_input.append(np.pad(inpt[i], (w_pad,h_pad)))

        inpt = np.array(padded_input)
        stride = 1

    
    #print("huh")
    out_height = ((inp_height - kernel_height + (2 * h_pad)) // stride) + 1
    out_width = ((inp_width - kernel_width + (2 * w_pad)) // stride) + 1


    output = np.zeros((out_width,out_height))

    #print(f"Conv input shape {input.shape}")
    #print(f"Conv output Shape {output.shape}")
    #print(f"Conv kernel Shape {kernel.shape}")
    
    for i in range(out_height):

        for j in range(out_width):

            for k in range(kernel_depth):                

                    #print(f"conv:  {i}, {j}, {d}")
                
                    output[i,j] += np.sum(inpt[:, i*stride : i*stride + kernel_width, j*stride : j*stride + kernel_height] * kernel[k])


    #print(f"full output {output.shape}")

    
    return output
        









    
        

 