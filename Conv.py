import numpy as np
import cupy as cp




def crossCorrelation3D(input: np.ndarray, kernel: np.ndarray, stride: int):

    inpt = input

    batch_size, inp_depth, inp_width, inp_height = input.shape
    kernel_depth, _, kernel_width, kernel_height  = kernel.shape

    output_height = (inp_height -  kernel_height) // stride + 1
    output_width = (inp_width - kernel_height) // stride + 1

    output = np.zeros((batch_size, kernel_depth, output_width, output_height))

    for n in range(batch_size):
        for k in range(kernel_depth):

            for j in range(output_height):

                for i in range(output_width):

                    region = inpt[n, :, i*stride : i*stride + kernel_width, j*stride : (j*stride) + kernel_height]

                    output[n,k,i,j] += np.sum(inpt[n, :, i*stride : i*stride + kernel_width, j*stride : (j*stride) + kernel_height] * kernel[k])

    #print(f"Shape: {input.shape}")

    # w_pad = 0
    # h_pad = 0

    # if mode == "full":

    #     #print(f"full kernels {kernel.shape}")
    #     #print(f"full input{input.shape}")

    #     w_pad = kernel_width - 1
    #     h_pad = kernel_height - 1

    #     inpt = (np.pad(inpt, ((0,0),(w_pad,h_pad),(w_pad,h_pad))))

    #     out_height = ((inp_height - kernel_height + (2 * h_pad))) + 1
    #     out_width = ((inp_width - kernel_width + (2 * w_pad))) + 1

    #     if out_height % stride != 0:

    #         inpt = np.pad(inpt,(((0,0),0,0),(0,1),(0,1)))

    #         out_height += 1
    #         out_width += 1

    #     stride = 1

    # else:

    #     #print("huh")
    #     out_height = ((inp_height - kernel_height + (2 * h_pad)) // stride) + 1
    #     out_width = ((inp_width - kernel_width + (2 * w_pad)) // stride) + 1

  

    # if output_size != 0:

    #     out_width,out_height = output_size,output_size


    # output = np.zeros((out_width,out_height))

    # print(f"Conv input shape {inpt.shape}")
    # #print(f"Conv output Shape {output.shape}")
    # print(f"Conv kernel Shape {kernel.shape}")
    # #h = 0
    # for i in range(out_height):
    #    # print(f"Height: {h}")
    #     #h += 1
    #     #w = 1
    #     for j in range(out_width):
    #         #print(f"Width {w}")

    #                 #print(f"conv:  {i}, {j}, {d}")
    #                 #print(f"Depth: {i}" )
    #                 output[i,j] += np.sum(inpt[:, :, i*stride : i*stride + kernel_width, j*stride : (j*stride) + kernel_height] * kernel)

                    
    #         #w += 1

    # #print(f"full output {output.shape}")

    return output

def conv_back(input: np.ndarray, kernel: np.ndarray, error: np.ndarray, stride: int)-> tuple[np.ndarray,np.ndarray]:

    inpt = input

    batch_size, inp_depth, inp_width, inp_height = input.shape
    kernel_depth, kernel_d, kernel_width, kernel_height  = kernel.shape
    b, error_depth, error_width, error_height = error.shape


    weights_error = np.zeros((kernel_depth, kernel_d, kernel_width, kernel_height))
    input_error = np.zeros((batch_size, inp_depth, inp_width, inp_height))

    for n in range(batch_size):

        b_input = inpt[n]
        b_input_error = input_error[n]

        for j in range(error_height):

            for i in range(error_width):
                    
                    for k in range(error_depth):

                        w_start = i*stride
                        w_end = i*stride + kernel_width
                        h_start = j*stride
                        h_end = j*stride + kernel_height

                        region = b_input[:, w_start : w_end, h_start : h_end]

                        weights_error[k, :, :, :] += region * error[n, k, i, j]

                        b_input_error[:, w_start : w_end, h_start : h_end] += kernel[k, :, :, :] * error[n, k, i, j]

    return (weights_error, input_error)

def depth_conv(input: np.ndarray, kernel: np.ndarray, stride: int = 1):

    inpt = input

    batch_size, inp_depth, inp_width, inp_height = input.shape
    f,kernel_depth, kernel_width, kernel_height  = kernel.shape


    output_height = ((inp_height -  kernel_height) // stride) + 1
    output_width = ((inp_width - kernel_height) // stride) + 1

    output = np.zeros((batch_size, inp_depth, output_width, output_height))


    for n in range(batch_size):

        for k in range(inp_depth):

            for j in range(output_height):

                for i in range(output_width):

                    output[n,k,i,j] += np.sum(inpt[n, k, i*stride : i*stride + kernel_width, j*stride : j*stride + kernel_height] * kernel[0,k])
    
    return output


def depth_conv_back(input: np.ndarray, kernel: np.ndarray, error: np.ndarray, stride: int = 1, padding: int = 0):

    inpt = input

    batch_size, inp_depth, inp_width, inp_height = input.shape
    kernel_depth, kernel_d, kernel_width, kernel_height  = kernel.shape
    b, error_depth, error_width, error_height = error.shape


    weights_error = np.zeros((kernel_depth, kernel_d, kernel_width, kernel_height))
    input_error = np.zeros((batch_size, inp_depth, inp_width, inp_height))

    for n in range(batch_size):



        for j in range(error_height):

            for i in range(error_width):
                    
                    for k in range(inp_depth):

                        w_start = i*stride
                        w_end = i*stride + kernel_width
                        h_start = j*stride
                        h_end = j*stride + kernel_height

                        region = inpt[n, k, w_start : w_end, h_start : h_end]

                        weights_error[0,k] += region * error[n, k, i, j]
                        input_error[n,k, w_start : w_end, h_start : h_end] += error[n, k, i, j] * kernel[0,k] 

    return (weights_error, input_error)



def convolution(kernels: np.ndarray, error_grad: np.ndarray, stride: int, output_shape: tuple = ()):

    batch, chan, error_width, error_height = error_grad.shape

    rot_error = np.flip(error_grad, (2,3))

    w_pad = error_width - 1
    h_pad = error_height - 1

    padded_kernels = np.pad(kernels, ((0,0),(0,0),(w_pad, w_pad), (h_pad, h_pad)))

    input_gradient = crossCorrelation3D(padded_kernels, rot_error, 1)

    return input_gradient


def crossCorrelation2D(input: np.ndarray, kernel: np.ndarray, stride: int, mode:str = "valid", output_size = 0):

    inpt = input
    inp_width,inp_height = input.shape
    kernel_width, kernel_height  = kernel.shape

    w_pad = 0
    h_pad = 0

    if mode == "full":

        w_pad = kernel_width - 1
        h_pad = kernel_height - 1

        inpt = np.pad(input,(w_pad,h_pad))
        stride = 1

    out_height = ((inp_height - kernel_height + (2 * h_pad)) // stride) + 1
    out_width = ((inp_width - kernel_width + (2 * w_pad)) // stride) + 1

    if output_size != 0:

        out_height,out_width = output_size,output_size

    output = np.zeros((out_width,out_height))

    for i in range(out_height):

        for j in range(out_width):
                
                print((kernel_width,kernel_height))
                
                output[i,j] += np.sum(inpt[i*stride : i*stride + kernel_width, j*stride : j*stride + kernel_height] * kernel)

    return output


def dialate2D(input: np.ndarray, stride: int):

    if stride == 1:
        return input

    width, height = input.shape
    s = stride

    output_width = ((width-1) * s) + 1
    output_height = ((height-1) * s) + 1

    output = np.zeros((output_width, output_height))

    for i in range(width):
        for j in range(height):

            output[i*stride,j*stride] = input[i,j] 

    return output

# Due to the stride causing inputs to be "skipped" when convolving we need to dialate the error to mimic the share of the error deserving to the parameters

def dialate(input: np.ndarray, stride: int):

    batch_size, depth, width, height = input.shape

    output_width = width + ((width - 1) * (stride-1))
    output_height = height + ((height - 1) * (stride - 1))

    # output_width = ((width-1) * stride) + 1
    # output_height = ((height-1) * stride) + 1
    
    output = np.zeros((batch_size, depth, output_width, output_height))

    output[:, :, ::stride, ::stride] = input

    return output



# def convolution3D(input: np.ndarray, kernel: np.ndarray, stride : int, mode: str = "full",output_shape: tuple = ()):

#     # To my understanding, implementations of Convlution in machine learning libraries differ on whether to rotate the kernel 180 degrees.
#     # In this case ill try to be mathematically accurate and rotate the kernel for convolution. 
#     # After the array needs to be dialated so that the matrix of the error with respect to the input matches the input size.


#     # print(f"Dilated kernel shape: {kernel.shape}")

#     return crossCorrelation3D(input, np.rot90(np.rot90(kernel)), stride, mode)


def convolution2D(input: np.ndarray, kernel: np.ndarray, stride : int, mode: str = "full", output_size: int= 0):

    kernel = dialate2D(kernel,stride)

    return crossCorrelation2D(input, np.rot90(np.rot90(kernel)), stride, mode,output_size)


def pad3D(input: np.ndarray, padding: int):

    inp_depth, w,h = input.shape
    output = np.zeros((inp_depth, w + (2 * padding), h + (2 * padding)))

    for i in range(inp_depth):

        output[i] = np.pad(input[i], (padding,padding))

    return output


# Optimizing - By using im2col we change the image / feature map into a list of arrays.
# Python loops are quite slow in traversing throught images instead we can turn them into list like arrays and multiply them using numpy. This is much quicker and uses less memory.

def im2col(input: np.ndarray | cp.ndarray, kernel_shape: tuple, stride: int = 1, padding: int = 0) -> np.ndarray | cp.ndarray:
        
        module = getModule(input)

        batch_size,inp_depth, inp_width, inp_height = input.shape
        kernel_width, kernel_height = kernel_shape

        paddedInput = module.pad(input, ((0,0),(0,0),(padding,padding), (padding,padding)))
        
        out_width = ((inp_width - kernel_width + (2 * padding))//stride) + 1
        out_height = ((inp_height - kernel_height + (2 * padding))//stride) + 1

        columns = module.zeros((kernel_width * kernel_height * inp_depth, batch_size * out_width * out_height))

        index = 0

        for h in range(out_height):
            for w in range(out_width):

                patch = paddedInput[:,:,w*stride: w*stride + kernel_width, h*stride: h*stride + kernel_height]

                columns[:, index: index + batch_size] = patch.reshape(batch_size, -1).T
                columns += batch_size

        return columns

    


def col2im(columns: np.ndarray, input_shape: tuple, kernel_shape: tuple[int, int], stride: int = 1, padding: int= 0):

    module = getModule(columns)

    batch_size, input_depth, input_width, input_height = input_shape
    kernel_width, kernel_height = kernel_shape

    output_width = ((input_width - kernel_width + (2 * padding))// stride) + 1
    output_height = ((input_height - kernel_height + (2 * padding))// stride) + 1

    index = 0 
    input_im = module.zeros((batch_size,input_depth, input_width + (2 * padding), input_height + (2 * padding)))

    print(f"Columns shape: {columns.shape}")

    for j in range(output_height):

        for i in range(output_width):

            input_im[:, :, i*stride : i*stride + kernel_width, j*stride : j*stride + kernel_height] += columns[ :,index].reshape(input_depth, kernel_width, kernel_height)
            index += 1
    print("Backed")
    
    return input_im[:,:, padding: input_width - padding, padding: input_height - padding]


def im2col_depthwise(input: np.ndarray, kernel_shape: tuple, stride: int = 1, padding: int = 0):

    module = getModule(input)

    batch_size, input_channels, input_height, input_width = input.shape
    kernel_width, kernel_height = kernel_shape

    output_height = (input_height + 2*padding - kernel_height) // stride + 1
    output_width = (input_width + 2*padding- kernel_width) // stride + 1

    # Pad
    X_padded = np.pad(input, ((0,0), (0,0), (padding,padding), (padding,padding)), mode='constant')

    cols = np.zeros((batch_size, input_channels, kernel_width*kernel_height, output_height * output_width))
    
    for y in range(kernel_height):

        y_max = y + stride*kernel_height

        for x in range(kernel_width):
            x_max = x + stride*output_width
            cols[:, :, y*kernel_height + x, :] = X_padded[:, :, y:y_max:stride, x:x_max:stride].reshape(batch_size, input_channels, -1)
    
    return cols  # shape: (N, C, KH*KW, OH*OW)


def col2im_depthwise(error_cols, input_shape, kernel_shape: tuple, stride: int = 1, padding: int =0 ):

    kernel_width, kernel_height = kernel_shape
    batch_size, input_channels, input_width, input_height = input_shape
    output_height = (input_height + 2*padding - kernel_height) // stride + 1
    output_width = (input_width + 2* padding - kernel_width) // stride + 1

    dX_padded = np.zeros((batch_size, input_channels, input_width + 2*padding, input_height+ 2*padding))

    for y in range(kernel_height):
        y_max = y + stride*output_height
        for x in range(kernel_width):
            x_max = x + stride*output_width
            dX_padded[:, :, y:y_max:stride, x:x_max:stride] += \
                error_cols[:, :, y*kernel_width + x, :].reshape(batch_size, input_channels, output_width, output_height)
    
    if padding > 0:

        return dX_padded[:, :, padding:-padding, padding:-padding]
    
    return dX_padded


def getModule(input: np.ndarray | cp.ndarray):

    if type(input) == np.ndarray:

        module = np

    else:

        module = cp

    return module



# def colConv(input: np.ndarray, kernel: np.ndarray, stride: int = 1, padding: int = 0, mode: str = "valid") -> tuple[np.ndarray, np.ndarray]:

#     if kernel.ndim == 4:

#         output_channels, kernel_depth, kernel_width, kernel_height = kernel.shape

#     else:

#         output_channels,kernel_width, kernel_height = kernel.shape

#     kernel_col = kernel.reshape(output_channels, -1)

#     if mode == "full":

#         padding = kernel_width - 1
#         stride = 1

#     input_col = im2col(input, (kernel_width,kernel_height), stride, padding)

#     return input_col,np.dot(kernel_col, input_col)



