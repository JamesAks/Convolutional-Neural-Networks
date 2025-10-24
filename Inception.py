import numpy as np
from Layer import Layer


class InceptionLayer(Layer):

    # Inception layer consist of multiple branches with multiple convolution layers working in "parrallel".
    # The output of each bracnch is then concatenated (depth-wise) into a single output

    def __init__(self, branches: list, gpu: bool = False):

        self.branches = branches
        self.num_branches = len(branches)
        self.partitions = []

        super().__init__(gpu)

    def forward(self, input: np.ndarray):

        inpt = input
        self.input = input
        self.input_shape = input.shape
        print(f"Inception Input{self.input.shape}")

        self.outputs = []
        self.output_depths = [] 

        for i in range(self.num_branches):

            output = inpt

            for layer in self.branches[i]:

                output = layer.forward(output)

            
            self.outputs.append(output) 
            self.output_depths.append(output.shape[1])

        self.output = self.module.concatenate(self.outputs, axis = 1)

        return self.output
    

    def backward(self, error_grad: np.ndarray, learning_rate: float) -> np.ndarray:

        error_list = []

        ind = 0

        for depth in self.output_depths:

            error_list.append(error_grad[:, ind: + ind + depth, :, :])

            ind += depth


        output = self.module.zeros(self.input.shape)

        for i in range(self.num_branches):

            error = error_list[i]

            self.branches[i].reverse()

            for layer in self.branches[i]:

                error = layer.backward(error, learning_rate)

            output += error

            self.branches[i].reverse()

            
        return output
    

    def update(self, learning_rate: float = 0.09):

        for i in range(self.num_branches):

            for layer in self.branches[i]:

                layer.update(learning_rate)

