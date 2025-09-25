import numpy as np
import cupy as cp
from DataLoader import DataLoader
from Display import EvalLogger

class Model():

    def __init__(self, model_name):

        self.model_name = model_name
        self.network = []

    def predict(self, inputs: np.ndarray):

        output = inputs

        for layer in self.network:

            output = layer.forward(output)
        
        return output
    


    def train(self, loss, lossPrime, data_loader: DataLoader, eval_logger: EvalLogger, learning_rate : float = 0.01, epochs: int = 100, fold: int = 1):

        eval_logger.createSheet(self.model_name)
        
        for e in range(epochs):

            loss_sum = 0
            batch = 0

            for x,y in data_loader:

                outputs = self.predict(x)

                print(f"Final output size: {outputs.size}" )
                print(f"Input Type: {type(x)} and Value Type: {type(y)}")


                loss_sum += float(loss(outputs, y))

                error_gradient = lossPrime(outputs, y)

                acc = float(self.findAccuracy(outputs, y))

                for layer in reversed(self.network):

                    output = layer.backward(error_gradient, learning_rate)
                    layer.update()

                    error_gradient = output

                print(loss_sum)

                batch += 1

                eval_logger.log(self.model_name,fold,(e + 1), (batch), loss_sum, acc)
                eval_logger.save()
            
            print(f"Epoch: {e + 1} --- Loss: {loss_sum}")

    def test(self, data_loader: DataLoader, eval_logger: EvalLogger, index: int = 1):

        predictions = []
        true_ys = []

        for x,y in data_loader:

            predictions.extend(self.predict(x))
            true_ys.extend(y)

        acc = self.findAccuracy(predictions,true_ys)
        eval_logger.testLog(self.model_name, acc, index)



    def findAccuracy(self, outputs, y) -> float:

        if type(outputs) == np.ndarray:
            module = np

        else:
            module = cp

        predicts = module.argmax(outputs, axis = 1)
        correct = (module.argmax(y, axis = 1))


        return module.mean(predicts == correct)
    



