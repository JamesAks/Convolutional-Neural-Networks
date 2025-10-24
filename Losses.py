import numpy as np
import cupy as cp

# Loss function determines how much information gain after an iteration or how far off our predictions were from the true values

def logLoss(predictions: np.ndarray| cp.ndarray, true_ys: np.ndarray | cp.ndarray) -> float:



    if type(predictions) == np.ndarray:

        module = np

    else:

        module = cp

    loss = 0

    for i in range(true_ys.shape[0]):

        loss += -true_ys[i] * module.log(predictions[i])

    loss = module.sum(loss)

    return loss


def logLossPrime(predictions: np.ndarray | cp.ndarray, true_ys: list) -> np.ndarray:

    if type(predictions) == np.ndarray:

        module = np

    else:

        module = cp


    preds = module.array(predictions)
    trues = module.array(true_ys)

    #print(f"prediction shape: {preds.shape}")
    #print(f"trues shape: {trues.shape}")

    error = (preds - trues)#/ len(predictions)

    # error = error.reshape(len(predictions[0]),1)

    return error

     