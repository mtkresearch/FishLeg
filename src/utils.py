import torch
import numpy as np
import math
import os
import json


def validateJSon(json_data):
    """Vaildate the json data is valid"""
    try:
        data = json.loads(json_data)
        return True
    except ValueError:
        return False
    return True


def MSE_accuracy(predictions, labels):
    """Return the mean squared"""
    return ((predictions - labels) ** 2).mean()


def bernoulli_accuracy(predictions, labels):
    predictions = torch.sigmoid(predictions)
    return MSE_accuracy(predictions, labels)


def class_accuracy(predictions, labels):
    y = torch.argmax(predictions, dim=1)
    return np.mean(float(y.eq(labels)))


def mse_for_whole_dataset(fl, theta, D, batch_size, nll_name):
    x, y = D
    N1 = np.min(np.array([batch_size, len(x)]))
    i = 0
    list_acc = []
    while i + N1 <= int(len(x)):
        if nll_name == "Gaussian":
            list_acc.append(
                MSE_accuracy(fl.net.forward(theta["net"], x[i : i + N1]), y[i : i + N1])
            )
        elif nll_name == "SoftMax":
            list_acc.append(
                class_accuracy(
                    fl.net.forward(theta["net"], x[i : i + N1]), y[i : i + N1]
                )
            )
        elif nll_name == "Bernoulli":
            list_acc.append(
                bernoulli_accuracy(
                    fl.net.forward(theta["net"], x[i : i + N1]), y[i : i + N1]
                )
            )
        else:
            print("This test error is not implemented")
            list_acc.append(None)
            break

        i += N1
    return sum(list_acc) / len(list_acc)
