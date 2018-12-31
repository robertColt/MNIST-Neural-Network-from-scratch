import numpy as np

def sigmoid(z):
    """
    applies the sigmoid function on the given z
    :param z: a numpy array
    :return:
    """
    return 1/(1 + np.exp(-z))


def softmax(z):
    """
    applies the softmax function on z
    :param z: a numpy array
    :return:
    """
    return np.exp(z) / np.sum(np.exp(z), axis=0)


def loss_cross_entropy(Y, y_actual, examples):
    """
    compute loss with cross entropy
    :param Y: desired output
    :param y_actual: actual output
    :param examples: for averaging
    :return: the value of the cost function
    """

    loss_sum = np.sum(np.multiply(Y, np.log(y_actual)))

    return -(1./examples) * loss_sum