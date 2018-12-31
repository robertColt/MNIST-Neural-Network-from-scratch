import numpy as np
from sklearn.datasets import fetch_mldata
import os.path
import pickle


def get_mnist_data():
    """
    :return: a tuple of X and y,
    where X are the mnist images with shape (70000,784) = (num_examples, pixels)
    and y are the labels of the images
    """

    fileNameData = "data/mnist.data"
    if os.path.isfile(fileNameData):
        print("Loading data from local folder...")
        with open(fileNameData, "rb") as mnistData:
            X, y = pickle.load(mnistData)
    else:
        print("Fetching data from server...")
        mnist = fetch_mldata('MNIST original')
        X, y = mnist["data"], mnist["target"]
        with open(fileNameData, "wb") as mnistData:
            pickle.dump((X, y), mnistData, pickle.HIGHEST_PROTOCOL)

    return X, y


def train_test_split(X, y, train_percent=0.75):
    """
    the given data and labels are split by the given percentage
    the data X will be transposed so that each image will be represented by a column

    :param X: the training data shape = (m,n)
    :param y: the labels for the data X shape= (10,m)
    :param train_percent: specifies how much of the data should be used for training
            should be a number in the interval [0, 1]
    :return: the train test split of the given data in the form
            (X_train, y_train, X_test, y_test) with shapes
            ((n, m*train_percent), (10, m*train_percent), (n, m - m*trainpercent), (10, 1 - m*train_percent))
    """
    total_data = X.shape[0]
    num_train = int(train_percent * total_data)
    num_test = total_data - num_train

    X_train, X_test = X[:num_train].T, X[num_train:].T
    y_train, y_test = y[: , :num_train], y[: , num_train:]

    return X_train, y_train, X_test, y_test


def saveToFile(fileName, obj):
    """
    saves the given object to the file specified by the fileName using pickle serialization
    :param fileName: the name of the file
    :param obj: the object to be saved
    :return: None
    """
    with open(fileName, "wb") as file:
        pickle.dump(obj, file, pickle.HIGHEST_PROTOCOL)


def loadDataFromFile(fileName):
    """
    loads data from a file given by fileName using pickle
    :param fileName: the name of the file
    :return: the data in the file
    """
    with open(fileName, "rb") as file:
        return pickle.load(file)



def display(inputData):
    """
    displays an numpy array with 784 entries as a matrix
    :param inputData:
    :return:
    """
    data = list(map(int, inputData))
    for i in range(0, 783, 28):
        print(data[i:i+28])
