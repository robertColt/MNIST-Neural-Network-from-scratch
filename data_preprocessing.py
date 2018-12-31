import utils
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

ARG_TRAIN_DATA = "train_split"
ARG_TEST_DATA = "test_split"

argParser = argparse.ArgumentParser()
argParser.add_argument("-train", "--{}".format(ARG_TRAIN_DATA), required=True, help="path to train_split")
argParser.add_argument("-test", "--{}".format(ARG_TEST_DATA), required=True, help="path to test_split")
args = vars(argParser.parse_args())

print("Getting data...")
X_train, y_train = utils.get_mnist_csv("data/mnist_train.csv") #X.shape = (60k, 784), y.shape = (60000,)
X_test, y_test = utils.get_mnist_csv("data/mnist_test.csv") #shapes (10000,784) (10000,1)

# normalize the mnist dataset
X_train = X_train/255
X_test = X_test/255

plt.imshow(X_train[20000, :].reshape(28,28), cmap = matplotlib.cm.binary)
# plt.imshow(X_train[:, 20000].reshape(28,28), cmap = matplotlib.cm.binary)
plt.axis("off")
plt.show()

X_train = np.where(X_train >= 0.4, 1.0, 0.0)
X_test = np.where(X_test >= 0.4, 1.0, 0.0)

n_digits = 10
n_samples = y_train.shape[0]

#one hot encoding of the labels
y_train = y_train.reshape(1, n_samples)
y_train_hot_encoded = np.eye(n_digits)[y_train.astype('int32')]
y_train_hot_encoded = y_train_hot_encoded.T.reshape(n_digits, n_samples)

n_samples = y_test.shape[0]
y_test = y_test.reshape(1, n_samples)
y_test_hot_encoded = np.eye(n_digits)[y_test.astype('int32')]
y_test_hot_encoded = y_test_hot_encoded.T.reshape(n_digits, n_samples)

#change shape to match network shape (784,m)
X_train = X_train.T
X_test = X_test.T

plt.imshow(X_train[:, 20000].reshape(28,28), cmap = matplotlib.cm.binary)
# plt.imshow(X_train[:, 20000].reshape(28,28), cmap = matplotlib.cm.binary)
plt.axis("off")
plt.show()

print("Saving splits...")

trainSplitFileName = args[ARG_TRAIN_DATA]
testSplitFileName = args[ARG_TEST_DATA]

utils.saveToFile(trainSplitFileName, (X_train, y_train_hot_encoded))
utils.saveToFile(testSplitFileName, (X_test, y_test_hot_encoded))
utils.saveToFile("data/mnist.dat", (X_train, y_train_hot_encoded, X_test, y_test_hot_encoded))
print("Saved")

print("Finished data preprocess...")

