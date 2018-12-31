import utils
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

ARG_TRAIN_DATA = "train_split"
ARG_TEST_DATA = "test_split"
ARG_PERCENT = "percent"

argParser = argparse.ArgumentParser()
argParser.add_argument("-train", "--{}".format(ARG_TRAIN_DATA), required=True, help="path to train_split")
argParser.add_argument("-test", "--{}".format(ARG_TEST_DATA), required=True, help="path to test_split")
argParser.add_argument("-percent", "--{}".format(ARG_PERCENT), required=True, help="percent of train data")
args = vars(argParser.parse_args())

print("Getting data...")
X, y = utils.get_mnist_data() #X.shape = (70k, 784), y.shape = (70000,)

# normalize the mnist dataset
X = X/255

# plt.imshow(X[20000, :].reshape(28,28), cmap = matplotlib.cm.binary)
# # plt.imshow(X_train[:, 20000].reshape(28,28), cmap = matplotlib.cm.binary)
# plt.axis("off")
# plt.show()

X = np.where(X >= 0.4, 1.0, 0.0)


n_digits = 10
n_samples = y.shape[0]

#one hot encoding of the labels
y = y.reshape(1, n_samples)
y_hot_encoded = np.eye(n_digits)[y.astype('int32')]
y_hot_encoded = y_hot_encoded.T.reshape(n_digits, n_samples)

print("Splitting data...")
#get the train test split of the data with
train_percent = float(args[ARG_PERCENT])
X_train, y_train, X_test, y_test = utils.train_test_split(X, y_hot_encoded, train_percent)

# utils.display(X_train[: , 9000])
print("\n")
# utils.display(X_train[: , 20000])

plt.imshow(X_train[:, 20000].reshape(28,28), cmap = matplotlib.cm.binary)
# plt.imshow(X_train[:, 20000].reshape(28,28), cmap = matplotlib.cm.binary)
plt.axis("off")
plt.show()

print("Saving splits...")
trainSplitFileName = args[ARG_TRAIN_DATA]
testSplitFileName = args[ARG_TEST_DATA]

utils.saveToFile(trainSplitFileName, (X_train, y_train))
utils.saveToFile(testSplitFileName, (X_test, y_test))

print("Finished data preprocess...")

