import argparse
from sklearn.metrics import classification_report
from network import NNOneHidden
import utils
import numpy as np

#command line arguments for this script
ARG_TRAINED_MODEL_IN = "model"
ARG_TEST_DATA = "test_data"

argParser = argparse.ArgumentParser()
argParser.add_argument("-m", "--{}".format(ARG_TRAINED_MODEL_IN), required=True, help="path to trained model")
argParser.add_argument("-t", "--{}".format(ARG_TEST_DATA), required=True, help="path to test data")
args = vars(argParser.parse_args())

trainedModelFileName = args[ARG_TRAINED_MODEL_IN]
testDataFileName = args[ARG_TEST_DATA]

print("Loading model from '{}' ...".format(trainedModelFileName))
W, b = utils.loadDataFromFile(trainedModelFileName)

print("Loading test data from '{}' ...".format(testDataFileName))
X_test, y_test = utils.loadDataFromFile(testDataFileName)
print("Test data shapes", X_test.shape, y_test.shape)

print("Fitting model and getting predictions...")
nnet = NNOneHidden()
nnet.fit(W, b)

classification = nnet.classify(X_test)
predictions = np.argmax(classification, axis=0)
labels = np.argmax(y_test, axis=0)

correct_predictions = np.sum(predictions == labels) #compute how many predictions were correct
print("Performance : {}/{} correct".format(correct_predictions, predictions.shape[0]))
print("\nClassification report\n")
print(classification_report(predictions, labels))
