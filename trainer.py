from network import NNOneHidden, np
import utils
import argparse
import matplotlib.pyplot as plot

#command line arguments for this script
ARG_TRAINED_MODEL_OUT = "save"
ARG_TRAIN_DATA = "train_data"
ARG_BATCH_TRAIN = "batch"

argParser = argparse.ArgumentParser()
argParser.add_argument("-d", "--{}".format(ARG_TRAIN_DATA), required=True, help="path to training data")
argParser.add_argument("-s", "--{}".format(ARG_TRAINED_MODEL_OUT), required=True, help="path to save trained model")
argParser.add_argument("-b", "--{}".format(ARG_BATCH_TRAIN), required=True, help="train with batch or not")
args = vars(argParser.parse_args())

trainDataFileName = args[ARG_TRAIN_DATA]

print("Getting training data from '{}' ...".format(trainDataFileName))
X_train, y_train = utils.loadDataFromFile(trainDataFileName)

print("Training model batch={}...".format(args[ARG_BATCH_TRAIN]))
nnet = NNOneHidden()
epochs = 2500
hidden_units = 50
if args[ARG_BATCH_TRAIN] == "true":
    W, b, train_history = nnet.mini_batch_train(X_train, y_train, epochs=epochs, classes=10, batch_size=256,
                                                learning_rate=4, hidden_units=hidden_units)
else:
    W, b, train_history = nnet.train(X_train, y_train, epochs=epochs, classes=10, learning_rate=4, hidden_units=hidden_units)

print("Finished training...")

#save model i.e weights and biases to the file
modelOutputFileName = args[ARG_TRAINED_MODEL_OUT]
print("Saving model to '{}' ...".format(modelOutputFileName))
utils.saveToFile(modelOutputFileName, (W, b))

historyOutputFile = "{}_history.png".format(modelOutputFileName)
print("Plotting history and saving...")

#plot the history of the training as a function of the epoch to the loss
x_axis_numbers = 10
plot.style.use("ggplot")
plot.figure()
plot.plot(range(epochs), train_history)
plot.title("Training loss")
plot.xlabel("Epoch #")
plot.ylabel("Loss")
# plot.xticks(np.arange(1, epochs+1, 1))
# plot.yticks(np.arange(train_history[-1], train_history[0], 0.1))
plot.savefig(historyOutputFile)


