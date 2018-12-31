import eel
import numpy as np
import argparse
import utils
from network import NNOneHidden

ARG_TRAINED_MODEL_IN = "model"

argParser = argparse.ArgumentParser()
argParser.add_argument("-m", "--{}".format(ARG_TRAINED_MODEL_IN), required=True, help="path to trained model")
args = vars(argParser.parse_args())

trainedModelFileName = args[ARG_TRAINED_MODEL_IN]

print("Loading model from '{}' ...".format(trainedModelFileName))
W, b = utils.loadDataFromFile(trainedModelFileName)

print("Fitting model...")

eel.init("web")

@eel.expose
def classify_image(X):
    pixels = list(map(int, X.split(" ")))
    #inputData = np.array([1-x for x in pixels])
    inputData = np.array(pixels).reshape(784,1)
    # plt.imshow(inputData.reshape(28, 28), cmap=matplotlib.cm.binary)
    # plt.axis("off")
    # plt.show()
    # utils.display(inputData)
    nnet = NNOneHidden()
    nnet.fit(W, b)
    classification = nnet.classify(inputData)
    prediction = np.argmax(classification, axis=0)
    confidence = (classification[prediction][0][0] * 100)
    print(prediction)
    eel.on_prediction_ready(str(prediction[0]) + " [{:.2f}% confidence]".format(confidence))

eel.start("main.html", block=False)

while True:
    eel.sleep(10)