from functions import *


class NNOneHidden():
    def __init__(self, hidden_layers=1):
        list_size = hidden_layers + 1
        self.W = [None] * list_size  # weights
        self.b = [None] * list_size  # biases
        self.Z = [None] * list_size  # inputs
        self.A = [None] * list_size  # activations
        self.dW = [None] * list_size  # weight derivatives
        self.db = [None] * list_size  # biases derivatives
        self.V_dW = [None] * list_size #averages over the derivatives of the weights
        self.V_db = [None] * list_size #averages of the derivatives of the biases
        self.X = None  # training data
        self.y = None  # labels
        self.train_history = []  # save epoch:cost pair


    def classify(self, X):
        """
        classifies the given data
        :param X: numpy array with shape (784,1)
        :return: a numpy array with shape (10,1) containing the predictions probability
        of each digit from 0-9
        """
        # z1 = np.dot(self.W[0], X) + self.b[0]
        # # print("z1", z1.shape)
        # a1 = sigmoid(z1)
        # # print("a1", a1.shape)
        # z2 = np.dot(self.W[1], a1) + self.b[1]
        # # print("z2", z2.shape)
        # a2 = softmax(z2)
        # # print("a2", a2.shape)
        # return a2
        self.X = X
        self.feed_forward()

        return self.A[1]


    def fit(self, W, b):
        """
        initializes the weights and biases of the network
        :param W: list of the weights of the network (numpy arrays) -> for 1st layer and hidden layer
        :param b: list of biases of the network (numpy array) -> for 1st layer and hidden layer
        :return: None
        """
        self.W = W
        self.b = b


    def train(self, X, y, epochs, classes, learning_rate=1, hidden_units=64):
        """
        trains the network with the specified parameters
        :param X: training data shape = (n features, m examples)
        :param y: training labels for X (classes, m examples)
        :param epochs: number of training iterations
        :param classes: number of output classes
        :param batch_size: size of the batches
        :param learning_rate: the learning rate
        :param hidden_units: numbe rof neurons in the hidden unit
        :return: the weights and biases for the layers of the network
        """
        self.X = X
        self.y = y
        input_size = X.shape[0]
        train_examples = X.shape[1]

        self.init_weights(hidden_units=hidden_units, input_size=input_size, classes=classes)

        for epoch in range(epochs):
            self.feed_forward()

            self.back_propagation(batch_size=train_examples)

            self.update(learning_rate=learning_rate, beta=0.9)

            self.X = X

            cost = loss_cross_entropy(y, self.A[1], train_examples)
            self.train_history.append(cost)
            print("epoch {}: cost {}".format(epoch, cost))

        print("final cost : ", cost)
        return self.W, self.b, self.train_history


    def mini_batch_train(self, X, y, epochs, classes, batch_size, learning_rate=1, hidden_units=64):
        """
        trains the network with the specified parameters
        :param X: training data shape = (n features, m examples)
        :param y: training labels for X (classes, m examples)
        :param epochs: number of training iterations
        :param classes: number of output classes
        :param batch_size: size of the batches
        :param learning_rate: the learning rate
        :param hidden_units: numbe rof neurons in the hidden unit
        :return: the weights and biases for the layers of the network
        """
        self.X = X
        self.y = y
        input_size = X.shape[0]
        train_examples = X.shape[1]
        batches = -(-train_examples // batch_size)

        self.init_weights(hidden_units=hidden_units, input_size=input_size, classes=classes)

        for epoch in range(epochs):
            shuffle_sequence = np.random.permutation(
                train_examples)  # compute random sequence i.e. shuffle of training examples
            # print("shape shuffle", shuffle_sequence.shape, np.max(shuffle_sequence), np.min(shuffle_sequence[-1]))
            self.X = X[:, shuffle_sequence]
            self.y = y[:, shuffle_sequence]

            for batch in range(batches):
                first = batch * batch_size
                last = min(first + batch_size, train_examples - 1)  # make sure to not go out of range
                batch_size_ = last - first  # actual batch size

                self.X = self.X[:, first:last]
                self.y = self.y[:, first:last]

                self.feed_forward()
                # print(self.A[1].shape)
                self.back_propagation(batch_size=batch_size_)

                self.update(learning_rate=learning_rate, beta=0.9)

            self.X = X
            self.feed_forward()

            cost = loss_cross_entropy(y, self.A[1], train_examples)
            self.train_history.append(cost)
            print("epoch {}: cost {}".format(epoch, cost))

        print("final cost : ", cost)
        return self.W, self.b, self.train_history


    def init_weights(self, hidden_units, input_size, classes):
        """
        initializes the weights and biases of the network with random variables
        """
        # initialize weights and biases
        self.W[0] = np.random.randn(hidden_units, input_size)  # weights for the hidden layer
        #  (from input j(col) into layer i (row)
        self.b[0] = np.zeros((hidden_units, 1))  # bias for the hidden layer 1
        self.W[1] = np.random.randn(classes, hidden_units)  # weights for the output layer
        self.b[1] = np.zeros((classes, 1))  # biases for the output layer

        # reduce the variance of the randomly selected weights to 1/n with n the inputs in the layer

        self.W[0] *= np.sqrt(1. / input_size)
        self.b[0] *= np.sqrt(1. / input_size)
        self.W[1] *= np.sqrt(1. / hidden_units)
        self.b[1] *= np.sqrt(1. / hidden_units)

        self.V_dW[0] = np.zeros(self.W[0].shape)
        self.V_db[0] = np.zeros(self.b[0].shape)
        self.V_dW[1] = np.zeros(self.W[1].shape)
        self.V_db[1] = np.zeros(self.b[1].shape)


    def feed_forward(self):
        """
        performs the feed forward operation of the network
        """
        self.Z[0] = np.matmul(self.W[0], self.X) + self.b[
            0]  # inputs in hidden layer 1 (size (hidden_units, train_examples)
        # represents input from train_example j (col) into hidden neuron i (row)
        self.A[0] = sigmoid(self.Z[0])  # activation of hidden layer 1 -> output from hidden 1

        self.Z[1] = np.matmul(self.W[1], self.A[0]) + self.b[1]  # inputs in output layer
        self.A[1] = softmax(self.Z[1])  # activation for outputlayer


    def back_propagation(self, batch_size):
        """
        backpropagation in the neural network i.e. calculating the partial derivatives
         of the weights and biases with respect to the cost function
        """
        dZ2 = self.A[1] - self.y
        self.dW[1] = (1. / batch_size) * np.matmul(dZ2, self.A[0].T)
        self.db[1] = (1. / batch_size) * np.sum(dZ2, axis=1, keepdims=True)

        dA1 = np.matmul(self.W[1].T, dZ2)
        dZ1 = dA1 * sigmoid(self.Z[0]) * (1 - sigmoid(self.Z[0]))
        self.dW[0] = (1. / batch_size) * np.matmul(dZ1, self.X.T)
        self.db[0] = (1. / batch_size) * np.sum(dZ1, axis=1, keepdims=True)


    def update(self, learning_rate, beta=0.9):
        """
        updates the weights and biases of the network
        :param learning_rate: learning rate to be used
        :return:
        """
        self.V_dW[0] = (beta * self.V_dW[0] + (1. - beta) * self.dW[0])
        self.V_db[0] = (beta * self.V_db[0] + (1. - beta) * self.db[0])
        self.V_dW[1] = (beta * self.V_dW[1] + (1. - beta) * self.dW[1])
        self.V_db[1] = (beta * self.V_db[1] + (1. - beta) * self.db[1])

        self.W[1] = self.W[1] - learning_rate * self.V_dW[1]
        self.b[1] = self.b[1] - learning_rate * self.V_db[1]
        self.W[0] = self.W[0] - learning_rate * self.V_dW[0]
        self.b[0] = self.b[0] - learning_rate * self.V_db[0]
