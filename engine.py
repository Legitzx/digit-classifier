import numpy as np
import random

"""
Simple Neural Network class.

Base steps the neural network takes to learn:
1. Forward pass
2. Calculate cost
3. Backwards propagation
4. Gradient Descent
5. Repeat steps 1-4 until cost function is at a minimum
"""


class NeuralNetwork:
    """
    sizes contains the number of neurons in each layer of the network.
    For example, if the list contains [2, 3, 3, 2], the first layer contains 2 neurons,
    the second layer containing 3 neurons, and so on.
    """

    def __init__(self, sizes, regularization_lambda_param, starting_learning_rate, learning_rate_decay):
        self.starting_learning_rate = starting_learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.regularization_lambda_param = regularization_lambda_param

        self.sizes = sizes

        self.num_layers = len(sizes)

        # All layers, except the input layer, will have biases
        # The shape of the biases matrix will be y rows and 1 column.
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]

        # Each matrix in the array is the weights for each layer.
        # x represents the number of neurons in columns (L-1) and y represents
        # the number of neurons in column (L).
        # Each row is the weights for a particular neuron in the layer
        self.weights = [np.random.randn(y, x) / np.sqrt(x) for x, y in zip(sizes[:-1], sizes[1:])]

    """
    a is the input
    """

    def feed_forward(self, a):
        """
         - weights matrix, where each row in weights[i] are all the weights connected to a singular neuron.
         - a, which is initially the input layer, is a column vector of inputs (or activations).
         - after the dot product and addition, the resulting matrix shape represents that of the input (or activation) matrix.
        """
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            a = sigmoid(np.dot(w, a) + b)

        z = np.dot(self.weights[-1], a) + self.biases[-1]
        a = softmax(z)

        return a

    def backpropagation(self, input_layer_data, desired_output):
        """
        Basic idea of back propagation:
        Do a forward pass and store the z's (z=wx+b) and a's (sigmoid function s(z)) of each neuron in the various layers.
        Then we start the backwards pass. The goal of the backward pass is to get the partial derivative of the cost function
        with respect to each individual weight and bias in the network. Once we have these partial derivatives, we can start
        gradient descent. This allows us to analyze how much each weight and bias contributes to the cost function.
        """

        weightGrad = [np.zeros(w.shape) for w in self.weights]
        biasesGrad = [np.zeros(b.shape) for b in self.biases]

        """
        Forward pass to collect activations (a=..) and (z=..) for each neuron
        """
        activation = input_layer_data
        # just column vectors of activations, where each row represents the activation of a neuron
        activations = [input_layer_data]
        zs = []

        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            z = np.dot(w, activation) + b

            zs.append(z)

            activation = sigmoid(z)
            activations.append(activation)

        z = np.dot(self.weights[-1], activation) + self.biases[-1]
        zs.append(z)
        activations.append(softmax(z))

        """
        Backward pass
        """
        # output node partials must be manually calculated, then rest of layers can run through a loop

        # for w: cost function derivative * sigmoid_deriv * previousLayerActivation
        # for b: cost function derivative * sigmoid_deriv * 1

        # softmax & cross entropy in last layer
        basePartial = cost_delta_output_layer(activations[-1], desired_output)
        weightGrad[-1] = np.dot(basePartial, activations[-2].transpose())
        biasesGrad[-1] = basePartial

        # starts at second-to-last layer (index = -2)
        for l in range(2, self.num_layers):
            # BASE: bias gradients of previous neurons * the previous weights * sigmoid_deriv of current neuron (with current z)
            z = zs[-l]  # get the z of current layer, this is a column vector where each row is a z for each neuron

            prevWeights = self.weights[-l + 1]

            sigDeriv = sigmoid_derivative(z)

            basePartial = np.dot(prevWeights.transpose(), basePartial) * sigDeriv

            # weight: BASE * activation of next (LEFT) layer
            weightGrad[-l] = np.dot(basePartial, activations[-l - 1].transpose())

            # bias: BASE
            biasesGrad[-l] = basePartial

        return weightGrad, biasesGrad

    def mini_batch_gradient_descent(self, inputs, desired_outputs, epochs, mini_batch_size, test_inputs=None,
                                    test_outputs=None):
        """
        The goal of gradient descent is to analyze the gradients and tune each weight and bias to reduce the
        cost. We do this by tuning each w = w - learning_rate*dC/dw (derivative of cost with respect to weight) and the same with
        bias. Here, we use mini-batch gradient descent. Mini-batch gradient descent processes x training examples per batch. In one
        epoch, we go through len(inputs)/x batches. During one batch, we add up the gradients from each training example
        and take the average (we only update the weights/biases after each batch, using the averaged gradients).
        """
        learning_rate = self.starting_learning_rate

        training_data = [(i, o) for i, o in zip(inputs, desired_outputs)]

        for epoch in range(epochs):
            random.shuffle(training_data)

            """
            Within each epoch, we go through the entire training set. We now divide the training set into mini_batches.
            """

            for start_index in range(0, len(inputs), mini_batch_size):
                batch_training_data = training_data[start_index:start_index + mini_batch_size]

                weightGrad = [np.zeros(w.shape) for w in self.weights]
                biasesGrad = [np.zeros(b.shape) for b in self.biases]

                for training_input, desired_output in batch_training_data:
                    curWeightGrad, curBiasGrad = self.backpropagation(training_input, desired_output)

                    weightGrad = [old + new for old, new in zip(weightGrad, curWeightGrad)]
                    biasesGrad = [old + new for old, new in zip(biasesGrad, curBiasGrad)]

                self.weights = [w - (learning_rate / len(batch_training_data)) * (partial + (self.regularization_lambda_param * w))
                                for w, partial in zip(self.weights, weightGrad)]

                self.biases = [b - (learning_rate / len(batch_training_data)) * (partial + (self.regularization_lambda_param * b))
                               for b, partial in zip(self.biases, biasesGrad)]

            self.evaluate(test_inputs, test_outputs)
            self.evaluate(inputs, desired_outputs)

            # decaying learning rate
            learning_rate = self.starting_learning_rate * (1.0 / (1 + self.learning_rate_decay * epoch))

    def evaluate(self, test_data_input, test_data_desired_output):
        """
        Used to pass in test input/output after model has been trained.
        """
        numCorrect = 0

        for input, desired_output in zip(test_data_input, test_data_desired_output):
            calculatedOutput = self.feed_forward(input)

            if np.argmax(calculatedOutput) == np.argmax(desired_output):
                numCorrect += 1

        print("{}/{} {:.1f}% Accuracy".format(numCorrect, len(test_data_input),
                                              (numCorrect / float(len(test_data_input))) * 100.0))


def categorical_cross_entropy_cost(output, desired):
    # Avoid numerical instability by clipping the output values to a small positive value
    output = np.clip(output, 1e-10, 1 - 1e-10)

    # Calculate the categorical cross-entropy cost
    cost = -np.sum(desired * np.log(output))

    return cost


def cost_delta_output_layer(output, desired):
    # Chain rule for softmax & cross entropy simplifies to this nice form - ref: https://www.youtube.com/watch?v=rf4WF-5y8uY
    return output - desired


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))


def softmax(output):
    # subtract the largest element to reduce possible overflow error, does not alter probability distribution
    exp_values = np.exp(output - np.max(output))

    return exp_values / np.sum(exp_values)