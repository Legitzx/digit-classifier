import numpy as np


class NeuralNetwork:
    """
    sizes contains the number of neurons in each layer of the network.
    For example, if the list contains [2, 3, 3, 2], the first layer contains 2 neurons,
    the second layer containing 3 neurons, and so on.
    """

    def __init__(self, sizes):
        self.sizes = sizes

        self.num_layers = len(sizes)

        # All layers, except the input layer, will have biases
        # The shape of the biases matrix will be y rows and 1 column.
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]

        # Each matrix in the array is the weights for each layer.
        # x represents the number of neurons in columns (L-1) and y represents
        # the number of neurons in column (L).
        # Each row is the weights for a particular neuron in the layer
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    """
    a is the input
    """
    def feed_forward(self, a):
        """
         - weights matrix, where each row in weights[i] are all the weights connected to a singular neuron.
         - a, which is initially the input layer, is a column vector of inputs (or activations).
         - after the dot product and addition, the resulting matrix shape represents that of the input (or activation) matrix.
        """
        for w, b in zip(self.weights, self.biases):
            a = sigmoid(np.dot(w, a) + b)

        return a

    def backpropagation(self, input_layer_data, desired_output):
        weightGrad = [np.zeros(w.shape) for w in self.weights]
        biasesGrad = [np.zeros(b.shape) for b in self.biases]

        """
        Forward pass to collect activations (a=..) and (z=..) for each neuron
        """
        activation = input_layer_data
        # just column vectors of activations, where each row represents the activation of a neuron
        activations = [input_layer_data]
        zs = []

        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, activation) + b

            zs.append(z)

            activation = sigmoid(z)
            activations.append(activation)

        """
        Backward pass
        """
        # output node partials must be manually calculated, then rest of layers can run through a loop

        # for w: cost function derivative * sigmoid_deriv * previousLayerActivation
        # for b: cost function derivative * sigmoid_deriv * 1

        basePartial = cost_derivative(activations[-1], desired_output) * sigmoid_derivative(zs[-1])
        weightGrad[-1] = np.dot(basePartial, activations[-2].transpose())
        biasesGrad[-1] = basePartial

        # starts at second-to-last layer (index = -2)
        for l in range(2, self.num_layers):
            # BASE: bias gradients of previous neurons * the previous weights * sigmoid_deriv of current neuron (with current z)
            z = zs[-l] # get the z of current layer, this is a column vector where each row is a z for each neuron

            prevWeights = self.weights[-l + 1]

            sigDeriv = sigmoid_derivative(z)

            basePartial = np.dot(prevWeights.transpose(), basePartial) * sigDeriv

            # weight: BASE * activation of next (LEFT) layer
            weightGrad[-l] = np.dot(basePartial, activations[-l - 1].transpose())

            # bias: BASE
            biasesGrad[-l] = basePartial

        return weightGrad, biasesGrad

    def gradient_descent(self, training_data, desired_output):
        learning_rate = 0.01

        while True:
            output = self.feed_forward(training_data)

            outputCost = cost(output, desired_output)

            #print("Cost: ", outputCost)

            if outputCost < 0.0001:
                print("Training finished.")
                break

            weightGrad, biasesGrad = self.backpropagation(training_data, desired_output)

            for layer in range(1, len(weightGrad)):
                self.weights[layer] = self.weights[layer] - learning_rate * weightGrad[layer]
                self.biases[layer] = self.biases[layer] - learning_rate * biasesGrad[layer]


def cost(output, desired):
    return 0.5 * (output - desired) ** 2


def cost_derivative(output, desired):
    return output - desired


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_derivative(z):
    return np.exp(-z) * sigmoid(z) * sigmoid(z)

"""
Neural Network Steps:
 1. Forward pass
 2. Compute cost function
 3. Backwards propagation
 4. GD
 5. REPEAT UNTIL COST FUNCTION IS NOT AT A MINIMUM
"""