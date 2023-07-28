from engine import *
from mnist_loader import *

def test():
    X_train, Y_train, X_test, Y_test = get_mnist_data() # , X_test, Y_test

    regularization_param = 0.0001
    starting_learning_rate = 0.001
    learning_rate_decay = 0.00001

    nn = NeuralNetwork([784, 100, 10], regularization_param, starting_learning_rate, learning_rate_decay)

    # Train on training data
    nn.mini_batch_gradient_descent(X_train, Y_train, 200, 10, X_test, Y_test)

    # Evaluate on test data
    nn.evaluate(X_test, Y_test)


if __name__ == '__main__':
    test()

