from engine import *
from mnist_loader import *

def test():
    X_train, Y_train, X_test, Y_test = get_mnist_data() # , X_test, Y_test

    nn = NeuralNetwork([784, 100, 100, 10])

    # Train on training data
    nn.stochastic_gradient_descent(35, X_train, Y_train)

    # Evaluate on test data
    nn.evaluate(X_test, Y_test)


if __name__ == '__main__':
    test()

