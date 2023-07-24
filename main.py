from engine import *
from mnist_loader import *

def test():
    X_train, Y_train, X_test, Y_test = get_mnist_data() # , X_test, Y_test

    nn = NeuralNetwork([784, 100, 10])

    # Train on training data
    nn.mini_batch_gradient_descent(X_train, Y_train, 50, 10, X_test, Y_test)

    # Evaluate on test data
    nn.evaluate(X_test, Y_test)


if __name__ == '__main__':
    test()

