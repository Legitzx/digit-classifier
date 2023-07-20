from engine import *


def test():
    input = np.array([[-1],
                      [2.1],
                      [0.4]])

    y_desired = np.array([1])

    nn = NeuralNetwork([3, 20, 20, 1])

    print("Before training: ")
    print(nn.feed_forward(input))

    nn.gradient_descent(input, y_desired)

    print("After training: ")
    print(nn.feed_forward(input))


if __name__ == '__main__':
    test()

