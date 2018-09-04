import matplotlib.pyplot as plt
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist_digits = input_data.read_data_sets('MNIST_data', one_hot=True)


def train_data(num):
    print('Training examples in dataset: ' + str(mnist_digits.train.images.shape))
    training_images = mnist_digits.train.images[:num, :]
    print('Training images loaded: ' + str(training_images.shape))
    training_labels = mnist_digits.train.labels[:num, :]
    print('Training labels loaded: ' + str(training_labels.shape))
    print('')

    return training_images, training_labels


def test_data(num):
    print('Test examples in dataset: ' + str(mnist_digits.test.images.shape))
    test_images = mnist_digits.test.images[:num, :]
    print('Test images loaded: ' + str(test_images.shape))
    test_labels = mnist_digits.test.labels[:num, :]
    print('Test labels loaded: ' + str(test_labels.shape))

    return test_images, test_labels
