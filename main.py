import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, balanced_accuracy_score, \
    roc_auc_score, log_loss
import keras
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.utils import plot_model


def explore_data_shape(x, y):
    print(f'Images: {str(x.shape[0])}, Dimensions: {str(x.shape[1])}x{str(x.shape[2])}')
    print(f'Labels: {str(y.shape)}')
    return x.shape, y.shape


def plot_random_image(x):
    select_image = np.random.randint(len(x))
    print(f'Selected image: {select_image}')
    plt.imshow(x[0], cmap="Greys")
    plt.show()
    return select_image


def normalise_data(x):
    return x / np.max(x)


def reshape_channel(x):
    return x.reshape(x.shape[0], x.shape[1], x.shape[2], 1)  # 1 for greyscale, 3 for rgb


def one_hot(y):
    return keras.utils.to_categorical(y, np.max(y) + 1)


def cnn(x, y):
    # Initialise CNN
    model = Sequential()
    # Input layer
    model.add(Conv2D(64, (3, 3), input_shape=(x.shape[1], x.shape[2], 1)))  # 64 filters (output space), 3x3 convolution
    # BatchNormalization() aids with overfitting, according to authors and Andrew Ng it should be applied immediately
    # before activation function (non-linearity)
    # model.add(BatchNormalization())
    model.add(Activation('relu'))  # rectified linear unit (fire or not)
    model.add(MaxPooling2D(pool_size=(2, 2)))  # maximum value for each patch on feature map reduced by 2x2 pool_size
    # Hidden layer
    model.add(Conv2D(64, (3, 3)))
    # model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Output Layer
    model.add(Flatten())  # flattens the input
    model.add(Dense(64))  # regular densely connected NN layer, no activation function means linear activation
    model.add(Dense(y.shape[1]))  # can also do e.g. model.add(Dense(64, activation='tanh'))
    # model.add(BatchNormalization())
    model.add(Activation('softmax'))  # softmax activation function as output, turns into weights that sum to 1
    # Compile
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
    return model


if __name__ == '__main__':
    # Load data
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    # Normalise and shape data
    x_train, x_test = normalise_data(x_train), normalise_data(x_test)
    x_train, x_test = reshape_channel(x_train), reshape_channel(x_test)
    y_train, y_test = one_hot(y_train), one_hot(y_test)
    # Model
    cnn_model = cnn(x_train, y_train)
    cnn_model.fit(x_train, y_train, batch_size=64, epochs=1)
    # Validate
    plot_model(cnn_model, to_file='model.png', show_shapes=True)
    test_loss, test_accuracy = cnn_model.evaluate(x_test, y_test, batch_size=64, verbose=1)
    print('Test loss', test_loss)
    print('Test accuracy', test_accuracy)
    predictions = cnn_model.predict(x_test)
    # TODO: Scikit metrics
