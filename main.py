import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, balanced_accuracy_score, \
    roc_auc_score, log_loss
import scikitplot as skplt
import keras
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
import pydot
from keras.utils import plot_model
keras.utils.vis_utils.pydot = pydot


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


def reshape(x):
    return x.reshape(x.shape[0], -1)


def one_hot(y):
    return keras.utils.to_categorical(y, np.max(y) + 1)


def model_performance(model, x_train, x_test, y_test):
    y_pred = model.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    # Could use sklearn.metrics.classification_report here, support is no. occurances
    # Precision (sensitivity) & Recall (specificity)
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    # TPR & TNR
    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)

    print(f"n={x_train.shape[1]}, accuracy={accuracy * 100.0:.2f}%, balanced_accuracy={((tpr + tnr) / 2) * 100.0:.2f}%")
    print(f"precision={p:.3f}, recall={r:.3f}")
    print(f"F1 Score={2 * (p * r) / (p + r):.3f}, G-Mean={np.sqrt(p * r):.3f}")

    y_pred_proba = model.predict_proba(x_test)
    # [:, 0] prob of '0', [:, 1] prob of '1'

    # fpr, tpr, thresh = roc_curve(y_test, y_pred_proba[:, 1])
    # print(f"auc={auc(fpr, tpr):.3f}")
    # precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba[:, 1])
    # print(f"pr_auc={auc(recall, precision):.3f}")

    skplt.metrics.plot_roc(y_test, y_pred_proba)
    skplt.metrics.plot_precision_recall(y_test, y_pred_proba)
    skplt.metrics.plot_ks_statistic(y_test, y_pred_proba)
    skplt.metrics.plot_cumulative_gain(y_test, y_pred_proba)
    skplt.metrics.plot_lift_curve(y_test, y_pred_proba)
    # TODO: Scikit metrics


def plot_model_history(metric):
    plt.figure()
    plt.plot(history.history[metric])
    plt.plot(history.history[f'val_{metric}'])
    plt.title(f'model {metric}')
    plt.ylabel(f'{metric}')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


def mlp(x, y):
    """Multilayer Perceptron"""
    # Initialise MLP
    model = Sequential()
    # Input layer
    model.add(Dense(512, activation='relu', input_shape=(x.shape[1],)))
    model.add(Dropout(0.2))  # apply dropout to input, randomly setting a fraction rate of input units to 0 at each
    # update during training time, which helps prevent overfitting
    # Hidden layer
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    # Output layer
    model.add(Dense(y.shape[1], activation='softmax'))
    # Compile
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.RMSprop(),
                  metrics=['accuracy'])
    return model


def cnn(x, y):
    """Convolutional Neural Network"""
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
    x_train, x_test = reshape(x_train), reshape(x_test)  # for mlp
    # x_train, x_test = reshape_channel(x_train), reshape_channel(x_test)  # for cnn
    y_train, y_test = one_hot(y_train), one_hot(y_test)
    # Model
    model = mlp(x_train, y_train)
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=64, epochs=1)
    # Validate
    plot_model(model, to_file='models/model.png', show_shapes=True)
    test_loss, test_accuracy = model.evaluate(x_test, y_test, batch_size=64, verbose=1)
    print(f'Test loss: {test_loss}, Test accuracy: {test_accuracy}')
    plot_model_history('accuracy')
    plot_model_history('loss')
