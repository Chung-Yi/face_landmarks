from keras.utils import np_utils
import numpy as np
import timeit
import tensorflow as tf
import _pickle as cPickle
from keras import Sequential
import matplotlib.pyplot as plt
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout
from keras.optimizers import SGD, Adam
from load_image_data import load_data


def cnn_model(x_train):

    input_shape = x_train.shape[1:]

    model = Sequential()
    # layer1
    model.add(
        Conv2D(
            32,
            padding='valid',
            kernel_size=(3, 3),
            activation='relu',
            strides=(1, 1),
            input_shape=input_shape))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    # layer2
    model.add(
        Conv2D(
            64,
            padding='valid',
            kernel_size=(3, 3),
            activation='relu',
            strides=(1, 1)))
    model.add(
        Conv2D(
            64,
            padding='valid',
            kernel_size=(3, 3),
            activation='relu',
            strides=(1, 1)))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    # layer3
    model.add(
        Conv2D(
            64,
            padding='valid',
            kernel_size=(3, 3),
            activation='relu',
            strides=(1, 1)))
    model.add(
        Conv2D(
            64,
            padding='valid',
            kernel_size=(3, 3),
            activation='relu',
            strides=(1, 1)))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    # layer4
    model.add(
        Conv2D(
            128,
            padding='valid',
            kernel_size=(3, 3),
            activation='relu',
            strides=(1, 1)))
    model.add(
        Conv2D(
            128,
            padding='valid',
            kernel_size=(3, 3),
            activation='relu',
            strides=(1, 1)))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding='valid'))
    # layer5
    model.add(
        Conv2D(
            256,
            padding='valid',
            kernel_size=(3, 3),
            activation='relu',
            strides=(1, 1)))
    # layer6
    model.add(Flatten())
    model.add(Dense(1024, activation='relu', use_bias=True))
    model.add(Dense(162, activation=None, use_bias=True))

    model.summary()
    adam = Adam(lr=0.001)
    # sgd = SGD(lr=0.01, momentum=0.9, nesterov=True)
    model.compile(
        loss='mean_squared_error', optimizer=adam, metrics=['accuracy'])

    return model


def SimpleCNN(x_train, withDropout=False):
    input_shape = x_train.shape[1:]
    model = Sequential()
    # layer1
    model.add(
        Conv2D(
            32,
            padding='valid',
            kernel_size=(3, 3),
            activation='relu',
            strides=(1, 1),
            input_shape=input_shape))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    if withDropout:
        model.add(Dropout(0.1))

    # layer2
    model.add(
        Conv2D(
            64,
            padding='valid',
            kernel_size=(3, 3),
            activation='relu',
            strides=(1, 1)))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    if withDropout:
        model.add(Dropout(0.1))

    # layer3
    model.add(
        Conv2D(
            128,
            padding='valid',
            kernel_size=(3, 3),
            activation='relu',
            strides=(1, 1)))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    if withDropout:
        model.add(Dropout(0.1))

    # layer4
    model.add(Flatten())

    model.add(Dense(1024, activation='relu', use_bias=True))
    if withDropout:
        model.add(Dropout(0.1))

    model.add(Dense(1024, activation='relu', use_bias=True))
    if withDropout:
        model.add(Dropout(0.1))

    model.add(Dense(162, activation=None, use_bias=True))

    model.summary()
    # adam = Adam(lr=0.001)
    sgd = SGD(lr=0.01, momentum=0.9, nesterov=True)
    model.compile(
        loss='mean_squared_error', optimizer=adam, metrics=['accuracy'])


def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train history')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


def plot_images_labels_prediction(images, prediction, num=None):
    fig = plt.gcf()
    fig.set_size_inches(12, 14)
    if num > 9:
        num = 9
    for i in range(num):
        marks = np.reshape(prediction[i], (-1, 2))
        x = np.array([mark[0] for mark in marks])
        y = np.array([mark[1] for mark in marks])
        ax = plt.subplot(3, 3, i + 1)
        ax.imshow(images[i], cmap="gray")
        ax.scatter(x, y)
        ax.set_title("picture " + str(i + 1))
    plt.show()


def main():
    # load train data and test data from bin
    (x_train, y_train), (x_test, y_test) = load_data()
    print(y_test)
    # # flatten y_train and y_test
    # y_train = np.array([y.flatten() for y in y_train])
    # y_test = np.array([y.flatten() for y in y_test])

    # # normalize
    # x_train = x_train.astype('float32')
    # x_test = x_test.astype('float32')
    # x_train = x_train / 255
    # x_test = x_test / 255

    # model = cnn_model(x_train)
    # train_history = model.fit(
    #     x_train,
    #     y_train,
    #     validation_split=0.2,
    #     epochs=2,
    #     batch_size=32,
    #     verbose=2,
    #     steps_per_epoch=50)

    # predict = model.predict(x_test)
    # plot_images_labels_prediction(x_test, predict, len(x_test))


if __name__ == "__main__":
    main()