import os
import numpy as np
import timeit
import tensorflow as tf
import _pickle as cPickle
import matplotlib.pyplot as plt
from keras.backend.tensorflow_backend import set_session
from attrdict import AttrDict
from keras import Sequential
from keras.utils import np_utils
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout
from keras.optimizers import SGD, Adam
from load_image_data import load_data
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

path = os.path.abspath(os.path.dirname(__file__))
model_path = os.path.join(path, 'models')

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.95
config.gpu_options.visible_device_list = "0"
set_session(tf.Session(config=config))


class DataModifier:
    def fit(self, x, y):
        return (NotImplementedError)


class FlipImg(DataModifier):
    def __init__(self, flip_indices=None):
        if flip_indices == None:
            flip_indices = [(34, 52), (42, 44), (62, 70), (72, 90), (78, 84),
                            (96, 108), (98, 106), (118, 110), (138, 144),
                            (140, 160)]

        self.flip_indices = flip_indices

    def fit(self, x_batch, y_batch):
        batch_size = x_batch.shape[0]
        indices = np.random.choice(
            batch_size, int(batch_size / 2), replace=False)
        x_batch[indices] = x_batch[indices, :, ::-1, :]
        y_batch[indices, ::2] = y_batch[indices, ::2] * -1

        for a, b in self.flip_indices:
            y_batch[indices, a], y_batch[indices, b] = (y_batch[indices, b],
                                                        y_batch[indices, a])

        return x_batch, y_batch


class ShiftImg(FlipImg):
    def __init__(self, flip_indices=None, prop=0.1):
        super().__init__(flip_indices)
        self.prop = prop

    def fit(self, x, y):
        x, y = super().fit(x, y)
        x, y = self.shift_image(x, y, self.prop)
        return x, y

    def shift_image(self, x, y, prop):
        for i in range(len(x)):
            x_ = x[i]
            y_ = y[i]
            x[i], y[i] = self.shift_single_image(x_, y_, self.prop)
            try:
                locations = fr.face_locations(x[i])
            except RuntimeError:
                continue

            if len(locations) == 0:
                continue
            if not utils.box_in_image(locations, x[i]):
                continue
        return x, y

    def shift_single_image(self, x_, y_, prop):
        # h_shift_max = int(x_.shape[0] * prop)
        # w_shift_max = int(x_.shape[1] * prop)

        h_shift_max = 4
        w_shift_max = 4

        w_keep, w_assign, w_shift = self.random_shift(w_shift_max)
        h_keep, h_assign, h_shift = self.random_shift(h_shift_max)
        x_[h_assign[0]:h_assign[1], w_assign[0]:
           w_assign[1], :] = x_[h_keep[0]:h_keep[1], w_keep[0]:w_keep[1], :]

        y_[0::2] = y_[0::2] - (w_shift / 200) / float(x_.shape[1] / 2.0)
        y_[1::2] = y_[1::2] - (h_shift / 200) / float(x_.shape[0] / 2.0)

        return x_, y_

    def random_shift(self, shift_range, n=200):
        shift = np.random.randint(-shift_range, shift_range)
        if shift < 0:
            keep = self.shift_left(n, shift)
            assign = self.shift_right(n, shift)
        else:
            keep = self.shift_right(n, shift)
            assign = self.shift_left(n, shift)

        return keep, assign, shift

    def shift_left(self, n, shift):
        shift = np.abs(shift)
        start = 0
        end = n - shift
        return (start, end)

    def shift_right(self, n, shift):
        shift = np.abs(shift)
        start = shift
        end = n
        return (start, end)


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
    adam = Adam(lr=0.001)
    sgd = SGD(lr=0.01, momentum=0.9, nesterov=True)
    model.compile(
        loss='mean_squared_error', optimizer=adam, metrics=['accuracy'])
    return model


def fit(model,
        modifier,
        train,
        validation,
        batch_size=32,
        epochs=2000,
        print_every=10,
        patience=np.Inf):

    x_train, y_train = train
    x_val, y_val = validation

    generator = ImageDataGenerator()
    history = {"accuracy": [], "val_acc": [], "loss": [], "val_loss": []}

    for e in range(epochs):
        if e % print_every == 0:
            print("Epoch {:4}/{}:".format(e + 1, epochs))

        batches = 0
        loss_epoch = []
        acc_epoch = []
        for x_batch, y_batch in generator.flow(
                x_train, y_train, batch_size=batch_size):
            x_batch, y_batch = modifier.fit(x_batch, y_batch)
            train_hist = model.fit(x_batch, y_batch, verbose=False, epochs=1)

            loss_epoch.extend(train_hist.history['loss'])
            # acc_epoch.append(train_hist.history['accuracy'])
            batches += 1
            if batch_size >= len(x_train) / batch_size:
                break

            loss = np.mean(loss_epoch)
            acc = np.mean(acc_epoch)
            history["loss"].append(loss)
            history["accuracy"].append(acc)

            y_pred = model.predict(x_val)
            val_loss = np.mean((y_pred - y_val)**2)
            history["val_loss"].append(val_loss)

            if e % print_every == 0:
                print("loss - {:6.5f}, val_loss - {:6.5f}".format(
                    loss, val_loss))

            min_val_loss = np.min(history["val_loss"])

            if patience is not np.Inf:
                if np.all(min_val_loss < np.array(history["val_loss"])
                          [-patience:]):
                    break
    hist = AttrDict({'history': history})
    return hist


def show_train_history(train_history, train, validation, name, plt):
    plt.plot(train_history.history[train], label='train: ' + name)
    plt.plot(train_history.history[validation], label='validation: ' + name)


def plot_images_labels_prediction(images, prediction, num=None):
    fig = plt.gcf()
    fig.set_size_inches(12, 14)
    if num > 9:
        num = 9
    for i in range(num):
        r = np.random.choice(images.shape[0], replace=False)
        marks = np.reshape(prediction[r], (-1, 2))
        x = np.array([mark[0] * 200 for mark in marks])
        y = np.array([mark[1] * 200 for mark in marks])
        ax = plt.subplot(3, 3, i + 1)
        ax.imshow(images[r], cmap="gray")
        ax.scatter(x, y)
        ax.set_title("picture " + str(r))
    plt.show()


def plot_all_models(images, *args):
    fig = plt.figure(figsize=(4, 10))
    fig.subplots_adjust(
        hspace=0.001, wspace=0.001, left=0, right=1, bottom=0, top=1)
    num_images = 5
    count = 1
    for i in range(num_images):
        img = np.random.choice(images.shape[0], replace=False)
        ax = fig.add_subplot(num_images, 4, count, xticks=[], yticks=[])
        r = np.random.choice(images.shape[0], replace=False)

        plot_images(images, args[0], ax, r)
        if count < 5:
            ax.set_title('model_1')

        count += 1
        ax = fig.add_subplot(num_images, 4, count, xticks=[], yticks=[])
        plot_images(images, args[1], ax, r)
        if count < 5:
            ax.set_title('model_2')

        count += 1
        ax = fig.add_subplot(num_images, 4, count, xticks=[], yticks=[])
        plot_images(images, args[2], ax, r)
        if count < 5:
            ax.set_title('model_3')

        count += 1
        ax = fig.add_subplot(num_images, 4, count, xticks=[], yticks=[])
        plot_images(images, args[3], ax, r)
        if count < 5:
            ax.set_title('model_4')

        count += 1
    plt.show()


def plot_images(images, prediction, ax, r):
    marks = np.reshape(prediction[r], (-1, 2))
    x = np.array([mark[0] * 200 for mark in marks])
    y = np.array([mark[1] * 200 for mark in marks])
    ax.imshow(images[r])
    ax.scatter(x, y)


def plot_sample(X, y, axs):
    marks = np.reshape(y, (-1, 2))
    x = np.array([mark[0] * 200 for mark in marks])
    y = np.array([mark[1] * 200 for mark in marks])
    axs.imshow(X.reshape(200, 200, 3))
    axs.scatter(x, y)


def save_model(model, name):
    '''
    save model architecture and model weights
    '''

    json_string = model.to_json()
    with open(os.path.join(models_path, name + '_architecture.json'),
              'w') as f:
        f.write(json_string)
    model.save_weights(model_path, name + '_weights.h5')
    model.save(model_path, name + '.h5')


def main():
    # load train data and test data from bin
    (x_train, y_train), (x_test, y_test) = load_data()

    # # flatten y_train and y_test
    y_train = np.array([y.flatten() for y in y_train])
    y_test = np.array([y.flatten() for y in y_test])

    # # normalize
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train = x_train / 255
    x_test = x_test / 255

    # model1
    model1 = cnn_model(x_train)
    train_history_1 = model1.fit(
        x_train,
        y_train,
        validation_split=0.2,
        epochs=2000,
        batch_size=32,
        verbose=2)

    # model2
    model2 = SimpleCNN(x_train)
    train_history_2 = model2.fit(
        x_train,
        y_train,
        validation_split=0.2,
        epochs=2000,
        batch_size=32,
        verbose=2)

    # model3
    modifier = FlipImg()
    model3 = SimpleCNN(x_train)
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.2, random_state=42)
    train_history_3 = fit(
        model3,
        modifier,
        train=(x_train, y_train),
        validation=(x_val, y_val),
        batch_size=32,
        epochs=2000,
        print_every=100)

    # model4
    model4 = SimpleCNN(x_train, withDropout=True)
    train_history_4 = model4.fit(
        x_train,
        y_train,
        validation_split=0.2,
        epochs=2000,
        batch_size=32,
        verbose=2)

    plt.figure(figsize=(8, 8))
    show_train_history(train_history_1, 'loss', 'val_loss', 'model1', plt)
    show_train_history(train_history_2, 'loss', 'val_loss', 'model2', plt)
    show_train_history(train_history_3, 'loss', 'val_loss', 'model3', plt)
    show_train_history(train_history_4, 'loss', 'val_loss', 'model4', plt)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()

    predict1 = model1.predict(x_test)
    # plot_images_labels_prediction(x_test, predict1, len(x_test))

    predict2 = model2.predict(x_test)
    # plot_images_labels_prediction(x_test, predict2, len(x_test))

    predict3 = model3.predict(x_test)
    # plot_images_labels_prediction(x_test, predict3, len(x_test))
    predict4 = model4.predict(x_test)

    plot_all_models(x_test, predict1, predict2, predict3, predict4)

    save_model(model1, 'cnn')
    save_model(model2, 'simplecnn')
    save_model(model3, 'flipcnn')
    save_model(model4, 'simplecnn_dropout')


if __name__ == "__main__":
    main()