import numpy as np
import _pickle as cPickle


def load_data():
    with open('bin/train_image_0', 'rb') as f:
        dic = cPickle.load(f)
        train_images_0 = dic['data']
        train_images_size_0 = len(train_images_0)
        train_images_0 = np.reshape(train_images_0,
                                    (train_images_size_0, 300, 300, 3))
        train_landmarks_0 = dic['landmarks']

    with open('bin/train_image_1', 'rb') as f:
        dic = cPickle.load(f)
        train_images_1 = dic['data']
        train_images_size_1 = len(train_images_1)
        train_images_1 = np.reshape(train_images_1,
                                    (train_images_size_1, 300, 300, 3))
        train_landmarks_1 = dic['landmarks']

    with open('bin/train_image_2', 'rb') as f:
        dic = cPickle.load(f)
        train_images_2 = dic['data']
        train_images_size_2 = len(train_images_2)
        train_images_2 = np.reshape(train_images_2,
                                    (train_images_size_2, 300, 300, 3))
        train_landmarks_2 = dic['landmarks']

    with open('bin/test_image', 'rb') as f:
        dic = cPickle.load(f)
        test_images = dic['data']
        test_images_size = len(test_images)
        test_images = np.reshape(test_images, (test_images_size, 300, 300, 3))
        test_landmarks = dic['landmarks']

    train_images = np.concatenate(
        (train_images_0, train_images_1, train_images_2), axis=0)
    train_landmarks = np.concatenate(
        (train_landmarks_0, train_landmarks_1, train_landmarks_2), axis=0)

    return (train_images, train_landmarks), (test_images, test_landmarks)