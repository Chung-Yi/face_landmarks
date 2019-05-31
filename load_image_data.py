import numpy as np
import _pickle as cPickle


def load_data():
    with open('bin/train_image', 'rb') as f:
        dic = cPickle.load(f)
        train_images = dic['data']
        train_images_size = len(train_images)
        train_images = np.reshape(train_images,
                                  (train_images_size, 200, 200, 3))
        train_landmarks = dic['landmarks']

    with open('bin/test_image', 'rb') as f:
        dic = cPickle.load(f)
        test_images = dic['data']
        test_images_size = len(test_images)
        test_images = np.reshape(test_images, (test_images_size, 200, 200, 3))
        test_landmarks = dic['landmarks']

    return (train_images, train_landmarks), (test_images, test_landmarks)