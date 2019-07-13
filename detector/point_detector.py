import numpy as np
import os
import timeit
from keras.models import load_model
from utils import *


class Point:
    def __init__(self, model1_name, model2_name):
        self.model1 = load_model(model1_name)
        self.model2 = model2_name

    def face_landmark(self, face_img, face_image, model_name):

        if 'cnn' in model_name:
            start = timeit.default_timer()
            points = self.model1.predict(face_img)
            end = timeit.default_timer()
            print("Time: {}s".format(end - start))
            points = np.reshape(points, (-1, 2))
        else:
            points = get_81_points(face_image, self.model2)
            points = np.array(points).astype('float32')

        return points
