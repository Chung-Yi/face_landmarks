import tensorflow as tf
import os
import numpy as np
import cv2
import timeit
import face_recognition as fr
from utils import *


class FaceLandmark():
    def __init__(self, model_path=None):
        self.model_path = model_path
        graph_def = tf.GraphDef()
        with tf.gfile.FastGFile(self.model_path, 'rb') as f:
            graph_def.ParseFromString(f.read())
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name='')
            self.sess = tf.Session(
                config=tf.ConfigProto(
                    intra_op_parallelism_threads=0,
                    inter_op_parallelism_threads=0),
                graph=graph)
            self.input_images = self.sess.graph.get_tensor_by_name(
                'conv2d_1_input:0')
            self.output_points = self.sess.graph.get_tensor_by_name(
                'dense_3/BiasAdd:0')

    def predict(self, face_image):
        face_img = np.reshape(face_image, (1, 200, 200, 3))
        face_img = face_img.astype('float32') / 255

        start = timeit.default_timer()
        points = self.sess.run(
            self.output_points, feed_dict={self.input_images: face_img})
        end = timeit.default_timer()
        print('Time: ', end - start)
        return points


def main():
    path = os.path.abspath(os.path.dirname(__file__))
    model_path = os.path.join(path, 'model/cnn_0628_dropout.pb')
    image = os.path.join(path, 'a.jpg')
    image = cv2.imread(image)

    locations = fr.face_locations(image)
    locs = []

    for loc in locations:
        start_x, start_y, end_x, end_y = loc[3], loc[0], loc[1], loc[2]
        locs.append((start_x, start_y, end_x, end_y))

    cut_face_imgs = cut_face(image, locs)

    fl = FaceLandmark(model_path=model_path)

    for face_img in cut_face_imgs:
        face_image = cv2.resize(face_img, (200, 200))
        points = fl.predict(face_image)
        points = np.reshape(points, (-1, 2)) * 200

        draw_landmak_point(face_image, points)

        cv2.imshow('My Image', face_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()