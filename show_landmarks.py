import cv2
import os
import sys
import glob
import math
import imutils
import numpy as np
import face_recognition as fr
from point_detector import Point
from pose_detector import PoseDetector
from utils import *
from keras.models import load_model
from argparse import ArgumentParser
from collections import OrderedDict
from process_image import Image

parser = ArgumentParser()
parser.add_argument('--model_name', default='cnn', help='choose a model')
args = parser.parse_args()

model_name = args.model_name

path = os.path.abspath(os.path.dirname(__file__))
model1_name = os.path.join(path, 'models/cnn_0628.h5')
model2_name = os.path.join(path,
                           'models/shape_predictor_81_face_landmarks.dat')

INPUT_SIZE = 200


def face_remap(shape):
    remapped_image = shape.copy()
    remapped_image[17] = shape[78]
    remapped_image[18] = shape[74]
    remapped_image[19] = shape[79]
    remapped_image[20] = shape[73]
    remapped_image[21] = shape[72]
    remapped_image[22] = shape[80]
    remapped_image[23] = shape[71]
    remapped_image[24] = shape[70]
    remapped_image[25] = shape[69]
    remapped_image[26] = shape[68]
    remapped_image[27] = shape[76]
    remapped_image[28] = shape[75]
    remapped_image[29] = shape[77]
    remapped_image[30] = shape[0]

    return remapped_image


def crop_landmark_face(points, face_image):
    #initialize mask array and draw mask image
    points_int = np.array([[int(p[0]), int(p[1])] for p in points])
    remapped_shape = np.zeros_like(points)
    landmark_face = np.zeros_like(face_image)
    feature_mask = np.zeros((face_image.shape[0], face_image.shape[1]))

    remapped_shape = face_remap(points_int)
    remapped_shape = cv2.convexHull(points_int)

    cv2.fillConvexPoly(feature_mask, remapped_shape[0:31], 1)
    feature_mask = feature_mask.astype(np.bool)
    landmark_face[feature_mask] = face_image[feature_mask]
    return landmark_face


def eyes_images(face, points):
    eye_r_x1 = int(points[42][0])
    eye_r_x2 = int(points[45][0])
    eye_r_y1 = int(
        min(points[42][1], points[43][1], points[44][1], points[45][1]))
    eye_r_y2 = int(
        max(points[42][1], points[47][1], points[46][1], points[45][1]))

    eye_l_x1 = int(points[36][0])
    eye_l_x2 = int(points[39][0])
    eye_l_y1 = int(
        min(points[36][1], points[37][1], points[38][1], points[39][1]))
    eye_l_y2 = int(
        max(points[39][1], points[41][1], points[40][1], points[36][1]))

    eye_l_img = face[eye_l_y1:eye_l_y2, eye_l_x1:eye_l_x2]
    eye_r_img = face[eye_r_y1:eye_r_y2, eye_r_x1:eye_r_x2]

    eye_l_area = eye_l_img.shape[0] * eye_l_img.shape[1]
    eye_l_gray = cv2.cvtColor(eye_l_img, cv2.COLOR_BGR2GRAY)
    eye_l_occ = np.count_nonzero(eye_l_gray < 55) / eye_l_area

    eye_r_area = eye_r_img.shape[0] * eye_r_img.shape[1]
    eye_r_gray = cv2.cvtColor(eye_r_img, cv2.COLOR_BGR2GRAY)
    eye_r_occ = np.count_nonzero(eye_r_gray < 55) / eye_r_area

    return (eye_l_occ, eye_r_occ), (eye_l_img, eye_r_img)


def main():

    # for image_name in glob.glob('test/test_images/*.jpg'):

    image = os.path.join(path, 'curry.jpg')
    image = cv2.imread(image)
    face = image.copy()
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pt = Point(model1_name, model2_name)
    pose = PoseDetector(image)
    im = Image()

    try:
        locations = fr.face_locations(image_rgb)
    except:
        pass

    locs = []

    for loc in locations:
        start_x, start_y, end_x, end_y = loc[3], loc[0], loc[1], loc[2]
        locs.append((start_x, start_y, end_x, end_y))

    cut_face_imgs, delta_locs = cut_face(image_rgb, locs)

    for i, (face_img, delta_locs) in enumerate(zip(cut_face_imgs, delta_locs)):

        if face_img.size == 0:
            continue

        face_resized = cv2.resize(face_img, (INPUT_SIZE, INPUT_SIZE))
        face_reshape = np.reshape(face_resized, (1, INPUT_SIZE, INPUT_SIZE, 3))
        face_normalize = face_reshape.astype('float32') / 255
        points = pt.face_landmark(face_normalize, face_resized, model_name)
        points_for_crop = points.copy()

        if len(locs[i]) == 0:
            points = []

        points_for_crop[:, 0] *= face_img.shape[1]
        points_for_crop[:, 1] *= face_img.shape[0]

        for point in points:
            point[0] *= face_img.shape[1]
            point[1] *= face_img.shape[0]
            point[0] += locs[i][0] - delta_locs[0]
            point[1] += locs[i][1] - delta_locs[1]

        # #######################################################################
        # # detect eyes area to check if landmarks match on face
        eyes_occ, eyes_img = eyes_images(face, points)
        eye_l_occ = eyes_occ[0]
        eye_r_occ = eyes_occ[1]
        eye_l_img = eyes_img[0]
        eye_r_img = eyes_img[1]

        # cv2.imwrite('eyes/2500_eye_left.jpg', eye_l_img)
        # cv2.imshow('eye_left', eye_l_img)
        # cv2.waitKey(0)
        # draw_landmak_point(image, points)
        # cv2.imwrite('eyes/2500_landmark.jpg', image)

        # cv2.imwrite('eyes/2500_eye_right.jpg', eye_r_img)
        # cv2.imshow('eye_right', eye_r_img)
        # cv2.waitKey(0)
        # print(eye_l_occ, eye_r_occ, np.subtract(eye_l_occ, eye_r_occ))
        if (eye_l_occ > 0.2 and eye_r_occ > 0.2) and abs(
                np.subtract(eye_l_occ, eye_r_occ)) < 0.2:
            draw_landmak_point(image, points)
            cv2.imshow('img', image)
            cv2.waitKey(0)

            face_bgr = cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR)
            landmark_face = crop_landmark_face(points_for_crop, face_bgr)
            skin = im.detect(face_bgr)

            cv2.imshow('My Image', np.hstack([landmark_face, skin]))
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            pose.detect_head_pose_with_6_points(points)
            cv2.imshow('output', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        else:
            draw_landmak_point(image, points)
            cv2.imshow('landmark', image)
            cv2.waitKey(0)

            cv2.imwrite('william_eye_left.jpg', eye_l_img)
            cv2.imshow('eye_left', eye_l_img)
            cv2.waitKey(0)

            cv2.imwrite('william_eye_right.jpg', eye_r_img)
            cv2.imshow('eye_right', eye_r_img)
            cv2.waitKey(0)
            continue
        # #######################################################################


if __name__ == '__main__':
    main()