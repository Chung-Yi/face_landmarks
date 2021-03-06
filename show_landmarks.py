import cv2
import os
import sys
import glob
import math
import timeit
import imutils
import numpy as np
import face_recognition as fr
from utils import *
from keras.models import load_model
from argparse import ArgumentParser
from collections import OrderedDict
from skin_detect import detect

parser = ArgumentParser()
parser.add_argument('--model_name', default='cnn', help='choose a model')
args = parser.parse_args()

model_name = args.model_name

path = os.path.abspath(os.path.dirname(__file__))
model1_name = os.path.join(path, 'models/cnn_0628.h5')
model2_name = os.path.join(path,
                           'models/shape_predictor_81_face_landmarks.dat')

INPUT_SIZE = 200

gamma = 0.95

Wcb = 46.97
Wcr = 38.76

WHCb = 14
WHCr = 10
WLCb = 23
WLCr = 20

Ymin = 16
Ymax = 235

Kl = 125
Kh = 188

WCb = 0
WCr = 0

CbCenter = 0
CrCenter = 0


class Face:
    def __init__(self, model1_name, model2_name):
        self.model1 = load_model(model1_name)
        self.model2 = model2_name
        self.lower = np.array([0, 40, 80], dtype='uint8')
        self.upper = np.array([20, 255, 255], dtype='uint8')
        self.landmarks_2d = [
            33, 17, 21, 22, 26, 36, 39, 42, 45, 31, 35, 48, 54, 57, 8
        ]

    def face_landmark(self, face_img, face_image, model_name):

        if 'cnn' in model_name:
            start = timeit.default_timer()
            points = self.model1.predict(face_img)
            end = timeit.default_timer()
            print(end - start)
            points = np.reshape(points, (-1, 2))
        else:
            points = get_81_points(face_image, self.model2)
            points = np.array(points).astype('float32')

        return points

    def head_pose(self, points, face_image):
        # 2D image points
        image_points = np.array([
            (points[30][0], points[30][1]),  # nose tip
            (points[8][0], points[8][1]),  # chin
            (points[36][0], points[36][1]),  # left eye
            (points[45][0], points[45][1]),  # right eye
            (points[48][0], points[48][1]),  # left mouth
            (points[54][0], points[54][1]),  # right mouth
        ])

        # 3D model points
        model_points = np.array([
            (0.0, 0.0, 0.0),  # Nose tip
            (0.0, -330.0, -65.0),  # Chin
            (-225.0, 170.0, -135.0),  # Left eye left corner
            (225.0, 170.0, -135.0),  # Right eye right corne
            (-150.0, -150.0, -125.0),  # Left Mouth corner
            (150.0, -150.0, -125.0)  # Right mouth corner
        ])

        # camera internals
        width = face_image.shape[1]
        height = face_image.shape[0]
        focal_length = width
        center = (width / 2, height / 2)
        camera_matrix = np.array([[focal_length, 0, center[0]],
                                  [0, focal_length, center[1]], [0, 0, 1]],
                                 dtype="double")

        print("Camera Matrix :\n {0}".format(camera_matrix))

        dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
        (success, rotation_vector, translation_vector) = cv2.solvePnP(
            model_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE)

        print("Rotation Vector:\n {0}".format(rotation_vector))
        print("Translation Vector:\n {0}".format(translation_vector))

        # Project a 3D point (0, 0, 1000.0) onto the image plane.
        (nose_end_point2D, jacobian1) = cv2.projectPoints(
            np.float32([[500, 0, 0], [0, 500, 0], [0, 0, 500]]),
            rotation_vector, translation_vector, camera_matrix, dist_coeffs)

        modelpts, jacobian2 = cv2.projectPoints(model_points, rotation_vector,
                                                translation_vector,
                                                camera_matrix, dist_coeffs)
        rvec_matrix = cv2.Rodrigues(rotation_vector)[0]
        proj_matrix = np.hstack((rvec_matrix, translation_vector))
        eulerAngles = cv2.decomposeProjectionMatrix(proj_matrix)[6]

        pitch, yaw, roll = [math.radians(_) for _ in eulerAngles]

        pitch = math.degrees(math.asin(math.sin(pitch)))
        roll = -math.degrees(math.asin(math.sin(roll)))
        yaw = math.degrees(math.asin(math.sin(yaw)))
        print(pitch, roll, yaw)

        for p in image_points:
            cv2.circle(face_image, (int(p[0]), int(p[1])), 2, (0, 0, 255), -1)

        p1 = (int(image_points[0][0]), int(image_points[0][1]))
        p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

        cv2.line(face_image, p1, tuple(nose_end_point2D[1].ravel()),
                 (0, 255, 0), 2)  #GREEN
        cv2.line(face_image, p1, tuple(nose_end_point2D[0].ravel()),
                 (255, 0, 0), 2)
        cv2.line(face_image, p1, tuple(nose_end_point2D[2].ravel()),
                 (0, 0, 255), 2)

        cv2.putText(
            face_image, ('{:05.2f}').format(float(str(pitch))), (10, 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1, (0, 255, 0),
            thickness=2,
            lineType=2)

        cv2.putText(
            face_image, ('{:05.2f}').format(float(str(roll))), (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1, (255, 0, 0),
            thickness=2,
            lineType=2)

        cv2.putText(face_image, ('{:05.2f}').format(float(str(yaw))),
                    (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255))

        cv2.imshow('output', face_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def detect(self, face_image):
        converted = cv2.cvtColor(face_image, cv2.COLOR_BGR2HSV)
        skinmask = cv2.inRange(converted, self.lower, self.upper)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
        skinmask = cv2.erode(skinmask, kernel, iterations=2)
        skinmask = cv2.dilate(skinmask, kernel, iterations=2)

        skinmask = cv2.GaussianBlur(skinmask, (3, 3), 0)
        skin = cv2.bitwise_and(face_image, face_image, mask=skinmask)
        return skin


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


def skin_detect(face_image):
    rows = face_image.shape[0]
    cols = face_image.shape[1]

    for r in range(rows):
        for c in range(cols):
            # get values of blue, green, red
            B = face_image.item(r, c, 0)
            G = face_image.item(r, c, 1)
            R = face_image.item(r, c, 2)
            # gamma correction
            B = int(B**gamma)
            G = int(G**gamma)
            R = int(R**gamma)

            # set values of blue, green, red
            face_image.itemset((r, c, 0), B)
            face_image.itemset((r, c, 1), G)
            face_image.itemset((r, c, 2), R)

    # convert color space from rgb to ycbcr
    imgYcc = cv2.cvtColor(face_image, cv2.COLOR_BGR2YCR_CB)

    # convert color space from bgr to rgb
    img = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

    # prepare an empty image space
    imgSkin = np.zeros(img.shape, np.uint8)
    # copy original image
    imgSkin = img.copy()

    for r in range(rows):
        for c in range(cols):
            skin = 0

            # color space transformation

            # get values from ycbcr color space
            Y = imgYcc.item(r, c, 0)
            Cr = imgYcc.item(r, c, 1)
            Cb = imgYcc.item(r, c, 2)

            if Y < Kl:
                WCr = WLCr + (Y - Ymin) * (Wcr - WLCr) / (Kl - Ymin)
                WCb = WLCb + (Y - Ymin) * (Wcb - WLCb) / (Kl - Ymin)

                CrCenter = 154 - (Kl - Y) * (154 - 144) / (Kl - Ymin)
                CbCenter = 108 + (Kl - Y) * (118 - 108) / (Kl - Ymin)

            elif Y > Kh:
                WCr = WHCr + (Y - Ymax) * (Wcr - WHCr) / (Ymax - Kh)
                WCb = WHCb + (Y - Ymax) * (Wcb - WHCb) / (Ymax - Kh)

                CrCenter = 154 + (Y - Kh) * (154 - 132) / (Ymax - Kh)
                CbCenter = 108 + (Y - Kh) * (118 - 108) / (Ymax - Kh)

            if Y < Kl or Y > Kh:
                Cr = (Cr - CrCenter) * Wcr / WCr + 154
                Cb = (Cb - CbCenter) * Wcb / WCb + 108

            if Cb > 77 and Cb < 127 and Cr > 133 and Cr < 173:
                skin = 1

            if 0 == skin:
                imgSkin.itemset((r, c, 0), 0)
                imgSkin.itemset((r, c, 1), 0)
                imgSkin.itemset((r, c, 2), 0)
    return img, imgSkin


def main():

    image = os.path.join(path, 'b.jpg')
    image = cv2.imread(image)
    face = image.copy()
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    try:
        locations = fr.face_locations(image_rgb)
    except:
        sys.exit(0)

    locs = []

    for loc in locations:
        start_x, start_y, end_x, end_y = loc[3], loc[0], loc[1], loc[2]
        locs.append((start_x, start_y, end_x, end_y))

    cut_face_imgs = cut_face(image_rgb, locs)

    for i, face_img in enumerate(cut_face_imgs):

        if face_img.size == 0:
            continue

        f = Face(model1_name, model2_name)
        face_resized = cv2.resize(face_img, (INPUT_SIZE, INPUT_SIZE))
        face_reshape = np.reshape(face_resized, (1, INPUT_SIZE, INPUT_SIZE, 3))
        face_normalize = face_reshape.astype('float32') / 255
        points = f.face_landmark(face_normalize, face_resized, model_name)

        if len(locs[i]) == 0:
            points = []

        if start_x <= 50 or start_y <= 50:
            for point in points:
                point[0] *= face_img.shape[1]
                point[1] *= face_img.shape[0]
                point[0] += locs[i][0]
                point[1] += locs[i][1]
        else:
            for point in points:
                point[0] *= face_img.shape[1]
                point[1] *= face_img.shape[0]
                point[0] += locs[i][0] - 50
                point[1] += locs[i][1] - 50

        # draw_landmak_point(image, points)
        # cv2.imshow('img', image)
        # cv2.waitKey(0)

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
        eye_l_occ = np.count_nonzero(eye_l_gray < 50) / eye_l_area

        eye_r_area = eye_r_img.shape[0] * eye_r_img.shape[1]
        eye_r_gray = cv2.cvtColor(eye_r_img, cv2.COLOR_BGR2GRAY)
        eye_r_occ = np.count_nonzero(eye_r_gray < 50) / eye_r_area

        #######################################################################
        # detect eyes area to check if landmarks match on face
        if eye_l_occ > 0.25 and eye_r_occ > 0.25:
            draw_landmak_point(image, points)
            cv2.imshow('landmark', image)
            cv2.waitKey(0)

            # cv2.imwrite('william_eye_left.jpg', eye_l_img)
            cv2.imshow('eye_left', eye_l_img)
            cv2.waitKey(0)

            # cv2.imwrite('william_eye_right.jpg', eye_r_img)
            cv2.imshow('eye_right', eye_r_img)
            cv2.waitKey(0)

            landmark_face = crop_landmark_face(points, face)

            skin = f.detect(face)

            draw_landmak_point(face, points)
            cv2.imshow('My Image', np.hstack([landmark_face, skin]))
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            draw_landmak_point(image, points)
            cv2.imshow('landmark', image)
            cv2.waitKey(0)

            # cv2.imwrite('william_eye_left.jpg', eye_l_img)
            cv2.imshow('eye_left', eye_l_img)
            cv2.waitKey(0)

            # cv2.imwrite('william_eye_right.jpg', eye_r_img)
            cv2.imshow('eye_right', eye_r_img)
            cv2.waitKey(0)
            continue
        #######################################################################

        # draw_landmak_point(face, points)
        # cv2.imshow('landmark', face)
        # cv2.waitKey(0)

        # # cv2.imwrite('eye_left.jpg', eye_l_img)
        # cv2.imshow('eye_left', eye_l_img)
        # cv2.waitKey(0)

        # # cv2.imwrite('eye_right.jpg', eye_r_img)
        # cv2.imshow('eye_right', eye_r_img)
        # cv2.waitKey(0)

        # landmark_face = crop_landmark_face(points, face_image)

        # skin = f.detect(face_image)

        # draw_landmak_point(face_image, points)
        # cv2.imshow('My Image', np.hstack([landmark_face, skin]))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        f.head_pose(points, face)


if __name__ == '__main__':
    main()