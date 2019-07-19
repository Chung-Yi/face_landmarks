import cv2
import os
import sys
import glob
import math
import imutils
import time
import pafy
import numpy as np
import face_recognition as fr
from detector.pose_detector import PoseDetector
from detector.point_detector import Point
from utils import *
from keras.models import load_model
from argparse import ArgumentParser
from collections import OrderedDict
from process_image import Image
from video_resource import ThreadingVideoResource
from uuid import uuid4

parser = ArgumentParser()
parser.add_argument(
    '--did',
    default='0',
    help="assign device ID, example: 0AA1EA9A5A04B78D4581DD6D17742627.")
parser.add_argument('--model_name', default='cnn', help='choose a model')
args = parser.parse_args()

model_name = args.model_name

path = os.path.abspath(os.path.dirname(__file__))
model1_name = os.path.join(path, 'models/cnn_0702.h5')
model2_name = os.path.join(path,
                           'models/shape_predictor_81_face_landmarks.dat')

INPUT_SIZE = 200


def get_face_info(frame, pt):
    locations = fr.face_locations(frame)
    locs = []
    pts = []
    pts_resized = []
    eyes_info = []
    nose_info = []
    face_resized_info = []
    for loc in locations:
        start_x, start_y, end_x, end_y = loc[3], loc[0], loc[1], loc[2]
        locs.append((start_x, start_y, end_x, end_y))

    cut_face_imgs, delta_locs = cut_face(frame, locs)

    for i, (face_img, delta_locs) in enumerate(zip(cut_face_imgs, delta_locs)):
        if face_img.size != 0:
            face_resized = cv2.resize(face_img, (INPUT_SIZE, INPUT_SIZE))
            face_reshape = np.reshape(face_resized,
                                      (1, INPUT_SIZE, INPUT_SIZE, 3))
            face_normalize = face_reshape.astype('float32') / 255
            points = pt.face_landmark(face_normalize, face_resized, model_name)
            # points_for_crop = points.copy()

            if len(locs[i]) == 0:
                points = np.array([])

            # points_for_crop[:, 0] *= face_img.shape[1]
            # points_for_crop[:, 1] *= face_img.shape[0]
            points_resized = points.copy()
            points_resized[:, 0] *= INPUT_SIZE
            points_resized[:, 1] *= INPUT_SIZE

            points[:, 0] *= face_img.shape[1]
            points[:, 1] *= face_img.shape[0]
            points[:, 0] += locs[i][0] - delta_locs[0]
            points[:, 1] += locs[i][1] - delta_locs[1]

            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            eyes_occ, eyes_img = eyes_images(frame_bgr, points)
            nose_hole_occ, gx = nose_image(frame_bgr, points)
            if (eyes_occ is not None) and (eyes_img is not None):
                eye_l_occ = eyes_occ[0]
                eye_r_occ = eyes_occ[1]
                eye_l_img = eyes_img[0]
                eye_r_img = eyes_img[1]

                pts.append(points)
                eyes_info.append([(eye_l_occ, eye_r_occ), (eye_l_img,
                                                           eye_r_img)])
                nose_info.append(nose_hole_occ)
                pts_resized.append(points_resized)
                face_resized_info.append(face_resized)
        else:
            return None

    return zip(*[pts, eyes_info, nose_info, face_resized_info, pts_resized])


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


def cheek_dist(image, points, nose_info):
    left_cheek_x = points[2][0]
    left_cheek_y = points[2][1]
    right_cheek_x = points[14][0]
    right_cheek_y = points[14][1]
    nose_tip_x = points[30][0]
    nose_tip_y = points[30][1]
    left_cheek_x = points[2][0]
    left_cheek_y = points[2][1]

    left_dist = math.sqrt((left_cheek_x - nose_tip_x)**2 +
                          (left_cheek_y - nose_tip_y)**2)
    right_dist = math.sqrt((right_cheek_x - nose_tip_x)**2 +
                           (right_cheek_y - nose_tip_y)**2)

    if abs(left_dist - right_dist) < 26:
        direction = 'm'
    elif left_dist > right_dist:
        direction = 'r'
    elif right_dist > left_dist:
        direction = 'l'

    cv2.putText(
        image, ('left_dist: {:05.2f}').format(float(str(left_dist))),
        (10, 160),
        cv2.FONT_HERSHEY_SIMPLEX,
        1, (0, 0, 255),
        thickness=2,
        lineType=2)

    cv2.putText(
        image, ('right_dist: {:05.2f}').format(float(str(right_dist))),
        (10, 200),
        cv2.FONT_HERSHEY_SIMPLEX,
        1, (255, 0, 0),
        thickness=2,
        lineType=2)

    cv2.putText(
        image, ('abs(left_dist - right_dist): {:05.2f}').format(
            float(str(abs(left_dist - right_dist)))), (10, 240),
        cv2.FONT_HERSHEY_SIMPLEX,
        1, (0, 255, 0),
        thickness=2,
        lineType=2)

    cv2.putText(
        image, ('nose_occ: {:05.2f}').format(float(str(nose_info))), (10, 280),
        cv2.FONT_HERSHEY_SIMPLEX,
        1, (0, 255, 0),
        thickness=2,
        lineType=2)

    return direction


def nose_image(face, points):
    nose_left_corner_x = points[31][0]
    nose_left_corner_y = points[31][1]
    nose_right_corner_x = points[35][0]
    nose_right_corner_y = points[35][1]
    nose_tip_x = points[30][0]
    nose_tip_y = points[30][1]
    nose_min_y = points[27][1]

    p1 = points[30]
    p2 = points[33]
    p3 = points[35]
    p4 = points[31]

    if p1[0] > p2[0]:
        nose = face[int(p1[1]):int((p3[1] + p4[1]) / 2),
                    int(p2[0]) - 2:int(p1[0]) + 2]
        # nose = face[int(p1[1]):int(p2[1]), int(p2[0]) - 1:int(p1[0]) + 1]
    else:
        nose = face[int(p1[1]):int((p3[1] + p4[1]) / 2),
                    int(p1[0]) - 2:int(p2[0]) + 2]
        # nose = face[int(p1[1]):int(p2[1]), int(p1[0]) - 1:int(p2[0]) + 1]

    nose = cv2.cvtColor(nose, cv2.COLOR_BGR2GRAY)
    gx, gy = np.gradient(nose)
    b = np.count_nonzero(gx < 10)
    nose_hole_occ = b / (gx.shape[0] * gx.shape[1])

    # cv2.imwrite('nose_gray/jason_nose_tip.jpg', nose)
    # cv2.imwrite('nose_gray/jason_nose_gx.jpg', gx)
    # cv2.imwrite('nose_gray/jason_nose_gy.jpg', gy)

    return nose_hole_occ, gx


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

    if (len(eye_l_img) == 0) or (len(eye_r_img) == 0):
        return None, None

    eye_l_area = eye_l_img.shape[0] * eye_l_img.shape[1]
    eye_l_gray = cv2.cvtColor(eye_l_img, cv2.COLOR_BGR2GRAY)
    eye_l_occ = np.count_nonzero(eye_l_gray < 50) / eye_l_area

    eye_r_area = eye_r_img.shape[0] * eye_r_img.shape[1]
    eye_r_gray = cv2.cvtColor(eye_r_img, cv2.COLOR_BGR2GRAY)
    eye_r_occ = np.count_nonzero(eye_r_gray < 50) / eye_r_area

    return (eye_l_occ, eye_r_occ), (eye_l_img, eye_r_img)


def main():
    sleep_time = 1
    time.sleep(sleep_time)
    print("Init sleep for %d seconds" % sleep_time)

    pt = Point(model1_name, model2_name)
    im = Image()

    cam_width, cam_height = 1280, 720

    if args.did.isdigit():
        virtual_device = int(args.did)
    elif 'https' in args.did:
        url = args.did
        vpafy = pafy.new(url)
        play = vpafy.getbest(preftype="mp4")
        virtual_device = play.url
    else:
        virtual_device = args.did

    resource = ThreadingVideoResource(virtual_device, cam_width, cam_height)

    crop_start = int((cam_width - cam_height) / 2 + 1)
    i = 0
    while True:
        try:
            frame, cap_time = resource.get_frame_date()
            frame = frame[:, crop_start:crop_start + cam_height, :]
            frame = cv2.flip(frame, 1)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            face = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        except Exception as err:
            resource.release()
            raise err

        pose = PoseDetector(frame_bgr)

        face_info = get_face_info(frame, pt)
        if face_info is not None:
            for pts, eye_info, nose_info, face_resized, points_resized in face_info:

                print(eye_info[0][0], eye_info[0][1],
                      abs(np.subtract(eye_info[0][0], eye_info[0][1])))

                if (eye_info[0][0] > 0.18 and eye_info[0][1] > 0.18) and abs(
                        np.subtract(eye_info[0][0],
                                    eye_info[0][1])) < 0.3 and nose_info < 0.3:
                    yaw_angle = pose.detect_head_pose_with_6_points(pts)
                    direction_ = pose.cheek_dist(pts, nose_info)
                    # nose_m, delta_x = pose.nose_slop(pts)

                    direction = pose.triangle_area(face_resized,
                                                   points_resized)
                    draw_landmak_point(frame_bgr, pts)

                    if direction == 'm' and direction_ == 'm':
                        cv2.imwrite('head/m/m_{}.jpg'.format(str(uuid4())[:8]),
                                    frame_bgr)
                        # cv2.cvtColor(frame_bgr, cv2.COLOR_RGB2BGR))
                    elif direction == 'l' and direction_ == 'l':
                        cv2.imwrite('head/l/l_{}.jpg'.format(str(uuid4())[:8]),
                                    frame_bgr)
                        # cv2.cvtColor(frame_bgr, cv2.COLOR_RGB2BGR))
                    elif direction == 'r' and direction_ == 'r':
                        cv2.imwrite('head/r/r_{}.jpg'.format(str(uuid4())[:8]),
                                    frame_bgr)
                        # cv2.cvtColor(frame_bgr, cv2.COLOR_RGB2BGR))
                    else:
                        cv2.imwrite('head/n/n_{}.jpg'.format(str(uuid4())[:8]),
                                    frame_bgr)

        #             # if abs(yaw_angle) < 20.0:
        #             #     cv2.imwrite(
        #             #         'head_pose/m/m_{}.jpg'.format(str(uuid4())[:8]),
        #             #         cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        #             # elif yaw_angle > 20.0:
        #             #     cv2.imwrite(
        #             #         'head_pose/l/l_{}.jpg'.format(str(uuid4())[:8]),
        #             #         cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        #             # elif yaw_angle < -20.0:
        #             #     cv2.imwrite(
        #             #         'head_pose/r/r_{}.jpg'.format(str(uuid4())[:8]),
        #             #         cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        #         else:
        #             draw_landmak_point(frame_bgr, pts)
        #             # cv2.imwrite(
        #             #     'head_pose/wrong/w_{}.jpg'.format(str(uuid4())[:8]),
        #             #     frame_bgr)

        cv2.imshow('output', frame_bgr)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()