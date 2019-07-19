import cv2
import os
import numpy as np
import face_recognition as fr
from keras.models import load_model
from utils import cut_face
from detector.pose_detector import PoseDetector
from detector.point_detector import Point
from utils import draw_landmak_point

path = os.path.abspath(os.path.dirname(__file__))
model1_name = os.path.join(path, 'models/cnn_0702.h5')
model2_name = os.path.join(path,
                           'models/shape_predictor_81_face_landmarks.dat')

INPUT_SIZE = 200

# def get_nose_area(image, points):

#     p27 = points[27]
#     p31 = points[31]
#     p35 = points[35]
#     p_max_y = max(points[31:36][1])
#     # nose_img = image[int(p27[1]) - 2:int(p_max_y) + 2,
#     #                  int(p31[0]) - 2:int(p35[0]) + 2]

#     # points[:, 0] -= int(p31[0]) - 2
#     # points[:, 1] -= int(p27[1]) - 2
#     # cv2.imwrite('landmark_image/nose/rivon.jpg', nose_img[:, :, ::-1])
#     # draw_landmak_point(nose_img, points)
#     # cv2.imshow('nose', nose_img)
#     # cv2.waitKey(0)

#     # draw_landmak_point(image, points)
#     # cv2.imshow('img', image[:, :, ::-1])
#     # cv2.waitKey(0)
#     # cv2.imwrite('landmark_image/resized_200/rivon.jpg', image[:, :, ::-1])


def triangle_area(image, points):

    p27 = points[27]
    p30 = points[30]
    p31 = points[31]
    p35 = points[35]

    p30_p27 = p27 - p30
    p30_p31 = p31 - p30
    p30_p35 = p35 - p30

    left_nose_area = 0.5 * abs(p30_p27[0] * p30_p31[1] -
                               p30_p27[1] * p30_p31[0])
    right_nose_area = 0.5 * abs(p30_p27[0] * p30_p35[1] -
                                p30_p27[1] * p30_p35[0])

    print('left_nose_area', left_nose_area)
    print('right_nose_area', right_nose_area)


def get_nose_angle(points):
    p27 = points[27]
    p30 = points[30]
    p31 = points[31]
    p35 = points[35]
    p33 = points[33]

    p30_p27 = p27 - p30
    p33_p31 = p31 - p33
    p33_p35 = p35 - p33

    cosine_left_angle = np.dot(
        p30_p27, p33_p31) / (np.linalg.norm(p30_p27) * np.linalg.norm(p33_p31))
    cosine_right_angle = np.dot(
        p30_p27, p33_p35) / (np.linalg.norm(p30_p27) * np.linalg.norm(p33_p35))

    left_angle = np.arccos(cosine_left_angle)
    right_angle = np.arccos(cosine_right_angle)

    left_angle = np.degrees(left_angle)
    right_angle = np.degrees(right_angle)
    print('left_angle', left_angle)
    print('right_angle', right_angle)

    # if left_angle > right_angle:
    #     print('r')
    # else:
    #     print('l')


def nose_slop(points):
    pt_top_x = points[27][0]
    pt_top_y = points[27][1]

    pt_bottom_x = points[30][0]
    pt_bottom_y = points[30][1]

    print('delta_y', pt_top_y - pt_bottom_y)
    print('delta_x', pt_top_x - pt_bottom_x)

    nose_m = -1 * ((pt_top_y - pt_bottom_y) / (pt_top_x - pt_bottom_x))

    return nose_m


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
                    int(p2[0]) - 1:int(p1[0]) + 1]
        # nose = face[int(p1[1]):int(p2[1]), int(p2[0]) - 1:int(p1[0]) + 1]
    else:
        nose = face[int(p1[1]):int((p3[1] + p4[1]) / 2),
                    int(p1[0]) - 1:int(p2[0]) + 1]
        # nose = face[int(p1[1]):int(p2[1]), int(p1[0]) - 1:int(p2[0]) + 1]

    # cv2.imwrite('m_nose_color.jpg', nose)
    nose = cv2.cvtColor(nose, cv2.COLOR_BGR2GRAY)
    # cv2.imwrite('m_nose_gray.jpg', nose)
    gx, gy = np.gradient(nose)

    # cv2.imwrite('m_nose_gy.jpg', gy)
    b = np.count_nonzero(gx < 10)
    nose_hole_occ = b / (gx.shape[0] * gx.shape[1])

    # cv2.imwrite('nose_gray/jason_nose_tip.jpg', nose)
    # cv2.imwrite('nose_gray/jason_nose_gx.jpg', gx)
    # cv2.imwrite('nose_gray/jason_nose_gy.jpg', gy)

    return nose_hole_occ, gx


def main():
    pt = Point(model1_name, model2_name)

    image = cv2.imread('m.jpg')
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    pose = PoseDetector(image)

    locations = fr.face_locations(image_rgb)
    locs = []
    for loc in locations:
        start_x, start_y, end_x, end_y = loc[3], loc[0], loc[1], loc[2]
        locs.append((start_x, start_y, end_x, end_y))

    cut_face_imgs, delta_locs = cut_face(image_rgb, locs)

    for i, (face_img, delta_locs) in enumerate(zip(cut_face_imgs, delta_locs)):
        if face_img.size != 0:
            face_resized = cv2.resize(face_img, (INPUT_SIZE, INPUT_SIZE))
            face_reshape = np.reshape(face_resized,
                                      (1, INPUT_SIZE, INPUT_SIZE, 3))
            face_normalize = face_reshape.astype('float32') / 255
            points = pt.face_landmark(face_normalize, face_resized,
                                      model1_name)

            points_resized = points.copy()

            if len(locs[i]) == 0:
                points = np.array([])

            points_resized[:, 0] *= INPUT_SIZE
            points_resized[:, 1] *= INPUT_SIZE

            points[:, 0] *= face_img.shape[1]
            points[:, 1] *= face_img.shape[0]
            points[:, 0] += locs[i][0] - delta_locs[0]
            points[:, 1] += locs[i][1] - delta_locs[1]

            eyes_occ, eyes_img = eyes_images(image, points)
            nose_hole_occ, gx = nose_image(image, points)

            # cv2.imwrite('m_nose.jpg', gx)

            if (eyes_occ[0] > 0.18 and eyes_occ[1] > 0.18) and abs(
                    np.subtract(eyes_occ[0],
                                eyes_occ[1])) < 0.3 and nose_hole_occ < 0.3:

                # nose_m = nose_slop(points)
                triangle_area(face_resized, points_resized)
                # get_nose_area(face_resized, points_resized)


if __name__ == '__main__':
    main()