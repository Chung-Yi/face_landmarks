import cv2
import numpy as np
import face_recognition as fr
from utils import *
from keras.models import load_model
from point_detector import Face_landmark
from pose_utils import Annotator

INPUT_SIZE = 200

landmarks_3d_list = [
    np.array(
        [
            [0.000, 0.000, 0.000],  # Nose tip
            [0.000, -8.250, -1.625],  # Chin
            [-5.625, 4.250, -3.375],  # Left eye left corner
            [5.625, 4.250, -3.375],  # Right eye right corner
            [-3.750, -3.750, -3.125],  # Left Mouth corner
            [3.750, -3.750, -3.125]  # Right mouth corner 
        ],
        dtype=np.double),
    np.array(
        [
            [0.000000, 0.000000, 6.763430],  # 52 nose bottom edge
            [6.825897, 6.760612, 4.402142],  # 33 left brow left corner
            [1.330353, 7.122144, 6.903745],  # 29 left brow right corner
            [-1.330353, 7.122144, 6.903745],  # 34 right brow left corner
            [-6.825897, 6.760612, 4.402142],  # 38 right brow right corner
            [5.311432, 5.485328, 3.987654],  # 13 left eye left corner
            [1.789930, 5.393625, 4.413414],  # 17 left eye right corner
            [-1.789930, 5.393625, 4.413414],  # 25 right eye left corner
            [-5.311432, 5.485328, 3.987654],  # 21 right eye right corner
            [2.005628, 1.409845, 6.165652],  # 55 nose left corner
            [-2.005628, 1.409845, 6.165652],  # 49 nose right corner
            [2.774015, -2.080775, 5.048531],  # 43 mouth left corner
            [-2.774015, -2.080775, 5.048531],  # 39 mouth right corner
            [0.000000, -3.116408, 6.097667],  # 45 mouth central bottom corner
            [0.000000, -7.415691, 4.070434]  # 6 chin corner
        ],
        dtype=np.double),
    np.array(
        [
            [0.000000, 0.000000, 6.763430],  # 52 nose bottom edge
            [5.311432, 5.485328, 3.987654],  # 13 left eye left corner
            [1.789930, 5.393625, 4.413414],  # 17 left eye right corner
            [-1.789930, 5.393625, 4.413414],  # 25 right eye left corner
            [-5.311432, 5.485328, 3.987654]  # 21 right eye right corner
        ],
        dtype=np.double)
]

lm_2d_index_list = [
    [30, 8, 36, 45, 48, 54],
    [33, 17, 21, 22, 26, 36, 39, 42, 45, 31, 35, 48, 54, 57, 8],  # 14 points
    [33, 36, 39, 42, 45]  # 5 points
]


class HeadPoseDetection:
    def __init__(self):
        self.landmarks_3d = landmarks_3d_list[1]
        self.landmarks_2d = lm_2d_index_list[1]
        self.fl = Face_landmark()

    def process_image(self, image):
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        locs = []
        locations = fr.face_locations(image)
        for loc in locations:
            start_x, start_y, end_x, end_y = loc[3], loc[0], loc[1], loc[2]
            locs.append((start_x, start_y, end_x, end_y))

        if len(locations) == 0:
            return image, None

        cut_face_imgs = cut_face(img, locs)

        for i, face_img in enumerate(cut_face_imgs):
            face_resized = cv2.resize(face_img, (INPUT_SIZE, INPUT_SIZE))
            face_reshape = np.reshape(face_resized, (1, 200, 200, 3))
            face_normalize = face_reshape.astype('float32') / 255
            points = self.fl.face_landmark(face_normalize)

            if len(points) == 0:
                return image, None
            for point in points:
                point[0] *= face_img.shape[1]
                point[1] *= face_img.shape[0]
                point[0] += locs[i][0] - 50
                point[1] += locs[i][1] - 50

            landmarks_2d = self.get_landmarks(points)
            rvec, tvec, cm, dc = self.get_headpose(image, landmarks_2d)
            angles = self.get_angles(rvec, tvec)

            annotator = Annotator(image, angles, landmarks_2d, rvec, tvec, cm,
                                  dc)
            im = annotator.draw_direction()

            return im, angles

    def get_landmarks(self, points):
        coords = []
        for i in self.landmarks_2d:
            coords.append([points[i][0], points[i][1]])
        points = np.array(coords)
        return points

    def get_headpose(self, image, landmarks_2d):
        height, width = image[:2]
        focal_length = width
        center = (width / 2, height / 2)
        camera_matrix = np.array([[focal_length, 0, center[0]],
                                  [0, focal_length, center[1]], [0, 0, 1]])

        dist_coeffs = np.zeros((4, 1))

        (success, rotation_vector, translation_vector) = cv2.solvePnP(
            self.landmarks_3d,
            landmarks_2d,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE)

        return rotation_vector, translation_vector, camera_matrix, dist_coeffs

    def get_angles(self, rvec, tvec):
        rmat = cv2.Rodrigues(rvec)[0]
        P = np.hstack((rmat, tvec))  # projection matrix [R | t]
        degrees = -cv2.decomposeProjectionMatrix(P)[6]
        rx, ry, rz = degrees[:, 0]
        return [rx, ry, rz]
