import numpy as np
import cv2
import math
from stabilizer import Stabilizer


class PoseDetector:
    def __init__(self, face_image):
        self.face_image = face_image
        self.width = self.face_image.shape[1]
        self.height = self.face_image.shape[0]
        self.focal_length = self.width
        self.center = (self.width / 2, self.height / 2)
        self.camera_matrix = np.array(
            [[self.focal_length, 0, self.center[0]],
             [0, self.focal_length, self.center[1]], [0, 0, 1]],
            dtype="double")

        self.dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion

        # Rotation vector and translation vector
        self.r_vec = np.array([[0.01891013], [0.08560084], [-3.14392813]])
        self.t_vec = np.array([[-14.97821226], [-10.62040383],
                               [-2053.03596872]])

        # 3D model points
        self.model_points = np.array([
            (0.0, 0.0, 0.0),  # Nose tip
            (0.0, -330.0, -65.0),  # Chin
            (-225.0, 170.0, -135.0),  # Left eye left corner
            (225.0, 170.0, -135.0),  # Right eye right corne
            (-150.0, -150.0, -125.0),  # Left Mouth corner
            (150.0, -150.0, -125.0)  # Right mouth corner
        ])

        # self.model_points = self._get_full_model_points()

    def _get_full_model_points(self, filename='assets/model.txt'):
        """ get full 3D model points from file """
        raw_value = []
        with open(filename) as f:
            f = f.readlines()
            for i in range(len(f) - 1):
                raw_value.append(f[i].strip().strip())
        model_points = np.array(raw_value, dtype=np.float32)
        model_points = np.reshape(model_points, (3, -1)).T
        model_points[:, -1] *= -1

        return model_points

    def detect_head_pose_with_6_points(self, points):
        # 2D image points
        image_points = np.array([
            (points[30][0], points[30][1]),  # nose tip
            (points[8][0], points[8][1]),  # chin
            (points[36][0], points[36][1]),  # left eye
            (points[45][0], points[45][1]),  # right eye
            (points[48][0], points[48][1]),  # left mouth
            (points[54][0], points[54][1]),  # right mouth
        ])

        (success, rotation_vector, translation_vector) = cv2.solvePnP(
            self.model_points,
            image_points,
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE)

        print("Rotation Vector:\n {0}".format(rotation_vector))
        print("Translation Vector:\n {0}".format(translation_vector))

        # Project a 3D point (0, 0, 1000.0) onto the image plane.
        (nose_end_point2D, jacobian1) = cv2.projectPoints(
            np.float32([[500, 0, 0], [0, 500, 0],
                        [0, 0, 500]]), rotation_vector, translation_vector,
            self.camera_matrix, self.dist_coeffs)

        (modelpts, jacobian2) = cv2.projectPoints(
            self.model_points, rotation_vector, translation_vector,
            self.camera_matrix, self.dist_coeffs)
        rvec_matrix = cv2.Rodrigues(rotation_vector)[0]

        proj_matrix = np.hstack((rvec_matrix, translation_vector))

        eulerAngles = cv2.decomposeProjectionMatrix(proj_matrix)[6]

        pitch, yaw, roll = [math.radians(_) for _ in eulerAngles]

        pitch = math.degrees(math.asin(math.sin(pitch)))
        roll = -math.degrees(math.asin(math.sin(roll)))
        yaw = math.degrees(math.asin(math.sin(yaw)))

        print(pitch, roll, yaw)

        for p in image_points:
            cv2.circle(self.face_image, (int(p[0]), int(p[1])), 2, (0, 0, 255),
                       -1)

        p1 = (int(image_points[0][0]), int(image_points[0][1]))

        cv2.line(self.face_image, p1, tuple(nose_end_point2D[1].ravel()),
                 (0, 255, 0), 2)  #Green
        cv2.line(self.face_image, p1, tuple(nose_end_point2D[0].ravel()),
                 (255, 0, 0), 2)  #Blue
        cv2.line(self.face_image, p1, tuple(nose_end_point2D[2].ravel()),
                 (0, 0, 255), 2)  #RED

        cv2.putText(
            self.face_image, ('pitch: {:05.2f}').format(float(str(pitch))),
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1, (0, 255, 0),
            thickness=2,
            lineType=2)

        cv2.putText(
            self.face_image, ('roll: {:05.2f}').format(float(str(roll))),
            (10, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1, (255, 0, 0),
            thickness=2,
            lineType=2)

        cv2.putText(
            self.face_image, ('yaw: {:05.2f}').format(float(str(yaw))),
            (10, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            1, (0, 0, 255),
            thickness=2,
            lineType=2)

    def detect_head_pose_with_68_points(self, points):

        model_points = self._get_full_model_points()
        image_points = points[:68]

        if self.r_vec is None:
            (_, rotation_vector, translation_vector) = cv2.solvePnP(
                model_points,
                image_points,
                self.camera_matrix,
                self.dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE)

            self.r_vec = rotation_vector
            self.t_vec = translation_vector

        (_, rotation_vector, translation_vector) = cv2.solvePnP(
            model_points,
            image_points,
            self.camera_matrix,
            self.dist_coeffs,
            rvec=self.r_vec,
            tvec=self.t_vec,
            useExtrinsicGuess=True)

        # Introduce scalar stabilizers for pose.
        pose_stabilizers = [
            Stabilizer(
                state_num=2, measure_num=1, cov_process=0.1, cov_measure=0.1)
            for _ in range(6)
        ]

        # Stabilize the pose.
        stabile_pose = []
        pose_np = np.array((rotation_vector, translation_vector)).flatten()
        for value, ps_stb in zip(pose_np, pose_stabilizers):
            ps_stb.update([value])
            stabile_pose.append(ps_stb.state[0])
        stabile_pose = np.reshape(stabile_pose, (-1, 3))

        print("Rotation Vector:\n {0}".format(stabile_pose[0]))
        print("Translation Vector:\n {0}".format(stabile_pose[1]))

        point_3d = []
        rear_size = 30
        rear_depth = 0
        point_3d.append((-rear_size, -rear_size, rear_depth))
        point_3d.append((-rear_size, rear_size, rear_depth))
        point_3d.append((rear_size, rear_size, rear_depth))
        point_3d.append((rear_size, -rear_size, rear_depth))
        point_3d.append((-rear_size, -rear_size, rear_depth))

        front_size = 50
        front_depth = 50
        point_3d.append((-front_size, -front_size, front_depth))
        point_3d.append((-front_size, front_size, front_depth))
        point_3d.append((front_size, front_size, front_depth))
        point_3d.append((front_size, -front_size, front_depth))
        point_3d.append((-front_size, -front_size, front_depth))
        point_3d = np.array(point_3d, dtype=np.float).reshape(-1, 3)

        # Map to 2d image points
        (point_2d,
         _) = cv2.projectPoints(point_3d, stabile_pose[0], stabile_pose[1],
                                self.camera_matrix, self.dist_coeffs)

        rvec_matrix = cv2.Rodrigues(stabile_pose[0])[0]

        translation_vector = np.expand_dims(stabile_pose[1], axis=1)
        proj_matrix = np.hstack((rvec_matrix, translation_vector))

        eulerAngles = cv2.decomposeProjectionMatrix(proj_matrix)[6]

        pitch, yaw, roll = [math.radians(_) for _ in eulerAngles]

        pitch = math.degrees(math.asin(math.sin(pitch)))
        roll = -math.degrees(math.asin(math.sin(roll)))
        yaw = math.degrees(math.asin(math.sin(yaw)))

        print(pitch, roll, yaw)
        point_2d = np.int32(point_2d.reshape(-1, 2))

        # Draw all the lines
        cv2.polylines(self.face_image, [point_2d], True, (255, 255, 255), 2,
                      cv2.LINE_AA)
        cv2.line(self.face_image, tuple(point_2d[1]), tuple(point_2d[6]),
                 (255, 255, 255), 2, cv2.LINE_AA)
        cv2.line(self.face_image, tuple(point_2d[2]), tuple(point_2d[7]),
                 (255, 255, 255), 2, cv2.LINE_AA)
        cv2.line(self.face_image, tuple(point_2d[3]), tuple(point_2d[8]),
                 (255, 255, 255), 2, cv2.LINE_AA)
