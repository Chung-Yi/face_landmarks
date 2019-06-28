import cv2
import os
import timeit
import face_recognition as fr
from utils import *
from keras.models import load_model
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--model_name', default='cnn', help='choose a model')
args = parser.parse_args()

model_name = args.model_name

path = os.path.abspath(os.path.dirname(__file__))
model1_name = os.path.join(path, 'models/cnn_0625_200.h5')
model2_name = os.path.join(path,
                           'models/shape_predictor_81_face_landmarks.dat')


class Face:
    def __init__(self, model1_name, model2_name):
        self.model1 = load_model(model1_name)
        self.model2 = model2_name

    def face_landmark(self, face_img, face_image, model_name):

        if 'cnn' in model_name:
            points = self.model1.predict(face_img)
            points = np.reshape(points, (-1, 2)) * 200
        else:
            points = get_81_points(face_image, self.model2)
            points = np.array(points).astype('float32')

        return points

    def head_pose(self, points, face_image):
        # 2D image points
        image_points = np.array([
            (points[33][0], points[33][1]),  # nose tip
            (points[8][0], points[8][1]),  # chin
            (points[45][0], points[45][1]),  # left eye
            (points[36][0], points[36][1]),  # right eye
            (points[54][0], points[54][1]),  # left mouth
            (points[48][0], points[48][1]),  # right mouth
        ])

        # 3D model points.
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
                                  [0, focal_length, center[1]], [0, 0, 1]])

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
        (nose_end_point2D, jacobian) = cv2.projectPoints(
            np.array([(0.0, 0.0, 1000.0)]), rotation_vector,
            translation_vector, camera_matrix, dist_coeffs)
        for p in image_points:
            cv2.circle(face_image, (int(p[0]), int(p[1])), 2, (0, 0, 255), -1)

        p1 = (int(image_points[0][0]), int(image_points[0][1]))
        p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

        cv2.line(face_image, p1, p2, (255, 0, 0), 2)

        cv2.imshow('output', face_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    # pts = os.path.join(path, 'l.pts')
    image = os.path.join(path, 'm.jpg')
    image = cv2.imread(image)
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    locations = fr.face_locations(image)
    locs = []

    for loc in locations:
        start_x, start_y, end_x, end_y = loc[3], loc[0], loc[1], loc[2]
        locs.append((start_x, start_y, end_x, end_y))

    cut_face_imgs = cut_face(image, locs)

    for face_img in cut_face_imgs:

        if face_img.size == 0:
            continue

        # f = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        f = Face(model1_name, model2_name)
        face_image = cv2.resize(face_img, (200, 200))
        face_img = np.reshape(face_image, (1, 200, 200, 3))
        face_img = face_img.astype('float32') / 255

        start = timeit.default_timer()
        points = f.face_landmark(face_img, face_image, model_name)
        end = timeit.default_timer()
        print(end - start)
        # points = np.reshape(points, (-1, 2)) * 200
        # points = scaled_points(image, points, locs)
        draw_landmak_point(face_image, points)

        cv2.imshow('My Image', face_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        f.head_pose(points, face_image)

    #81 points
    start = timeit.default_timer()
    points = get_81_points(image, model_name)
    end = timeit.default_timer()
    print(end - start)
    draw_landmak_point(image, points)

    # cnn model
    # model = load_model('models/cnn_0619.h5')
    # points = model.predict(face_img)
    # points = np.reshape(points, (-1, 2))
    # points = scaled_points(image, points)
    # draw_landmak_point(image, points)

    cv2.imshow('My Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()