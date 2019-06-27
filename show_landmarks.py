import cv2
import os
import timeit
import face_recognition as fr
from utils import *
from process_data import cut_face
from keras.models import load_model

model_name = 'models/shape_predictor_81_face_landmarks.dat'


def main():
    # pts = os.path.join(path, 'l.pts')
    model = load_model('models/cnn_0625_asian.h5')
    path = os.path.abspath(os.path.dirname(__file__))
    image = os.path.join(path, 'liann.jpg')
    image = cv2.imread(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    locations = fr.face_locations(image)
    locs = []

    for loc in locations:
        start_x, start_y, end_x, end_y = loc[3], loc[0], loc[1], loc[2]
        locs.append((start_x, start_y, end_x, end_y))

    cut_face_imgs = cut_face(image, locs)

    for face_img in cut_face_imgs:

        if face_img.size == 0:
            continue
        # filter blurry face
        # if variance_of_laplacian(face_img) < MIN_BLUR:
        #     continue

        # if shape != None:
        #     assert isinstance(shape, int)
        #     face_img = cv2.resize(face_img, (shape, shape))
        # points = get_81_points(face_img, model_name)
        # try:
        #     assert len(points) == PTS
        # except:
        #     print("face landmarks is not 81")
        #     continue
        # if points_are_valid(points, face_img) is False:
        #     counter['invalid_face'] += 1
        #     print('points are out of image')
        #     continue
        f = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_image = cv2.resize(face_img, (200, 200))
        face_img = np.reshape(face_image, (1, 200, 200, 3))
        face_img = face_img.astype('float32') / 255
        start = timeit.default_timer()
        points = model.predict(face_img)
        end = timeit.default_timer()
        print(end - start)
        points = np.reshape(points, (-1, 2)) * 200
        # points = scaled_points(image, points, locs)
        draw_landmak_point(face_image, points)

        # if face_img.size == 0:
        #     continue
        # # filter blurry face
        # if variance_of_laplacian(face_img) < MIN_BLUR:
        #     continue

        # if shape != None:
        #     assert isinstance(shape, int)
        #     face_img = cv2.resize(face_img, (shape, shape))
        # points = get_81_points(face_img, model_name)
        # try:
        #     assert len(points) == PTS
        # except:
        #     print("face landmarks is not 81")
        #     continue
        # if points_are_valid(points, face_img) is False:
        #     counter['invalid_face'] += 1
        #     print('points are out of image')
        #     continue

        # for point in points:
        #     cv2.circle(face_img, (int(point[0]), int(point[1])), 2,
        #                (0, 255, 0), -1, cv2.LINE_AA)

        face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imshow('My Image', face_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # face_img = cv2.resize(face_img, (200, 200))
    # face_img = np.reshape(face_img, (1, 200, 200, 3))
    # face_img = face_img.astype('float32') / 255

    # pts = os.path.join(path, 'l.pts')
    # model_name = os.path.join(path,
    #                           'models/shape_predictor_81_face_landmarks.dat')
    # 68 points
    # points = get_68_points(pts)
    # draw_landmak_point(image, points)

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