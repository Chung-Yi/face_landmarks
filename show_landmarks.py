import cv2
import os
from utils import *
from keras.models import load_model


def main():
    path = os.path.abspath(os.path.dirname(__file__))
    image = os.path.join(path, 'f.jpg')
    image = cv2.imread(image)
    face_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    face_img = cv2.resize(face_img, (200, 200))
    face_img = np.reshape(face_img, (1, 200, 200, 3))
    face_img = face_img.astype('float32') / 255

    pts = os.path.join(path, 'l.pts')
    model_name = os.path.join(path,
                              'models/shape_predictor_81_face_landmarks.dat')
    # 68 points
    # points = get_68_points(pts)
    # draw_landmak_point(image, points)

    # 81 points
    # points = get_81_points(image, model_name)
    # draw_landmak_point(image, points)

    # cnn model
    model = load_model('models/cnn.h5')
    points = model.predict(face_img)
    points = np.reshape(points, (-1, 2))
    points = scaled_points(image, points)
    draw_landmak_point(image, points)

    cv2.imshow('My Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()