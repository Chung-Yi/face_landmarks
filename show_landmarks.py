import cv2
import os
from utils import *


def main():
    path = os.path.abspath(os.path.dirname(__file__))
    image = os.path.join(path, 'test_image', 'i.jpg')
    image = cv2.imread(image)

    pts = os.path.join(path, 'l.pts')
    model_name = os.path.join(path,
                              'models/shape_predictor_81_face_landmarks.dat')
    # 68 points
    # points = get_68_points(pts)
    # draw_landmak_point(image, points)

    # 81 points
    points = get_81_points(image, model_name)
    draw_landmak_point(image, points)

    cv2.imshow('My Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()