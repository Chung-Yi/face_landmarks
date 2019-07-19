import cv2
import numpy as np


def main():
    image = cv2.imread('eye_images/jason_left_eyes.jpg')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    width = gray.shape[1]
    height = gray.shape[0]
    # gray = np.resize(gray, (4, 12))
    # cv2.imwrite('eye_images_gray/marco_right.jpg', gray)
    print(gray.shape)
    area = width * height
    # print(gray)
    eyes_ball_count = np.count_nonzero(gray < 40)
    print(eyes_ball_count)
    print('eye_occ: ', eyes_ball_count / area)


if __name__ == '__main__':
    main()