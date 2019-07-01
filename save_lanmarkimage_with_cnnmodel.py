import os
import glob
import cv2
import numpy as np
import face_recognition as fr
from keras.models import load_model
from utils import *
from uuid import uuid4
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--folder_path', help='choose a image folder')

args = parser.parse_args()

path = os.path.abspath(os.path.dirname(__file__))
model_name = os.path.join(path, 'models/cnn_0628.h5')
data_path = args.folder_path


def main():

    if 'face_images' in data_path:

        for image in glob.glob(os.path.join(data_path, '*.jpg')):
            img = cv2.imread(image, cv2.IMREAD_UNCHANGED)
            face_image = cv2.resize(img, (200, 200))
            face_img = np.reshape(face_image, (1, 200, 200, 3))
            face_img = face_img.astype('float32') / 255

            model = load_model(model_name)
            points = model.predict(face_img)
            points = np.reshape(points, (-1, 2)) * 200

            for point in points:
                cv2.circle(face_image, (int(point[0]), int(point[1])), 2,
                           (0, 255, 0), -1, cv2.LINE_AA)

                cv2.imwrite('./test/landmark_image/' + image.split('/')[-1],
                            face_image)
    else:

        for image in glob.glob(os.path.join(data_path, '*.jpg')):
            img = cv2.imread(image, cv2.IMREAD_UNCHANGED)

            if img is None:
                continue

            try:
                locations = fr.face_locations(img)
            except:
                continue

            if len(locations) == 0:
                continue

            locs = []
            for loc in locations:
                start_x, start_y, end_x, end_y = loc[3], loc[0], loc[1], loc[2]
                locs.append((start_x, start_y, end_x, end_y))

            # crop from image
            cut_face_imgs = cut_face(img, locs)

            for face_img in cut_face_imgs:
                image_name = image.split('/')[-1] + str(uuid4())[:3]

                if face_img.size == 0:
                    continue

                face_image = cv2.resize(face_img, (200, 200))
                face_img = np.reshape(face_image, (1, 200, 200, 3))
                face_img = face_img.astype('float32') / 255

                model = load_model(model_name)
                points = model.predict(face_img)
                points = np.reshape(points, (-1, 2)) * 200

                for point in points:
                    cv2.circle(face_image, (int(point[0]), int(point[1])), 2,
                               (0, 255, 0), -1, cv2.LINE_AA)

                    cv2.imwrite(
                        './test/landmark_image/' + image.split('/')[-1],
                        face_image)


if __name__ == "__main__":
    main()