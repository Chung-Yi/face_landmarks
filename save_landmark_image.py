import glob
import os
import cv2
import face_recognition as fr
import multiprocessing as mp
from argparse import ArgumentParser
from uuid import uuid4
from utils import *
from log_manager import LogManager
from shutil import rmtree

parser = ArgumentParser()
parser.add_argument(
    '--folder_path', default="face_images", help='choose a image folder')
# parser.add_argument(
#     '--save_landmark', help='save landmark image', action='store_true')
# parser.add_argument('--save_face', help='save face image', action='store_true')
args = parser.parse_args()

IMG_SIZE = 200
MIN_BLUR = 50
SSD_THR = 0.6
SCORE_THR = 0.5
BIN_NUM = 1
PTS = 81
path = os.path.abspath(os.path.dirname(__file__))
data_path = args.folder_path
# save_face = args.save_face
# save_landmark = args.save_landmark
model_name = os.path.join(path, 'models/shape_predictor_81_face_landmarks.dat')


def save_face(face_img, image):
    print(image)
    cv2.imwrite(
        os.path.join(data_path + '_face',
                     image.split('/')[-1]), face_img)


def save_landmark(face_img, points, image):

    for point in points:
        cv2.circle(face_img, (int(point[0]), int(point[1])), 2, (0, 255, 0),
                   -1, cv2.LINE_AA)

        cv2.imwrite(
            os.path.join('landmark_image',
                         image.split('/')[-1]), face_img)


def main():
    rmtree(data_path + '_face')
    pool = mp.Pool()
    os.mkdir(data_path + '_face')
    counter = {'invalid_face': 0}

    for image in glob.glob(os.path.join(data_path, '*.jpg')):

        img = cv2.imread(image, cv2.IMREAD_UNCHANGED)
        if img is None:
            continue

        if data_path != 'face_images':

            #if there is face in an image
            try:
                locations = fr.face_locations(img)
            except RuntimeError:
                LogManager().error(
                    "{} is an unsupported image, type must be 8bit gray or RGB image"
                    .format(image))
                continue

            if len(locations) == 0:
                LogManager().info('There is no face in the {}!'.format(image))
                print('There is no face in the image!!')
                continue
            LogManager().info('Detecting faces in the {}!'.format(image))

            locs = []
            for loc in locations:
                start_x, start_y, end_x, end_y = loc[3], loc[0], loc[1], loc[2]
                locs.append((start_x, start_y, end_x, end_y))

            # crop from image
            cut_face_imgs = cut_face(img, locs)

            for face_img in cut_face_imgs:
                image_name = image.split('/')[-1].split('.')[0] + str(
                    uuid4())[:3]
                image_name = image_name + '.jpg'

                pool.apply_async(save_face, (face_img, image_name))

                if face_img.size == 0:
                    continue

                points = get_81_points(face_img, model_name)
                if points == None:
                    continue

                if points_are_valid(points, face_img) is False:
                    print('points are out of image')
                    LogManager().info(
                        "{}: points are out of image".format(image))
                    continue

                pool.apply_async(save_landmark, (
                    face_img,
                    points,
                    image_name,
                ))

        else:

            points = get_81_points(img, model_name)
            if points == None:
                continue

            if points_are_valid(points, img) is False:
                counter['invalid_face'] += 1
                print('points are out of image')
                LogManager().info("{}: points are out of image".format(image))
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            pool.apply_async(save_landmark, (
                img,
                points,
                image,
            ))


if __name__ == '__main__':
    main()