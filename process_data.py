import os
import cv2
import glob
import _pickle as cPickle
import face_recognition as fr
from utils import *
from log_manager import LogManager
from cv_core.detector.face_inference import SsdFaceLocationDetector
from cv_core.detector.face_score_inference import KerasFaceScoreInference
from argparse import ArgumentParser
from uuid import uuid4

parser = ArgumentParser()
parser.add_argument(
    '--folder_path', default="face_images", help='choose a image folder')
args = parser.parse_args()

IMG_SIZE = 200
MIN_BLUR = 50
SSD_THR = 0.6
SCORE_THR = 0.5
BIN_NUM = 1
PTS = 81
path = os.path.abspath(os.path.dirname(__file__))
data_path = args.folder_path
model_name = os.path.join(path, 'models/shape_predictor_81_face_landmarks.dat')
save_path = 'bin'
save_file_name = 'train_image'


def normalize_points(image, points):
    width = image.shape[1]
    height = image.shape[0]

    for point in points:
        point[0] /= width
        point[1] /= height
    return points


def fr_read_images(data_path, shape=None):
    face_images = []
    fnames = []
    landmarks = []
    locs = []
    counter = {'invalid_face': 0}
    for image in glob.glob(os.path.join(data_path, '*.jpg')):

        img = cv2.imread(image, cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        print(image)
        print(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # points = get_81_points(img, model_name)
        # try:
        #     assert len(points) == PTS
        # except:
        #     print("face landmarks is not 81")
        #     continue

        # for point in points:
        #     cv2.circle(img, (int(point[0]), int(point[1])), 2, (0, 255, 0), -1,
        #                cv2.LINE_AA)

        # cv2.imwrite(os.path.join('landmark_image', image.split('/')[-1]), img)

        if data_path != 'face_images':
            #if there is face in an image
            try:
                locations = fr.face_locations(img)
            except RuntimeError:
                LogManager.error(
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
                image_name = image.split('/')[-1] + uuid4()[:3]

                if face_img.size == 0:
                    continue
                # filter blurry face
                if variance_of_laplacian(face_img) < MIN_BLUR:
                    continue

                if shape != None:
                    assert isinstance(shape, int)
                    face_img = cv2.resize(face_img, (shape, shape))
                points = get_81_points(face_img, model_name)
                try:
                    assert len(points) == PTS
                except:
                    print("face landmarks is not 81")
                    continue
                if points_are_valid(points, face_img) is False:
                    counter['invalid_face'] += 1
                    print('points are out of image')
                    continue

                for point in points:
                    cv2.circle(face_img, (int(point[0]), int(point[1])), 2,
                               (0, 255, 0), -1, cv2.LINE_AA)

                cv2.imwrite(
                    os.path.join('landmark_image',
                                 image.split('/')[-1]), face_img)
                points = normalize_points(face_img, points)
                face_images.append(face_img)
                landmarks.append(points)
                fnames.append(image_name)

        else:
            points = get_81_points(img, model_name)
            try:
                assert len(points) == PTS
            except:
                print("face landmarks is not 81")
                continue

            for point in points:
                cv2.circle(img, (int(point[0]), int(point[1])), 3, (0, 255, 0),
                           -1, cv2.LINE_AA)

                cv2.imwrite(
                    os.path.join('landmark_image',
                                 image.split('/')[-1]), img)
                # points = normalize_points(img, points)
                # face_images.append(img)
                # landmarks.append(points)
            fnames.append(image.split('/')[-1])

    LogManager.info('invalid_face:{}'.format(counter.values()))
    face_images = np.array(face_images)
    landmarks = np.array(landmarks)

    return face_images, landmarks, fnames


def baseline(images, detector, score_predictor, shape=None):
    face_images = []
    landmarks = []
    counter = {'invalid_face': 0}
    for image in glob.glob('ii/*.jpg'):
        img = cv2.imread(image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        locations = detector.predict(img, SSD_THR)
        try:
            locations = detector.predict(img, SSD_THR)
            LogManager().info(image + " finish detecting locations")
        except:
            LogManager().error(image + " fail!!!")
            continue

        if len(locations) == 0:
            print('There is no face in the image!!')
            continue

        cut_face_imgs = cut_face(img, locations)

        for face_img in cut_face_imgs:

            if face_img.size == 0:
                continue

            # filter blurry face
            if variance_of_laplacian(face_img) < MIN_BLUR:
                continue

            # filter low face score out
            if SCORE_THR < score_predictor.predict(face_img)[0][1]:
                if shape != None:
                    assert isinstance(shape, int)
                    face_img = cv2.resize(face_img, (shape, shape))

                face_images.append(face_img)
                points = get_81_points(face_img, model_name)
                try:
                    assert len(points) == PTS
                except:
                    continue

                # check if points are there in image
                if points_are_valid(points, face_img) is False:
                    counter['invalid_face'] += 1
                    continue

                points = normalize_points(face_img, points)

                landmarks.append(points)

    face_images = np.array(face_images)
    landmarks = np.array(landmarks)

    return face_images, landmarks


def pickeld(save_path,
            save_file_name,
            face_images,
            landmarks,
            fnames,
            bin_num=None):
    assert os.path.isdir(save_path)
    total_num = len(face_images)
    samples_per_bin = total_num / bin_num
    assert samples_per_bin > 0
    for i in range(bin_num):
        start = int(i * samples_per_bin)
        end = int((i + 1) * samples_per_bin)
        if end <= total_num:
            dic = {
                'data': face_images[start:end, :],
                'landmarks': landmarks[start:end],
                'filenames': fnames[start:end]
            }
        else:
            dic = {
                'data': face_images[start:, :],
                'landmarks': landmarks[start:],
                'filenames': fnames[start:]
            }

        with open(os.path.join(path, save_path, save_file_name), 'wb') as f:
            cPickle.dump(dic, f)


def main():
    detector = SsdFaceLocationDetector()
    score_predictor = KerasFaceScoreInference()
    face_images, landmarks = fr_read_images(data_path, shape=IMG_SIZE)
    # face_images, landmarks = baseline(
    #     data_path, detector, score_predictor, shape=IMG_SIZE)

    # pickeld(save_path, save_file_name, face_images, landmarks, fnames, 1)

    # for i, face_img in enumerate(face_images):
    #     draw_landmak_point(face_img, [[px * IMG_SIZE, py * IMG_SIZE]
    #                                   for px, py in landmarks[i]])

    #     cv2.imwrite('./landmark_image/{}.jpg'.format(i), face_img)
    #     cv2.imshow('My Image', face_img)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()


if __name__ == "__main__":
    main()