import os
import cv2
import glob
import face_recognition as fr
from utils import get_81_points, draw_landmak_point
from cv_core.detector.face_inference import SsdFaceLocationDetector
from cv_core.detector.face_score_inference import KerasFaceScoreInference

IMG_SIZE = 250
SSD_THR = 0.6
SCORE_THR = 0.5
BIN_NUM = 1
PTS = 81
path = os.path.abspath(os.path.dirname(__file__))
data_path = os.path.join(path, 'images')
model_name = os.path.join(path, 'models/shape_predictor_81_face_landmarks.dat')
save_path = './bin'


def cut_face(image, locations):
    face_imgs = []
    for loc in locations:
        start_y, end_x, end_y, start_x = loc
        face_img = image[start_y - 30:end_y + 30, start_x - 30:end_x + 30, :]
        face_imgs.append(face_img)
    return face_imgs


def fr_read_images(images, shape=None):
    face_images = []
    landmarks = []
    locs = []
    for image in glob.glob('test_image/*.jpg'):
        img = cv2.imread(image, cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # if there is face in an image
        locations = fr.face_locations(img)
        if len(locations) == 0:
            print('There is no face in the image!!')
            continue
        # crop from image
        cut_face_imgs = cut_face(img, locations)
        for i, face_img in enumerate(cut_face_imgs):
            if shape != None:
                assert isinstance(shape, int)
                face_img = cv2.resize(face_img, (shape, shape))
            points = get_81_points(face_img, model_name)
            assert len(points) == PTS
            face_images.append(face_img)
            landmarks.append(points)

    return face_images, landmarks


def relocations(locations):
    locs = []
    for loc in locations:
        locs.append((loc[1], loc[2], loc[3], loc[0]))
    return locs


def baseline(images, detector, score_predictor, shape=None):
    face_images = []
    landmarks = []
    for image in glob.glob('test_image/*.jpg'):
        img = cv2.imread(image, cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        locations = detector.predict(img, SSD_THR)
        locations = relocations(locations)

        if len(locations) == 0:
            print('There is no face in the image!!')
            continue

        cut_face_imgs = cut_face(img, locations)

        for face_img in cut_face_imgs:
            if face_img.size == 0:
                continue
            if SCORE_THR < score_predictor.predict(face_img)[0][1]:
                if shape != None:
                    assert isinstance(shape, int)
                    face_img = cv2.resize(face_img, (shape, shape))
                face_images.append(face_img)
                points = get_81_points(face_img, model_name)
                assert len(points) == PTS
                landmarks.append(points)

    return face_images, landmarks


# def pickeld(save_path, face_images, landmarks, bin_num=None):
#     total_num = len(face_images)
#     samples_per_bin = total_num / bin_num
#     assert samples_per_bin > 0
#     idx = 0
#     for i in range(bin_num):
#         start = int(i * samples_per_bin)
#         end = int((i+1) * samples_per_bin)

#         if end <= total_num:


def main():
    detector = SsdFaceLocationDetector()
    score_predictor = KerasFaceScoreInference()
    # face_images, landmarks = fr_read_images(data_path, shape=IMG_SIZE)
    face_images, landmarks = baseline(
        data_path, detector, score_predictor, shape=IMG_SIZE)
    # pickeld(save_path, face_images, landmarks, 1)
    # print(len(locations[1]))
    for i, face_img in enumerate(face_images):
        draw_landmak_point(face_img, landmarks[i])
        cv2.imshow('My Image', face_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()