import cv2
import dlib
import numpy as np


def get_68_points(file_name=None):
    points = []
    with open(file_name) as file:
        line_count = 0
        for line in file:
            if "version" in line or "n_points" in line or "{" in line or "}" in line:
                continue
            else:
                px, py = line.strip().split()
                points.append((float(px), float(py)))
                line_count += 1
    return points


def draw_landmak_point(image, points):
    for point in points:
        cv2.circle(image, (int(point[0]), int(point[1])), 3, (0, 255, 0), -1,
                   cv2.LINE_AA)
        # cv2.rectangle(image, (locations[3], locations[0]),
        #               (locations[1], locations[2]), (0, 255, 0), 4,
        #               cv2.LINE_AA)


def get_81_points(image, model_path):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(model_path)
    face_dets = detector(image, 0)

    for i, d in enumerate(face_dets):
        shape = predictor(image, d)
        landmarks = [[p.x, p.y] for p in shape.parts()]
        return landmarks


def get_minimal_box(points):
    min_x = int(min([point[0] for point in points]))
    max_x = int(max([point[0] for point in points]))
    min_y = int(min([point[1] for point in points]))
    max_y = int(max([point[1] for point in points]))
    return [min_x, min_y, max_x, max_y]


def points_are_valid(points, image):
    min_box = get_minimal_box(points)
    if box_in_image(min_box, image):
        return True
    return False


def box_in_image(box, image):
    rows = image.shape[0]
    cols = image.shape[1]
    return box[0] >= 0 and box[1] >= 0 and box[2] <= cols and box[3] <= rows