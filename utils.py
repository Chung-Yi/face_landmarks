import cv2
import dlib
import numpy as np


def cut_face(image, locations):
    face_imgs = []
    delta_locs = []
    height = image.shape[0]
    width = image.shape[1]

    for loc in locations:
        start_x, start_y, end_x, end_y = loc
        delta = 40

        new_start_y = start_y - delta
        new_end_y = end_y + delta
        new_start_x = start_x - delta
        new_end_x = end_x + delta

        if new_start_x <= 0:
            new_start_x += 10

        if new_start_y <= 0:
            new_start_y += 10

        if new_end_x >= width:
            new_end_x -= 10

        if new_end_y >= height:
            new_end_y -= 10

        face_img = image[new_start_y:new_end_y, new_start_x:new_end_x, :]
        face_imgs.append(face_img)

        delta_locs.append((abs(new_start_x - start_x),
                           abs(new_start_y - start_y)))

    return face_imgs, delta_locs


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


def scaled_points(image, points):
    width_scale = image.shape[1] / 200
    height_scale = image.shape[0] / 200
    for point in points:
        point[0] *= 200 * width_scale
        point[1] *= 200 * height_scale
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


def variance_of_laplacian(image):
    # blur detect
    return cv2.Laplacian(image, cv2.CV_64F).var()