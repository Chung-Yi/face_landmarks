import pandas as pd
import os
import cv2
import glob
from uuid import uuid4

path = os.path.abspath(os.path.dirname(__file__))
data_path = os.path.join(path, 'images')


def crop_image(image, location):

    start_x, start_y, end_x, end_y = location
    face_img = image[start_y - 50:end_y + 50, start_x - 50:end_x + 50, :]

    return face_img


def main():

    df = pd.read_csv('filter_20190618.csv')
    is_face = df['type'] == 2
    is_available = df['available'] == 1
    df_face = df[is_face & is_available]

    for i, img_info in df_face.iterrows():

        img_name = img_info['name']
        start_x = img_info['start_x']
        start_y = img_info['start_y']
        end_x = img_info['end_x']
        end_y = img_info['end_y']

        location = (start_x, start_y, end_x, end_y)
        image = cv2.imread(os.path.join('images', img_name))
        face = crop_image(image, location)

        if not os.path.isfile('./face_images' + img_name):
            cv2.imwrite('./face_images/' + img_name, face)
        else:
            cv2.imwrite('./face_images/' + img_name + uuid4()[:3], face)


if __name__ == "__main__":
    main()