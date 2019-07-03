import glob
import os
from shutil import copyfile, move
from random import shuffle


def make_bucket(image_list, bucket_size=500):
    for i in range(0, len(image_list), bucket_size):
        yield image_list[i:i + bucket_size]


def main():
    image_files = glob.glob('landmark_image/*.jpg')
    shuffle(image_files)
    buckets = list(make_bucket(image_files, 500))
    for i, bucket in enumerate(buckets):
        os.mkdir("buckets/index" + str(i))
        for image_path in bucket:
            basename = os.path.basename(image_path)
            move(image_path, 'buckets/index' + str(i))


if __name__ == "__main__":
    main()