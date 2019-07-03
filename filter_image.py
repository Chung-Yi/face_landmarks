import glob
import os


def main():
    buckets_image_files = glob.glob("landmark_image/*.jpg")
    image_files = glob.glob("face_images/*.jpg")
    images_name = []
    buckets_image_name = []

    for image_path in buckets_image_files:
        basename = os.path.basename(image_path)
        buckets_image_name.append(basename)
    s1 = set(buckets_image_name)

    for image_path in image_files:
        basename = os.path.basename(image_path)
        images_name.append(basename)
    s2 = set(images_name)

    s3 = s2.difference(s1)

    for image in s3:
        os.remove(os.path.join('face_images', image))


if __name__ == "__main__":
    main()