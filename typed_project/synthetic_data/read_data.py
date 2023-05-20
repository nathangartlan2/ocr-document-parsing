import os
from pdf2image import convert_from_path
from PIL import Image
import cv2
import numpy as np


def read_labels(directory_path: str, extension: str):
    full_directory = directory_path + extension

    f = os.path.join(full_directory, "manifest.txt")
    labels = []
    paths = []

    if os.path.isfile(f):
        with open(f) as reader:
            lines = reader.readlines()
            for line in lines:
                line_split = line.split(" : ")
                labels.append(line_split[1])
                paths.append(os.path.join(full_directory, "/" + line_split[0]))

    return paths, labels


def load_images_raw(
        dir,
        paths,
        labels,
        load_function):
    images = []

    for filename in paths:
        file = dir + filename
        img = load_function(file)
        images.append(img)

    return images, labels


def load_single_image(path):
    pages = convert_from_path(path)

    # Iterate over the pages and process each one
    for i, page in enumerate(pages):
        # Convert PIL image to OpenCV format
        cv_image = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
        return cv_image


def read_data(dir: str):
    paths, labels = read_labels(dir, "verses")

    f = os.path.join(dir, "verses")

    images, labels = load_images_raw(
        f,
        paths,
        labels,
        load_single_image
    )

    return labels, images
