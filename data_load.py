import os
import tensorflow as tf
import numpy as np
import pandas as pd


def read_word_labels(directory_path: str):
    dataset = []
    f = os.path.join(directory_path, "ascii/words.txt")

    if os.path.isfile(f):
        with open(f) as reader:
            lines = reader.readlines()
            for line in lines:
                if line.startswith("#"):
                    continue

                line_split = line.split(" ")
                if line_split[1] == "err":
                    continue

                folder1 = line_split[0][:3]
                folder2 = line_split[0][:8]
                file_name = line_split[0] + ".png"
                label = line_split[-1].rstrip('\n')

                rel_path = os.path.join(
                    directory_path, "words", folder1, folder2, file_name)
                if not os.path.exists(rel_path):
                    continue
                dataset.append([rel_path, label])
    return dataset


def load_and_label_word_images(word_labels: list):
    images = []
    labels = []
    for path_label in word_labels:
        img = tf.keras.utils.load_img(
            path_label[0],
            grayscale=True,
            target_size=(180, 180)
        )
        image_array = tf.keras.utils.image.img_to_array(img)
        images.append(image_array)
        labels.append(path_label[1])

    np_image = np.array(images)
    np_labels = np.array(labels)

    return tf.data.Dataset.from_tensor_slices((np_image, np_labels))
