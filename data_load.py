import os
import tensorflow as tf
import numpy as np
import pandas as pd
import keras


def read_word_labels(directory_path: str, use_mini: bool):
    dataset = []

    word_file = "ascii/words.txt"

    if use_mini:
        word_file = "ascii/words_mini.txt"

    f = os.path.join(directory_path, word_file)

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
            color_mode="grayscale",
            target_size=(180, 180)
        )
        image_array = tf.keras.utils.img_to_array(img)
        images.append(image_array)
        labels.append(path_label[1])

    image_tensor = tf.convert_to_tensor(np.array(images), dtype=tf.float32)
    labels_tensor = tf.convert_to_tensor(np.array(labels), dtype=tf.string)

    data_set = tf.data.Dataset.from_tensor_slices(
        (images, labels))
    return data_set
