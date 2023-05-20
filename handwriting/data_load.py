import os
import tensorflow as tf
import numpy as np
import pandas as pd
import keras
import tesseract_wrapper as tw


def load_text_labels_image_paths(directory_path: str, use_mini: bool):
    text_labels = []

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
                text_labels.append([rel_path, label])
    return text_labels


def loadl_label_images_tensors(word_labels, text_to_numeric: dict):
    images = []
    labels = []

    for path_label in word_labels:
        img = tf.keras.utils.load_img(
            path_label[0],
            color_mode="grayscale",
            target_size=(300, 300)
        )
        image_array = tf.keras.utils.img_to_array(img)
        images.append(image_array)
        labels.append(text_to_numeric[path_label[1]])

    data_set = tf.data.Dataset.from_tensor_slices(
        (images, labels))
    return data_set


def load_label_images_simple(word_labels, text_to_numeric: dict):
    images = []
    labels = []

    for path_label in word_labels:
        img = tf.keras.utils.load_img(
            path_label[0],
            color_mode="grayscale",
            target_size=(300, 300)
        )
        image_array = tf.keras.utils.img_to_array(img)
        images.append(image_array)
        labels.append(text_to_numeric[path_label[1]])

    return images, labels


def make_text_to_label_dict(word_labels):
    text_to_numeric = {}
    numeric_to_text = {}
    label_address = 0

    unique_text = set()
    for label in word_labels:
        unique_text.add(label[1])

    for text in unique_text:
        text_to_numeric[text] = label_address
        numeric_to_text[label_address] = text
        label_address += 1
    return text_to_numeric, numeric_to_text


def convert_text_to_numeric(texts, words_to_labels: dict):
    numeric_labels = []

    for text in texts:
        numeric_labels.append(words_to_labels[text])

    return numeric_labels


def load_images_raw(word_labels,
                    load_function):
    images = []
    labels = []

    for path_label in word_labels:
        img = load_function(path_label[0])
        images.append(img)
        labels.append([path_label[1]])

    return images, labels
