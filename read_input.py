import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import math

import pandas as pd

import itertools

import sklearn.linear_model as skl_lm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression

from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import Ridge, RidgeCV

import statsmodels.api as sm
import matplotlib.pyplot as plt

import os
import cv2

import tensorflow as tf
import keras as k

import ImageInput as CImg


def load_dat_by_line(filePath, data_type, fileExtension):
    data_types = ["OCR:\n", "CSR:\n"]

    data_dictionary = {}

    for dir1_name in os.listdir(filePath):
        filePath1 = os.path.join(filePath, dir1_name)
        if not os.path.isdir(filePath1):
            continue

        for dir2_name in os.listdir(filePath1):
            file2Path = os.path.join(filePath1, dir2_name)

            if not os.path.isdir(file2Path):
                continue

            for filename in os.listdir(file2Path):
                f = os.path.join(file2Path, filename)

                if os.path.isfile(f):
                    with open(f) as reader:
                        lines = reader.readlines()
                    addLines = False
                    key = filename.replace(fileExtension, "")

                    line_count = 1
                    for line in lines:
                        string_count = str(line_count)
                        if(line_count < 10):
                            string_count = "0" + string_count

                        if line in data_types:
                            if data_type in line:
                                addLines = True
                            else:
                                addLines = False
                        else:
                            if addLines == True and line != '\n':
                                unique_key = str(key) + "_" + string_count
                                data_dictionary[unique_key] = line
                                line_count += 1

    return data_dictionary


def load_lines_by_directory(filePath: str, fileExtension: str):
    data_dictionary = {}

    for dir1_name in os.listdir(filePath):
        filePath1 = os.path.join(filePath, dir1_name)
        if not os.path.isdir(filePath1):
            continue

        for dir2_name in os.listdir(filePath1):
            file2Path = os.path.join(filePath1, dir2_name)
            folder_dictionary = {}

            if not os.path.isdir(file2Path):
                continue

            for filename in os.listdir(file2Path):
                f = os.path.join(file2Path, filename)

                if os.path.isfile(f):
                    with open(f) as reader:
                        lines = reader.readlines()
                    key = filename.replace(fileExtension, "")
                    folder_dictionary[key] = lines

            data_dictionary[dir2_name] = folder_dictionary
    return data_dictionary


def load_label_dictionary(filePath: str, data_type: str, fileExtension: str):
    data_types = ["OCR:\n", "CSR:\n"]

    data_dictionary = {}

    for dir1_name in os.listdir(filePath):
        filePath1 = os.path.join(filePath, dir1_name)
        if not os.path.isdir(filePath1):
            continue

        for dir2_name in os.listdir(filePath1):
            file2Path = os.path.join(filePath1, dir2_name)

            if not os.path.isdir(file2Path):
                continue

            for filename in os.listdir(file2Path):
                f = os.path.join(file2Path, filename)

                if os.path.isfile(f):
                    with open(f) as reader:
                        lines = reader.readlines()
                    key = filename.replace(fileExtension, "")
                    sub_dictionary = {}
                    section = ""
                    linelist = []
                    for line in lines:
                        if line in data_types:
                            if section in data_types:
                                sub_dictionary[section] = linelist
                                linelist = []
                            section = line.replace("\n", "")

                        elif line != '\n':
                            linelist.append(line.replace("\n", ""))
                    sub_dictionary[section] = linelist
                    data_dictionary[key] = sub_dictionary

    return data_dictionary


def load_images_to_array(filePath,  fileExtension):

    data_dictionary = {}

    for dir1_name in os.listdir(filePath):
        filePath1 = os.path.join(filePath, dir1_name)
        if not os.path.isdir(filePath1):
            continue

        for dir2_name in os.listdir(filePath1):
            file2Path = os.path.join(filePath1, dir2_name)

            if not os.path.isdir(file2Path):
                continue

            for filename in os.listdir(file2Path):
                f = os.path.join(file2Path, filename)

                if os.path.isfile(f):
                    img = process_image(f)
                    key = filename.replace(fileExtension, "")
                    image_array = tf.keras.preprocessing.image.img_to_array(
                        img)
                    data_dictionary[key] = image_array

    return data_dictionary


def process_image(filepath: str):
    img = tf.keras.preprocessing.image.load_img(
        filepath,
        color_mode="grayscale",
        target_size=(200, 200),
        interpolation="nearest"
    )
    return img


def load_image_by_directory(filePath,  fileExtension):

    data_dictionary = {}

    for dir1_name in os.listdir(filePath):
        filePath1 = os.path.join(filePath, dir1_name)
        if not os.path.isdir(filePath1):
            continue

        for dir2_name in os.listdir(filePath1):
            file2Path = os.path.join(filePath1, dir2_name)
            image_list = []

            if not os.path.isdir(file2Path):
                continue

            for filename in os.listdir(file2Path):
                f = os.path.join(file2Path, filename)

                if os.path.isfile(f):
                    img = process_image(f)
                    key = filename.replace(fileExtension, "")
                    wrapperObject = CImg.ImageInput(key, img)
                    image_list.append(wrapperObject)
            data_dictionary[dir2_name] = image_list

    return data_dictionary
