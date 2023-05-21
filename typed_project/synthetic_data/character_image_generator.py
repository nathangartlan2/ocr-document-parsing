import string
import cv2

from fpdf import FPDF
import os
import numpy as np
import itertools


SUPPORTED_CHARS = string.ascii_letters + \
    string.digits + string.punctuation + " "


def create_permutations(supported_characters: string):
    allChars = list(supported_characters)
    pairlist = list(itertools.combinations(supported_characters, 2))

    pairStrings = []
    for pair in pairlist:
        pairStrings.append(pair[0] + pair[1])

    allChars = allChars + pairStrings
    return allChars


def create_base_image(input: string, input_font: int,  demo: bool):
    width, height = 400, 200
    image = 255 * np.ones((height, width, 3), dtype=np.uint8)

    # Define the text properties
    text = input
    font = input_font
    font_scale = 1.5
    font_color = (0, 0, 0)  # Black color
    thickness = 2
    (text_width, text_height), _ = cv2.getTextSize(
        text, font, font_scale, thickness)

    # Calculate the position to center the text on the image
    x = (width - text_width) // 2
    y = (height + text_height) // 2

    # Draw the text on the image
    cv2.putText(image, text, (x, y), font, font_scale, font_color, thickness)

    if(demo):
        # show images for demo
        cv2.imshow("image", image)
        cv2.waitKey(100)
    # Return resulting image
    return image


def create_base_images(input_char: string, demo: bool):
    base_images = []

    base_images.append(create_base_image(
        input_char, cv2.FONT_HERSHEY_COMPLEX, demo))
    base_images.append(create_base_image(
        input_char, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, demo))
    base_images.append(create_base_image(
        input_char, cv2.FONT_HERSHEY_SIMPLEX, demo))
    base_images.append(create_base_image(
        input_char, cv2.FONT_HERSHEY_TRIPLEX, demo))

    return base_images


def create_all_base_images(all_chars, demo: bool):
    labels = []
    image_set = []

    for character in all_chars:
        labels.append(character)
        image_set.append(create_base_images(character, demo))

    return labels, image_set


def label_all_images(characters, image_arrays):
    all_labels = []
    images_flat = []

    for i in range(0, len(characters)):
        for image in image_arrays[i]:
            all_labels.append(characters[i])
            images_flat.append(image)
    return all_labels, images_flat
