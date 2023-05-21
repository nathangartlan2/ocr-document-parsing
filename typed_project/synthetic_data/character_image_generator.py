import string
import cv2

from fpdf import FPDF
import os
import numpy as np
import itertools

# all English characters supported by this program
SUPPORTED_CHARS = string.ascii_letters + \
    string.digits + string.punctuation + " "

# creates permutations of character data
# for example, if supported_characters = "abc" and permute = True
# this will return  ["a", "b", "c", "ab", "bc", "ac"]


def create_permutations(supported_characters: string, permute: bool):
    allChars = list(supported_characters)
    pairlist = list(itertools.combinations(supported_characters, 2))

    pairStrings = []

    if permute:
        for pair in pairlist:
            pairStrings.append(pair[0] + pair[1])

    allChars = allChars + pairStrings
    return allChars

# creates and image of a single character


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

# creates base images in multiple fonts


def create_base_images(input_char: string, demo: bool):
    base_images = []

    base_images.append(create_base_image(
        input_char, cv2.FONT_HERSHEY_COMPLEX, demo))
    base_images.append(create_base_image(
        input_char, cv2.FONT_HERSHEY_COMPLEX_SMALL, demo))
    base_images.append(create_base_image(
        input_char, cv2.FONT_HERSHEY_DUPLEX, demo))
    base_images.append(create_base_image(
        input_char, cv2.FONT_HERSHEY_PLAIN, demo))
    base_images.append(create_base_image(
        input_char, cv2.FONT_HERSHEY_SCRIPT_COMPLEX, demo))
    base_images.append(create_base_image(
        input_char, cv2.FONT_ITALIC, demo))
    base_images.append(create_base_image(
        input_char, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, demo))
    base_images.append(create_base_image(
        input_char, cv2.FONT_HERSHEY_SIMPLEX, demo))
    base_images.append(create_base_image(
        input_char, cv2.FONT_HERSHEY_TRIPLEX, demo))

    return base_images

# creates images for all_chars


def create_all_base_images(all_chars, demo: bool):
    labels = []
    image_set = []

    for character in all_chars:
        labels.append(character)
        image_set.append(create_base_images(character, demo))

    return labels, image_set

# apply character label to all images


def label_all_images(characters, image_arrays):
    all_labels = []
    images_flat = []

    for i in range(0, len(characters)):
        for image in image_arrays[i]:
            all_labels.append(characters[i])
            images_flat.append(image)
    return all_labels, images_flat


def rotate_image(image, demo: bool):
    rotation_angles = [15, 30, 45, 60, 75, 90, 105, 120, 135, 150,
                       165, 180, 195, 210, 225, 240, 255, 270, 285, 300, 315, 330, 345]

    rotated_images = []

    # Get the image dimensions
    height, width = image.shape[:2]

    for angle in rotation_angles:
        # Calculate the rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(
            (width / 2, height / 2), angle, 1)

        # Perform the rotation
        rotated_image = cv2.warpAffine(
            image, rotation_matrix, (width, height), borderValue=(255, 255, 255))

        # Display the rotated image
        if(demo):
            cv2.imshow("Rotated Image", rotated_image)
            cv2.waitKey(200)
        rotated_images.append(rotated_image)

    return rotated_images


def create_distortions(labels, image_sets,
                       num_repeats: int,  demo: bool):
    expaned_sets = []

    for i in range(0, len(labels)):
        label = labels[i]
        image_set = image_sets[i]
        set_with_rotated = []
        for image in image_set:
            set_with_rotated += rotate_image(image, demo)

        set_array = np.array(set_with_rotated)
        set_array = np.repeat(
            set_array, repeats=num_repeats, axis=0)
        expaned_sets.append(set_array)

    # Copy each element of the array a number of times

    return labels, expaned_sets
