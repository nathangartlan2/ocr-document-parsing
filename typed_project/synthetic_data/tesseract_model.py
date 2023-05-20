from PIL import Image

import pytesseract
import cv2
import read_data as rd

from PIL import Image

import pytesseract
import cv2
import os
from pdf2image import convert_from_path
import numpy as np
import read_data as rd


class tesseract_model():
    def __init__(self, labels, images) -> None:
        self.labels = labels
        self.images = images

    def inference(self):
        for img in self.images:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            print("model = " + pytesseract.image_to_string(img_rgb, config='--psm 6'))


# def printText(image_path):
#     img_cv = cv2.imread(image_path[0])
#     img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
#     print("model = " + pytesseract.image_to_string(img_rgb, config='--psm 6')
#           + "\nlabel = " + str(image_path[1]))
