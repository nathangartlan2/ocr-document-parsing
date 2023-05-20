from PIL import Image

import pytesseract
import cv2


def printText(image_path):
    img_cv = cv2.imread(image_path[0])
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
    print("model = " + pytesseract.image_to_string(img_rgb, config='--psm 6')
          + "\nlabel = " + str(image_path[1]))
