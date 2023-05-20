import cv2
import read_data as rd
import PIL as Image
import cv2
import pytesseract as pt


def preprocess_lines(labels, images):

    all_images = []

    for i in range(0, len(images)):
        img = images[i]
        label = labels[i]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thresh1 = cv2.threshold(
            gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

        # make initial bounding boxes be the height of the
        # image to that the dots in "i" and "j" are not identified
        # by their contours

        height = img.shape[0]
        width = img.shape[1]

        # make minimum box width be width of the page
        # so there is only one box per line
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                                                (width,
                                                 1))

        # Applying dilation on the threshold image
        dilation = cv2.dilate(thresh1, rect_kernel, iterations=1)

        # Finding contours
        contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_NONE)

        # Sort countours from top to bottom
        contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[1])

        # Creating a copy of image
        im2 = img.copy()

        number_of_countours = len(contours)

        image_lines = []

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)

            # Drawing a rectangle on copied image
            rect = cv2.rectangle(
                im2, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.imshow('withrect', rect)
            # Cropping the text block for giving input to OCR
            cropped = im2[y:y + h, x:x + w]

            image_lines.append(cropped)

        all_images.append(image_lines)
    return labels, image_lines


def process_characters(labels, images):
    for i in range(0, len(images)):
        img = images[i]
        label = labels[i]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thresh1 = cv2.threshold(
            gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

        # make initial bounding boxes be the height of the
        # image to that the dots in "i" and "j" are not identified
        # by their contours

        height = img.shape[0]
        width = img.shape[1]
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                (int(2),
                                                 int(50)))

        # Applying dilation on the threshold image
        dilation = cv2.dilate(thresh1, rect_kernel, iterations=1)

        # Finding contours
        contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_NONE)

        # Sort countours from left to right
        contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[:2])

        # Creating a copy of image
        im2 = img.copy()

        number_of_countours = len(contours)

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)

            # Drawing a rectangle on copied image
            rect = cv2.rectangle(
                im2, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.imshow('withrect', rect)
            # Cropping the text block for giving input to OCR
            cropped = im2[y:y + h, x:x + w]

            print(pt.image_to_string(cropped))
            # cv2.imshow('current_segment', cropped)
            cv2.waitKey(500)
        print(labels[i])


class open_cv_model():
    def __init__(self, labels, images) -> None:
        self.labels = labels
        self.images = images

    def preprocess(self):
        preprocess_lines(self.labels, self.images)
