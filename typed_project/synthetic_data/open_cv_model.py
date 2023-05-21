import cv2
import read_data as rd
import PIL as Image
import cv2
import pytesseract as pt
import time


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
                                                 5))

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

            border = 10  # Adjust the border size as needed
            # x -= border
            # y -= border
            # w += 2 * border
            # h += 2 * border
            # y = int(y * .9)
            # h = int(h * 1.25)
            # Drawing a rectangle on copied image
            # rect = cv2.rectangle(
            #     im2, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Cropping the text block for giving input to OCR
            cropped = im2[y:y + h, x:x + w]
            image_lines.append(cropped)

        all_images.append(image_lines)
    return labels, all_images


def process_characters(labels, images):
    all_images_as_chars = []

    for i in range(0, len(images)):
        img_line_array = images[i]
        label = labels[i]
        individual_char_images = []

        for j in range(0, len(img_line_array)):
            img = img_line_array[j]

            # cv2.imshow('line_image', img)
            # cv2.waitKey(1000)
            # make initial bounding boxes be the height of the
            # image to that the dots in "i" and "j" are not identified
            # by their contours

            height = img.shape[0]
            width = img.shape[1]
            rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                                                    (1,
                                                     height))

            # grey out image
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            ret, thresh1 = cv2.threshold(
                gray, 125, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

            # Applying dilation on the threshold image
            dilation = cv2.dilate(thresh1, rect_kernel, iterations=1)

            # cv2.imshow('dilation', dilation)
            # cv2.waitKey(500)
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

                # cv2.imshow('withrect', im2)
                # Cropping the text block for giving input to OCR
                cropped = im2[y:y + h, x:x + w]
                # cv2.imshow('cropped', cropped)

                # cv2.imshow('current_segment', cropped)
                # cv2.waitKey(100)
                individual_char_images.append(cropped)
            all_images_as_chars.append(individual_char_images)
    return labels, all_images_as_chars


def match_chars_to_image(labels, individual_char_images, ignore_spaces: bool):
    num_mismatch = 0

    for i in range(0, len(labels)):
        label = labels[i]
        char_images = individual_char_images[i]

        label = list(filter(lambda element: element != "\n", label))

        if(ignore_spaces):
            label = list(filter(lambda element: element != " ", label))
        if(len(label) != len(char_images)):
            num_mismatch += 1

        for char_img in char_images:
            cv2.imshow("bible", char_img)
            cv2.waitKey(1000)

    print("Total: " + str(len(labels)))
    print("Mismatched: " + str(num_mismatch))
    print("Done")


class open_cv_model():
    def __init__(self, labels, images) -> None:
        self.labels = labels
        self.images = images

    def preprocess_training(self):
        print("Begin segmenting lines")
        labels, images_as_lines = preprocess_lines(self.labels, self.images)

        print("Begin segmenting charcters")
        labels, individual_char_images = process_characters(
            labels, images_as_lines)

        match_chars_to_image(labels, individual_char_images, True)
