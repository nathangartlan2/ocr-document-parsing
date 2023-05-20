import cv2
import models.model_base as base
import data_load as dl
import PIL as Image
import cv2
import pytesseract as pt


class open_cv_model(base.model_base):
    def __init__(self,
                 word_labels,
                 width,
                 height):

        images, labels = dl.load_images_raw(
            word_labels,
            cv2.imread
        )

        self.images = images
        self.labels = labels
        self.width = width
        self.height = height
        self.name = "open_cv_with_contour"

    def preprocess(self):
        for i in range(0, len(self.images)):
            img = self.images[i]
            label = self.labels[i]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, thresh1 = cv2.threshold(
                gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

            # make initial bounding boxes be the height of the
            # image to that the dots in "i" and "j" are not identified
            # by their contours

            height = img.shape[0]
            width = img.shape[1]
            rect_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                    (int(5),
                                                     int(5)))

            # Applying dilation on the threshold image
            dilation = cv2.dilate(thresh1, rect_kernel, iterations=1)

            # Finding contours
            contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL,
                                                   cv2.CHAIN_APPROX_NONE)

            # Creating a copy of image
            im2 = img.copy()

            number_of_countours = len(contours)

            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                with_countours = cv2.drawContours(
                    img, contours, -1, (0, 255, 0), 3)
                # Drawing a rectangle on copied image
                rect = cv2.rectangle(
                    im2, (x, y), (x + w, y + h), (0, 255, 0), 2)

                cv2.imshow('with_countours', with_countours)
                # Cropping the text block for giving input to OCR
                cropped = im2[y:y + h, x:x + w]

                print(pt.image_to_string(cropped))
                cv2.imshow('current_segment', cropped)
                cv2.waitKey(2000)
            print(self.labels[i])

    def label_image_contours(image, countours, label):
        cropped_images = []

    def train(self, x_train, x_val, y_train, y_val):
        pass

    def plot_training(self):
        pass

    def runTestInference(self, images, labels):
        pass

    def getName(self):
        return self.name
