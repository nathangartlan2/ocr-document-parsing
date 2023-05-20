import read_input as rd
import data_load as dl
import models.tesseract_model as tm
import models.open_cv_model as ocv
import accuracy_tester

import keras
import tensorflow
from keras import layers
import sklearn as sk
from PIL import Image


width = 300
height = 300


def cleanseInference(inference_labels):
    cleansed = []
    for label in inference_labels:
        clean = label.replace("\n", "")
        cleansed.append(clean)
    return cleansed


def main():
    root_directory = "/Users/nathangartlan1/Documents/AA-Nathan/uchicago-mpcs-work/AppliedDataAnalysis/Project/Technical/iam_datasets"

    word_labels = dl.load_text_labels_image_paths(root_directory, True)
    text_to_numeric, numeric_to_text = dl.make_text_to_label_dict(
        word_labels=word_labels)

    images_and_labels_raw = dl.load_images_raw(
        word_labels,
        Image.open)

    tess = tm.tesseract_model()

    cv_model = ocv.open_cv_model(
        word_labels=word_labels,
        width=width,
        height=height
    )
    cv_model.preprocess()

    ground_truths, translations = tess.runTestInference(
        images=images_and_labels_raw[0][:100],
        labels=images_and_labels_raw[1][:100])

    CER = accuracy_tester.full_set_CER(
        ground_truths,
        cleanseInference(translations),
        accuracy_tester.character_error_rate
    )

    print("Tesseract CER = " + str(CER))

    print("Done!")


if __name__ == "__main__":
    main()
