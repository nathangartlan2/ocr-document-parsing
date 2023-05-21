import pandas as pd
import synthetic_data_generator as sn
import read_data as rd
import tesseract_model as ts
import open_cv_model as ocv
import character_image_generator as ch
import basic_keras as kr

import cv2

url = "https://raw.githubusercontent.com/scrollmapper/bible_databases/master/csv/t_kjv.csv"

outputPath = "/Users/nathangartlan1/Documents/AA-Nathan/uchicago-mpcs-work/AppliedDataAnalysis/Project/synthetic_data/"
mode = "CHARACTER"

WIDTH = 50
HEIGHT = 50

IS_DEMO = False
MODEL_DEMO = False


def main():

    if mode == "GENERATE_DATA":
        input_text = pd.read_csv(url)
        texts = input_text["t"]
        generator = sn.synthetic_data_generator(texts, outputPath)
        generator.generator_verse_pdf(100, 20)
    elif mode == "TESSERACT":

        labels, images = rd.read_data(outputPath)

        tesseract = ts.tesseract_model(labels, images)

        tesseract.inference()
    elif mode == "OPEN_CV":
        labels, images = rd.read_data(outputPath)

        ocv_model = ocv.open_cv_model(labels, images)
        ocv_model.preprocess_training()
    elif mode == "CHARACTER":
        print("Creating permutations")
        char_list = ch.create_permutations("ab", permute=True)

        print("Creating base images")
        characters, image_sets = ch.create_all_base_images(
            char_list, demo=IS_DEMO)

        print("Making distortions and copies")
        characters, image_sets = ch.create_distortions(
            characters, image_sets, num_repeats=5, demo=IS_DEMO)

        print("Making distortions and copies")
        labels, images = ch.label_all_images(characters, image_sets)

        print("Initializing model")
        keras_model = kr.basic_keras(
            images=images, labels=labels, width=WIDTH, height=HEIGHT, demo=MODEL_DEMO)

        print("Training model")
        keras_model.train()
        keras_model.plot_training()

        print("Time for the model")

    print("Äll done")


if __name__ == "__main__":
    main()
