import pandas as pd
import synthetic_data_generator as sn
import read_data as rd
import tesseract_model as ts
import open_cv_model as ocv


def main():
    url = "https://raw.githubusercontent.com/scrollmapper/bible_databases/master/csv/t_kjv.csv"

    outputPath = "/Users/nathangartlan1/Documents/AA-Nathan/uchicago-mpcs-work/AppliedDataAnalysis/Project/synthetic_data/"
    mode = "OPEN_CV"

    if mode == "GENERATE_DATA":
        input_text = pd.read_csv(url)
        texts = input_text["t"]
        generator = sn.synthetic_data_generator(texts, outputPath)
        generator.generator_verse_pdf(100)
    elif mode == "TESSERACT":

        labels, images = rd.read_data(outputPath)

        tesseract = ts.tesseract_model(labels, images)

        tesseract.inference()
    elif mode == "OPEN_CV":
        labels, images = rd.read_data(outputPath)

        ocv_model = ocv.open_cv_model(labels, images)
        ocv_model.preprocess()

    print("Äll done")


if __name__ == "__main__":
    main()
