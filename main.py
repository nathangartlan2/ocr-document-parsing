import read_input as rd
import data_load as dl


def combine_data_labels(label_dictionary, image_dictionary, label_type: str):
    data_types = ["OCR:\n", "CSR:\n"]

    for key_directory in image_dictionary.keys():
        data = image_dictionary[key_directory]
        labels = label_dictionary[key_directory]

        extracted_lined = extractLinedata(labels, label_type)


def extractLinedata(label_lines, label_type: str):
    data_types = ["OCR:\n", "CSR:\n"]

    sub_dictionary = {}
    section = ""
    linelist = []
    for line in label_lines:
        if line in data_types:
            if section in data_types:
                sub_dictionary[section] = linelist
                linelist = []
            section = line.replace("\n", "")

        elif line != '\n':
            linelist.append(line.replace("\n", ""))
    sub_dictionary[section] = linelist

    return sub_dictionary[label_type]


def main():
    root_directory = "/Users/nathangartlan1/Documents/AA-Nathan/uchicago-mpcs-work/AppliedDataAnalysis/Project/Technical/iam_datasets"

    word_labels = dl.read_word_labels(root_directory)
    word_images = dl.load_and_label_word_images(word_labels)
    print("Data loaded")


if __name__ == "__main__":
    main()
