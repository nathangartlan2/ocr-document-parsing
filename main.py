import read_input as rd


def main():
    print("Hello World!")
    # get the directory ID for the folder you want to read
    acii_folder = '/Users/nathangartlan1/Documents/AA-Nathan/uchicago-mpcs-work/AppliedDataAnalysis/Project/Technical/iam_datasets/ascii'
    images_folder = '/Users/nathangartlan1/Documents/AA-Nathan/uchicago-mpcs-work/AppliedDataAnalysis/Project/Technical/iam_datasets/lineimages'

    extension = '.txt'

    labels_dictionary = rd.load_label_dictionary(acii_folder, 'CSR', extension)

    image_array_dictionary = rd.load_images_to_array(images_folder,  'tif')

    print("Data loaded")


data_dictionary = {}


if __name__ == "__main__":
    main()
