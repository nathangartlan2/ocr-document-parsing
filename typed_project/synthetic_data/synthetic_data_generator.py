from fpdf import FPDF
import os
import pandas as pd


class synthetic_data_generator():
    def __init__(self, training_data, output_root_directory):
        self.training_data = training_data
        self.output_root_directory = output_root_directory

    def generator_verse_pdf(self, number_verses, font_size: int):
        num = number_verses
        if number_verses == 0:
            num = self.training_data.size[1]

        manifest = []
        dir_path = str(self.output_root_directory) + "verses/"
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        line_array = []

        for i in range(0, num):
            pdf = FPDF()
            # Add a page
            pdf.add_page()

            # set style and size of font
            # that you want in the pdf
            pdf.set_font("Times", size=font_size)

            # create a cell
            pdf.multi_cell(0, 10, txt=self.training_data[i], align='L')
            manifest.append([i, self.training_data[i]])
            file_path = dir_path + "verse_" + str(i) + ".pdf"
            pdf.output(file_path)
            line_str = "verse_" + str(i) + ".pdf" + \
                " : " + str(self.training_data[i])
            line_array.append(line_str)

        with open(dir_path + 'manifest.txt', 'w') as file:
            for string in line_array:
                file.write(string + '\n')
